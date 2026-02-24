#!/usr/bin/env python3
"""
Daily Cost Report — sends yesterday's AI spend breakdown to Telegram at 8am.

Reads ~/.amplified/cost-log.jsonl and produces a structured summary:
  - Total spend with currency conversion (USD → GBP)
  - Breakdown by model
  - Breakdown by tool/agent
  - Cache hit rate (zero-cost calls)
  - Model routing savings (Haiku calls that would have been Sonnet)
  - Trend vs 7-day average

Scheduled via launchd StartCalendarInterval — runs once daily at 08:00.
Install: python3 daily_cost_report.py --install-plist
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import date, timedelta, datetime

sys.path.insert(0, str(Path.home() / ".amplified"))
from amplified_secrets import secrets  # noqa: F401

COST_LOG = Path.home() / ".amplified/cost-log.jsonl"
VAULT_REPORTS = Path.home() / "Manual Library/real-vault/work/sessions/cost-reports"
GBP_RATE = 0.79  # approximate — good enough for daily alerting


def _load_entries(target_date: str) -> list:
    if not COST_LOG.exists():
        return []
    entries = []
    for line in COST_LOG.read_text().splitlines():
        try:
            e = json.loads(line)
            if e.get("date") == target_date:
                entries.append(e)
        except Exception:
            pass
    return entries


def _load_period(days: int) -> list:
    cutoff = (date.today() - timedelta(days=days)).isoformat()
    if not COST_LOG.exists():
        return []
    entries = []
    for line in COST_LOG.read_text().splitlines():
        try:
            e = json.loads(line)
            if e.get("date", "") >= cutoff:
                entries.append(e)
        except Exception:
            pass
    return entries


def _summarise(entries: list) -> dict:
    total = 0.0
    by_model = {}
    by_tool = {}
    cache_hits = 0
    routing_saves = 0

    for e in entries:
        cost = e.get("cost_usd", 0)
        total += cost
        model = e.get("model", "unknown")
        tool = e.get("tool", "unknown")
        original = e.get("original_model", model)

        by_model[model] = by_model.get(model, 0) + cost
        by_tool[tool] = by_tool.get(tool, 0) + cost

        if e.get("from_cache"):
            cache_hits += 1
        if original != model and "haiku" in model.lower():
            routing_saves += 1

    return {
        "total_usd": round(total, 4),
        "total_gbp": round(total * GBP_RATE, 4),
        "calls": len(entries),
        "cache_hits": cache_hits,
        "routing_saves": routing_saves,
        "by_model": dict(sorted(by_model.items(), key=lambda x: x[1], reverse=True)),
        "by_tool": dict(sorted(by_tool.items(), key=lambda x: x[1], reverse=True)),
    }


def _format_telegram(yesterday: str, summary: dict, avg_7d_usd: float) -> str:
    total = summary["total_usd"]
    total_gbp = summary["total_gbp"]
    calls = summary["calls"]
    cache_hits = summary["cache_hits"]
    routing_saves = summary["routing_saves"]
    cache_rate = round(cache_hits / calls * 100, 1) if calls else 0
    routing_rate = round(routing_saves / calls * 100, 1) if calls else 0

    trend = ""
    if avg_7d_usd > 0:
        diff_pct = (total - avg_7d_usd) / avg_7d_usd * 100
        if diff_pct > 10:
            trend = f" ↑{diff_pct:.0f}% vs 7d avg"
        elif diff_pct < -10:
            trend = f" ↓{abs(diff_pct):.0f}% vs 7d avg"
        else:
            trend = " → steady"

    lines = [
        f"*Daily AI Cost — {yesterday}*",
        f"",
        f"💷 *£{total_gbp:.2f}* (${total:.2f}){trend}",
        f"📞 {calls} calls | 🎯 {cache_rate}% free (cache) | 🔀 {routing_rate}% Haiku-routed",
        f"",
        f"*By model:*",
    ]
    for model, cost in summary["by_model"].items():
        short = model.replace("claude-", "").replace("-20251001", "")
        pct = round(cost / total * 100, 0) if total else 0
        gbp = round(cost * GBP_RATE, 3)
        lines.append(f"  • {short}: £{gbp:.3f} ({pct:.0f}%)")

    lines += ["", "*By agent:*"]
    for tool, cost in summary["by_tool"].items():
        if cost < 0.0001:
            continue
        pct = round(cost / total * 100, 0) if total else 0
        gbp = round(cost * GBP_RATE, 3)
        lines.append(f"  • {tool}: £{gbp:.3f} ({pct:.0f}%)")

    if routing_saves > 0:
        # Rough saving estimate: each routed call saved ~$0.003 (avg)
        est_saved = round(routing_saves * 0.003 * GBP_RATE, 3)
        lines += ["", f"🏦 Model routing saved ~£{est_saved:.2f} today"]

    return "\n".join(lines)


def _save_to_vault(yesterday: str, summary: dict, msg: str):
    VAULT_REPORTS.mkdir(parents=True, exist_ok=True)
    report_path = VAULT_REPORTS / f"{yesterday}-cost-report.md"
    content = f"""---
date: {yesterday}
type: cost-report
total_usd: {summary['total_usd']}
total_gbp: {summary['total_gbp']}
calls: {summary['calls']}
cache_hits: {summary['cache_hits']}
routing_saves: {summary['routing_saves']}
---

# AI Cost Report — {yesterday}

{msg.replace('*', '').replace('💷', '£').replace('📞', '').replace('🎯', '').replace('🔀', '').replace('🏦', '')}

## Raw breakdown by model
```json
{json.dumps(summary['by_model'], indent=2)}
```

## Raw breakdown by agent
```json
{json.dumps(summary['by_tool'], indent=2)}
```
"""
    report_path.write_text(content)
    return report_path


def _send_telegram(msg: str):
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("EWAN_TELEGRAM_CHAT_ID") or os.getenv("TELEGRAM_CHAT_ID")
    if not (token and chat_id and chat_id.strip()):
        print("[WARN] Telegram not configured (EWAN_TELEGRAM_CHAT_ID missing)")
        return
    import urllib.request
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = json.dumps({"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"}).encode()
    try:
        urllib.request.urlopen(
            urllib.request.Request(url, data, {"Content-Type": "application/json"}),
            timeout=10
        )
        print("[OK] Report sent to Telegram")
    except Exception as e:
        print(f"[WARN] Telegram send failed: {e}")


def run_report(target_date: str = None, dry_run: bool = False):
    yesterday = target_date or (date.today() - timedelta(days=1)).isoformat()
    entries = _load_entries(yesterday)

    if not entries:
        print(f"No entries for {yesterday} — nothing to report")
        return

    summary = _summarise(entries)

    # 7-day average for trend
    week_entries = _load_period(7)
    days_with_data = len(set(e["date"] for e in week_entries))
    avg_7d = sum(e.get("cost_usd", 0) for e in week_entries) / days_with_data if days_with_data else 0

    msg = _format_telegram(yesterday, summary, avg_7d)
    vault_path = _save_to_vault(yesterday, summary, msg)

    print(f"\n--- Daily Report ({yesterday}) ---")
    print(msg.replace("*", ""))
    print(f"\nSaved to: {vault_path}")

    if not dry_run:
        _send_telegram(msg)


# ---- Launchd plist ---------------------------------------------------------

PLIST = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.amplified.daily-cost-report</string>
  <key>ProgramArguments</key>
  <array>
    <string>{python}</string>
    <string>{script}</string>
  </array>
  <key>StartCalendarInterval</key>
  <dict>
    <key>Hour</key>
    <integer>8</integer>
    <key>Minute</key>
    <integer>0</integer>
  </dict>
  <key>StandardOutPath</key>
  <string>{home}/.amplified/daily-report.log</string>
  <key>StandardErrorPath</key>
  <string>{home}/.amplified/daily-report.log</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key>
    <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string>
  </dict>
</dict>
</plist>
"""


def install_plist():
    plist_path = Path.home() / "Library/LaunchAgents/com.amplified.daily-cost-report.plist"
    plist_path.write_text(PLIST.format(
        python=sys.executable,
        script=str(Path(__file__).resolve()),
        home=str(Path.home()),
    ))
    print(f"Written: {plist_path}")
    print(f"Activate: launchctl load {plist_path}")
    print(f"Test:     python3 {Path(__file__).name} --dry-run")
    print(f"Reports saved to: {VAULT_REPORTS}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Daily AI cost report")
    parser.add_argument("--install-plist", action="store_true", help="Install launchd plist")
    parser.add_argument("--date", type=str, help="Date to report (YYYY-MM-DD, default: yesterday)")
    parser.add_argument("--dry-run", action="store_true", help="Print report, don't send Telegram")
    args = parser.parse_args()

    if args.install_plist:
        install_plist()
        sys.exit(0)

    run_report(target_date=args.date, dry_run=args.dry_run)
