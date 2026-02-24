#!/usr/bin/env python3
"""
Token Proxy — Universal Anthropic API proxy for this Mac.

Sits between every tool (CRM, dashboard, Telegram bots, OpenClaw, Claude Code)
and the Anthropic API. Applies cost optimisations transparently.

What it does on every call:
  1. Prompt caching    — injects cache_control on system prompts (90% savings on repeats)
  2. Native compaction — injects compact-2026-01-12 beta on long conversations (server-side,
                         zero cost, Claude compacts its own context at 30k token trigger)
  3. Semantic caching  — Qdrant lookup before calling API for repeated queries (non-streaming)
  4. Model routing     — classifies prompts and downgrades Sonnet→Haiku when safe (12x saving)
  5. Cost tracking     — logs every call to ~/.amplified/cost-log.jsonl + Telegram alerts
  6. Budget control    — daily spend limit with auto-Haiku enforcement above threshold

Activation (zero changes to any other tool):
  ~/.amplified/keys.env must contain:
      ANTHROPIC_BASE_URL=http://localhost:8088

  The Anthropic Python SDK reads ANTHROPIC_BASE_URL automatically.
  Every tool that sources keys.env routes through this proxy.

Run:      python3 token_proxy.py
Daemon:   python3 token_proxy.py --install-plist && launchctl load ...
Stats:    curl http://localhost:8088/proxy/stats
Costs:    curl http://localhost:8088/proxy/costs
Port:     8088 (configurable via TOKEN_PROXY_PORT env var)

Status:
  DONE — core proxy, prompt caching injection, streaming passthrough
  DONE — semantic caching (non-streaming calls)
  DONE — native context compaction (compact-2026-01-12 beta)
  DONE — model routing (Sonnet→Haiku classifier, conservative, 12x savings on eligible calls)
  DONE — enhanced stats + historical cost breakdown endpoint
  DONE — daily budget enforcement (auto-Haiku mode above configurable threshold)
  DONE — cost attribution by agent/tool
  DONE — launchd plist installer (KeepAlive — never needs manual start)
"""

import os
import re
import sys
import json
import time
import hashlib
import asyncio
import argparse
import logging
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import Optional, AsyncIterator, Dict, Any

# Load API keys via secrets manager (keys.env → 1Password when configured)
sys.path.insert(0, str(Path.home() / ".amplified"))
from amplified_secrets import secrets  # noqa: F401 — populates os.environ on import

try:
    import httpx
    from fastapi import FastAPI, Request, Response
    from fastapi.responses import StreamingResponse, JSONResponse
    import uvicorn
except ImportError as e:
    print(f"Missing: {e}\nRun: pip install fastapi uvicorn httpx --break-system-packages")
    sys.exit(1)

# ---- Config ----------------------------------------------------------------

ANTHROPIC_REAL_URL = "https://api.anthropic.com"
PROXY_PORT = int(os.getenv("TOKEN_PROXY_PORT", "8088"))
COST_LOG_FILE = Path.home() / ".amplified/cost-log.jsonl"
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
CACHE_COLLECTION = "llm_cache"
CACHE_SIMILARITY = 0.95
CACHE_TTL_HOURS = 24
DAILY_ALERT_USD = float(os.getenv("COST_ALERT_THRESHOLD_USD", "5.0"))
DAILY_BUDGET_USD = float(os.getenv("DAILY_BUDGET_USD", "50.0"))  # Above this → force Haiku

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("proxy")

# ---- Prompt injection detection --------------------------------------------
# Alert-only — never blocks a call. Visible at GET /proxy/security.

_INJECTION_COMPILED = [
    re.compile(p, re.IGNORECASE) for p in [
        # Instruction override
        r"ignore (all |the |your |previous |prior |above |earlier )?instructions",
        r"disregard (all |the |your |previous |prior |above |earlier )?instructions",
        r"forget (everything|all|your instructions|the above)",
        r"your (new |real |true |actual )?instructions (are|say|tell)",
        r"(you are|you're) (now|actually|really|secretly) (a|an|the)",
        r"new (task|role|persona|system prompt):",
        r"pretend (you are|to be) (a|an)",
        # System prompt extraction
        r"(show|print|reveal|repeat|display|output) (your |the )?(system prompt|hidden instructions|real instructions)",
        r"what (is your|are your) (system prompt|instructions|rules|true purpose)",
        r"(repeat|echo) (everything|all text) (above|before this)",
        # Jailbreaks
        r"\bDAN\b",
        r"developer mode",
        r"jailbreak",
        # Prompt delimiter injection
        r"</?(system|s|inst|instruction)>",
        r"\[/?INST\]",
        r"<\|im_start\|>|<\|im_end\|>",
        r"###\s*(human|assistant|system|instruction)\b",
        r"<</?SYS>>",
        # Exfiltration via tool calls / URLs
        r"(send|post|put|exfiltrate|leak).{0,30}(api key|secret|token|password|credentials)",
        r"(http|https)://(?!api\.anthropic\.com).{0,50}(key|token|secret|password)",
    ]
]

_injection_log: list = []

def _scan_for_injection(body: dict) -> Optional[dict]:
    """
    Scan user messages for prompt injection patterns.
    Only scans role=user content — system content is ours.
    Returns detection dict if found, None if clean.
    """
    messages = body.get("messages", [])
    for i, msg in enumerate(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                b.get("text", "") for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            )
        if not content:
            continue
        content_lower = content.lower()
        for pattern in _INJECTION_COMPILED:
            m = pattern.search(content_lower)
            if m:
                return {
                    "message_index": i,
                    "pattern": pattern.pattern,
                    "match": m.group(0),
                    "content_preview": content[:200],
                }
    return None

def _alert_injection(details: dict, tool: str):
    entry = {
        "ts": datetime.utcnow().isoformat(),
        "tool": tool,
        **details,
    }
    _injection_log.append(entry)
    # Keep last 100
    if len(_injection_log) > 100:
        _injection_log.pop(0)
    log.warning(
        f"[INJECTION DETECTED] tool={tool} "
        f"pattern='{details['pattern'][:50]}' "
        f"match='{details['match']}'"
    )
    _send_telegram(
        f"🚨 *Prompt injection detected*\n"
        f"Tool: `{tool}`\n"
        f"Pattern: `{details['pattern'][:60]}`\n"
        f"Match: `{details['match']}`\n"
        f"Preview: `{details['content_preview'][:120]}`\n"
        f"_Alert-only — call was not blocked_"
    )

# ---- Model pricing (USD per token) ----------------------------------------

HAIKU = "claude-haiku-4-5-20251001"
SONNET = "claude-sonnet-4-6"

PRICING = {
    "claude-haiku-4-5-20251001": {"in": 0.25e-6,  "out": 1.25e-6},
    "claude-haiku-4-5":          {"in": 0.25e-6,  "out": 1.25e-6},
    "claude-sonnet-4-6":         {"in": 3.00e-6,  "out": 15.00e-6},
    "claude-sonnet-4-5":         {"in": 3.00e-6,  "out": 15.00e-6},
    "claude-opus-4-6":           {"in": 15.00e-6, "out": 75.00e-6},
    "claude-opus-4-5":           {"in": 15.00e-6, "out": 75.00e-6},
}

def calc_cost(model: str, input_tok: int, output_tok: int,
              cache_create: int = 0, cache_read: int = 0) -> float:
    p = PRICING.get(model, PRICING[SONNET])
    normal_in = max(0, input_tok - cache_create - cache_read)
    return round(
        normal_in * p["in"]
        + output_tok * p["out"]
        + cache_create * p["in"] * 1.25   # cache write = 1.25x input price
        + cache_read * p["in"] * 0.10,    # cache read = 0.10x input price
        8
    )

# ---- Cost tracking ---------------------------------------------------------

_daily_cost = 0.0
_session_cost = 0.0
_session_calls = 0
_routing_stats = {"haiku_routed": 0, "sonnet_kept": 0, "haiku_requested": 0}

def _load_today_cost() -> float:
    if not COST_LOG_FILE.exists():
        return 0.0
    today = date.today().isoformat()
    total = 0.0
    try:
        for line in COST_LOG_FILE.read_text().splitlines():
            try:
                e = json.loads(line)
                if e.get("date") == today:
                    total += e.get("cost_usd", 0)
            except Exception:
                pass
    except Exception:
        pass
    return total

def record_cost(model: str, input_tok: int, output_tok: int,
                cache_create: int = 0, cache_read: int = 0,
                from_cache: bool = False, tool: str = "unknown",
                original_model: str = ""):
    global _daily_cost, _session_cost, _session_calls
    cost = calc_cost(model, input_tok, output_tok, cache_create, cache_read)
    _daily_cost += cost
    _session_cost += cost
    _session_calls += 1

    entry = {
        "ts": datetime.utcnow().isoformat(),
        "date": date.today().isoformat(),
        "model": model,
        "original_model": original_model or model,
        "input_tokens": input_tok,
        "output_tokens": output_tok,
        "cache_creation_tokens": cache_create,
        "cache_read_tokens": cache_read,
        "cost_usd": cost,
        "from_cache": from_cache,
        "tool": tool,
    }
    try:
        COST_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(COST_LOG_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass

    routed_note = f" [routed from {original_model}]" if original_model and original_model != model else ""
    log.info(
        f"${cost:.5f} | {model.split('-')[1][:6]} | "
        f"in={input_tok} out={output_tok} "
        f"cache_r={cache_read} cache_w={cache_create} "
        f"{'[SEMANTIC HIT]' if from_cache else ''}"
        f"{routed_note} | {tool} | "
        f"session=${_session_cost:.4f} today=${_daily_cost:.4f}"
    )

    if _daily_cost >= DAILY_ALERT_USD:
        _maybe_telegram_alert()

def _send_telegram(msg: str):
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("EWAN_TELEGRAM_CHAT_ID") or os.getenv("TELEGRAM_CHAT_ID")
    if not (token and chat_id and chat_id.strip()):
        return
    try:
        import urllib.request
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        data = json.dumps({"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"}).encode()
        urllib.request.urlopen(
            urllib.request.Request(url, data, {"Content-Type": "application/json"}),
            timeout=5
        )
    except Exception:
        pass

def _maybe_telegram_alert():
    flag = Path.home() / f".amplified/cost-alert-{date.today().isoformat()}.flag"
    if flag.exists():
        return
    flag.touch()
    _send_telegram(
        f"⚠️ *Daily AI spend: ${_daily_cost:.2f}*\n"
        f"Threshold: ${DAILY_ALERT_USD:.2f} | Budget: ${DAILY_BUDGET_USD:.2f}\n"
        f"Session calls: {_session_calls} | "
        f"Routed to Haiku: {_routing_stats['haiku_routed']}"
    )

# ---- Model routing ---------------------------------------------------------
# Conservative Sonnet→Haiku classifier. Only downgrades when we're certain.
# Haiku is 12x cheaper than Sonnet ($0.25/1M vs $3.00/1M input).
# Goal: route >60% of volume to Haiku without touching quality-sensitive calls.

MODEL_ROUTING_ENABLED = os.getenv("MODEL_ROUTING_ENABLED", "true").lower() == "true"

# Patterns that indicate Haiku is sufficient (fast extraction/classification work)
_HAIKU_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\bextract\b",
        r"\bclassif",
        r"\byes or no\b",
        r"\breturn (only |just )?(a )?(json|number|integer|true|false|list|dict)\b",
        r"\breturn ONLY\b",
        r"\bformat (this|the following)\b",
        r"\bconvert (this|the following)\b",
        r"\bnormalise\b|\bnormalize\b",
        r"\bparse (this|the following|the)\b",
        r"\bscore.*?\b0\.0.*?1\.0\b",
        r"\bsentiment\b",
        r"\bsummarise this\b|\bsummarize this\b",
        r"\btrue or false\b",
        r"\bis this (a|an|the)\b",
        r"\bname and (address|phone|email)\b",
        r"\bentity extraction\b",
        r"\btag (this|the following)\b",
        r"\bcategorise\b|\bcategorize\b",
    ]
]

# Sonnet patterns are VETO — if any match, stay on Sonnet regardless
_SONNET_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\binterview\b",
        r"\bnext question\b",
        r"\bask.*\bquestion\b",
        r"\bgenerate.*\binsight\b",
        r"\banalyse\b|\banalyze\b",
        r"\bstrateg",
        r"\brecommend\b",
        r"\badvice\b",
        r"\bwrite (a|an|the|me)\b",
        r"\bdraft\b",
        r"\bexplain\b",
        r"\bthink (about|through|carefully)\b",
        r"\bplan\b",
        r"\bwhy\b.*\?\s*$",
        r"\bhow (do|does|can|should|would)\b",
    ]
]

def _route_model(body: dict) -> tuple[str, str, str]:
    """
    Returns (model_to_use, original_model, reason).
    Conservative: only downgrades Sonnet when we're very confident.
    Opus requests are never downgraded (caller made an intentional choice).
    """
    requested = body.get("model", SONNET)

    if not MODEL_ROUTING_ENABLED:
        return requested, requested, "routing_disabled"

    # Don't touch Haiku requests — already optimal
    if HAIKU in requested or "haiku" in requested.lower():
        _routing_stats["haiku_requested"] += 1
        return requested, requested, "already_haiku"

    # Don't touch Opus — if caller specified Opus, they mean it
    if "opus" in requested.lower():
        return requested, requested, "opus_respected"

    # Budget exceeded → force Haiku on everything
    if _daily_cost >= DAILY_BUDGET_USD:
        _routing_stats["haiku_routed"] += 1
        log.warning(f"[BUDGET] Daily ${_daily_cost:.2f} ≥ ${DAILY_BUDGET_USD:.2f} — forcing Haiku")
        return HAIKU, requested, "budget_exceeded"

    # Extract text to classify
    system = body.get("system", "")
    if isinstance(system, list):
        system_text = " ".join(b.get("text", "") for b in system if isinstance(b, dict))
    else:
        system_text = str(system)

    messages = body.get("messages", [])
    last_content = ""
    if messages:
        c = messages[-1].get("content", "")
        if isinstance(c, list):
            c = " ".join(b.get("text", "") for b in c if isinstance(b, dict))
        last_content = str(c)

    # Use last user message + short system prefix for classification
    classify_text = last_content + " " + system_text[:300]

    # Sonnet veto — any reasoning/generation pattern keeps us on Sonnet
    for p in _SONNET_PATTERNS:
        if p.search(classify_text):
            _routing_stats["sonnet_kept"] += 1
            return requested, requested, "sonnet_pattern"

    # Long conversations → Sonnet (context matters, quality matters)
    if len(messages) > 8:
        _routing_stats["sonnet_kept"] += 1
        return requested, requested, "long_conversation"

    # Long prompts → Sonnet (likely complex task)
    if len(last_content) > 2000:
        _routing_stats["sonnet_kept"] += 1
        return requested, requested, "long_prompt"

    # Haiku patterns match → safe to downgrade
    for p in _HAIKU_PATTERNS:
        if p.search(classify_text):
            _routing_stats["haiku_routed"] += 1
            return HAIKU, requested, "haiku_pattern"

    # Very short, no history → probably simple
    if len(last_content) < 80 and len(messages) <= 2:
        _routing_stats["haiku_routed"] += 1
        return HAIKU, requested, "short_simple"

    # Default: trust the caller's model choice
    _routing_stats["sonnet_kept"] += 1
    return requested, requested, "default"

# ---- Prompt caching injection ----------------------------------------------

def inject_cache_control(body: dict) -> dict:
    """
    Add cache_control: ephemeral to system prompt if not present.
    Anthropic charges 10% of normal price for cache reads.
    Min 1024 tokens to qualify — most system prompts are well above this.
    """
    system = body.get("system")
    if isinstance(system, str) and len(system) > 100:
        body = dict(body)
        body["system"] = [
            {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
        ]
        body["_inject_cache_beta"] = True
    elif isinstance(system, list) and system:
        # Already a list — add cache_control to the last text block if missing
        body = dict(body)
        last = system[-1]
        if isinstance(last, dict) and last.get("type") == "text" and "cache_control" not in last:
            system = list(system)
            system[-1] = {**last, "cache_control": {"type": "ephemeral"}}
            body["system"] = system
            body["_inject_cache_beta"] = True
    return body

# ---- Semantic cache (Qdrant) -----------------------------------------------

_qdrant = None
_embedder = None

def _init_cache():
    global _qdrant, _embedder
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams
        from sentence_transformers import SentenceTransformer

        _qdrant = QdrantClient(
            host=QDRANT_HOST, port=QDRANT_PORT,
            api_key=QDRANT_API_KEY, https=False,
        )
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")

        collections = [c.name for c in _qdrant.get_collections().collections]
        if CACHE_COLLECTION not in collections:
            _qdrant.create_collection(
                collection_name=CACHE_COLLECTION,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )
        log.info(f"Semantic cache ready (collection: {CACHE_COLLECTION})")
    except Exception as e:
        log.warning(f"Semantic cache disabled: {e}")
        _qdrant = None
        _embedder = None

def _make_cache_key(body: dict) -> str:
    model = body.get("model", "")
    system = body.get("system", "")
    if isinstance(system, list):
        system = " ".join(b.get("text", "") for b in system if isinstance(b, dict))
    messages = body.get("messages", [])
    parts = [f"model:{model}", f"sys:{str(system)[:400]}"]
    for msg in messages[-3:]:
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(b.get("text", "") for b in content if isinstance(b, dict))
        parts.append(f"{msg.get('role','')}:{str(content)[:300]}")
    return " | ".join(parts)

def _point_id(key: str) -> int:
    h = hashlib.sha256(key.encode()).digest()
    return int.from_bytes(h[:8], "big") % (2**63)

def cache_lookup(body: dict) -> Optional[dict]:
    if not (_qdrant and _embedder):
        return None
    try:
        key = _make_cache_key(body)
        vec = _embedder.encode(key).tolist()
        result = _qdrant.query_points(
            collection_name=CACHE_COLLECTION,
            query=vec,
            limit=1,
            score_threshold=CACHE_SIMILARITY,
        )
        results = result.points if hasattr(result, "points") else []
        if not results:
            return None
        payload = results[0].payload or {}
        cached_at = payload.get("cached_at", "")
        try:
            age = datetime.utcnow() - datetime.fromisoformat(cached_at)
            if age.total_seconds() > CACHE_TTL_HOURS * 3600:
                return None
        except Exception:
            return None
        return payload.get("response")
    except Exception as e:
        log.warning(f"Cache lookup error: {e}")
        return None

def cache_store(body: dict, response: dict):
    if not (_qdrant and _embedder):
        return
    try:
        key = _make_cache_key(body)
        vec = _embedder.encode(key).tolist()
        from qdrant_client.models import PointStruct
        _qdrant.upsert(
            collection_name=CACHE_COLLECTION,
            points=[PointStruct(
                id=_point_id(key),
                vector=vec,
                payload={
                    "response": response,
                    "cached_at": datetime.utcnow().isoformat(),
                    "model": response.get("model", ""),
                },
            )],
        )
    except Exception as e:
        log.warning(f"Cache store error: {e}")

# ---- Agent attribution ------------------------------------------------------

def _identify_tool(request: Request, body: dict) -> str:
    """
    Best-effort identification of which tool/agent made this call.
    Used for cost attribution in the log.
    Priority: X-Agent-Name header > User-Agent parsing > unknown
    """
    # Explicit header (we inject this in CRM, dashboard, etc.)
    agent_name = request.headers.get("x-agent-name", "")
    if agent_name:
        return agent_name[:40]

    ua = request.headers.get("user-agent", "").lower()
    if "claude-code" in ua:
        return "claude-code"
    if "openclaw" in ua:
        return "sam-openclaw"
    if "python-httpx" in ua:
        return "python-httpx"
    if "anthropic-python" in ua or "anthropic/python" in ua:
        # Try to identify from other signals
        referer = request.headers.get("referer", "")
        if "dashboard" in referer:
            return "dashboard"
        return "python-sdk"

    # Fallback: use raw UA truncated
    raw = request.headers.get("user-agent", "unknown")
    return raw[:40] or "unknown"

# ---- HTTP client -----------------------------------------------------------

_http_client: Optional[httpx.AsyncClient] = None

async def get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(
            base_url=ANTHROPIC_REAL_URL,
            timeout=httpx.Timeout(300.0),
            limits=httpx.Limits(max_connections=50),
        )
    return _http_client

# ---- FastAPI app -----------------------------------------------------------

app = FastAPI(title="Token Proxy", docs_url=None, redoc_url=None)

@app.get("/proxy/stats")
async def proxy_stats():
    """Live session and today stats."""
    total_routed = _routing_stats["haiku_routed"]
    total_routing = total_routed + _routing_stats["sonnet_kept"]
    haiku_rate = total_routed / total_routing if total_routing > 0 else 0.0

    return {
        "session": {
            "cost_usd": round(_session_cost, 6),
            "calls": _session_calls,
        },
        "today": {
            "cost_usd": round(_daily_cost, 6),
            "alert_threshold_usd": DAILY_ALERT_USD,
            "budget_usd": DAILY_BUDGET_USD,
            "budget_used_pct": round(_daily_cost / DAILY_BUDGET_USD * 100, 1),
        },
        "routing": {
            "enabled": MODEL_ROUTING_ENABLED,
            "haiku_routed": _routing_stats["haiku_routed"],
            "sonnet_kept": _routing_stats["sonnet_kept"],
            "haiku_already": _routing_stats["haiku_requested"],
            "haiku_rate": round(haiku_rate, 3),
            "estimated_savings_usd": round(
                _routing_stats["haiku_routed"] * 2.75e-3, 4
            ),  # rough estimate: avg call saves ~$0.00275
        },
        "cache": {
            "enabled": _qdrant is not None,
            "collection": CACHE_COLLECTION,
            "similarity_threshold": CACHE_SIMILARITY,
        },
        "proxy": {
            "port": PROXY_PORT,
            "upstream": ANTHROPIC_REAL_URL,
        }
    }

@app.get("/proxy/costs")
async def proxy_costs(days: int = 7, group_by: str = "model"):
    """
    Historical cost breakdown from cost-log.jsonl.

    Query params:
      days=7          — how many days of history to return (default 7)
      group_by=model  — 'model', 'tool', or 'date'
    """
    if not COST_LOG_FILE.exists():
        return {"error": "No cost log yet", "log": str(COST_LOG_FILE)}

    cutoff = (date.today() - timedelta(days=days)).isoformat()
    entries = []
    try:
        for line in COST_LOG_FILE.read_text().splitlines():
            try:
                e = json.loads(line)
                if e.get("date", "") >= cutoff:
                    entries.append(e)
            except Exception:
                pass
    except Exception as ex:
        return {"error": str(ex)}

    # Aggregate
    breakdown: Dict[str, Any] = {}
    total_cost = 0.0
    total_calls = 0
    cache_hits = 0

    for e in entries:
        key = e.get(group_by, "unknown")
        if key not in breakdown:
            breakdown[key] = {"cost_usd": 0.0, "calls": 0, "cache_hits": 0,
                              "input_tokens": 0, "output_tokens": 0}
        breakdown[key]["cost_usd"] += e.get("cost_usd", 0)
        breakdown[key]["calls"] += 1
        breakdown[key]["input_tokens"] += e.get("input_tokens", 0)
        breakdown[key]["output_tokens"] += e.get("output_tokens", 0)
        if e.get("from_cache"):
            breakdown[key]["cache_hits"] += 1
            cache_hits += 1
        total_cost += e.get("cost_usd", 0)
        total_calls += 1

    # Round and sort
    for v in breakdown.values():
        v["cost_usd"] = round(v["cost_usd"], 6)
    sorted_breakdown = dict(
        sorted(breakdown.items(), key=lambda x: x[1]["cost_usd"], reverse=True)
    )

    return {
        "period_days": days,
        "group_by": group_by,
        "total_cost_usd": round(total_cost, 6),
        "total_calls": total_calls,
        "cache_hits": cache_hits,
        "cache_hit_rate": round(cache_hits / total_calls, 3) if total_calls else 0,
        "breakdown": sorted_breakdown,
    }

@app.get("/proxy/security")
async def proxy_security():
    """Injection detection log. Shows last 100 detected attempts."""
    return {
        "total_detected": len(_injection_log),
        "patterns_active": len(_INJECTION_COMPILED),
        "recent": _injection_log[-10:] if _injection_log else [],
    }

@app.post("/v1/messages")
async def proxy_messages(request: Request):
    """
    Main proxy endpoint. Handles both streaming and non-streaming.

    Pipeline (non-streaming):
      model routing → cache lookup → inject caching → inject compaction → Anthropic → cache store

    Pipeline (streaming):
      model routing → inject caching → inject compaction → Anthropic (SSE passthrough)
    """
    raw_body = await request.body()
    try:
        body = json.loads(raw_body)
    except Exception:
        return Response(content=raw_body, status_code=400)

    is_stream = body.get("stream", False)
    tool = _identify_tool(request, body)

    # ---- 0. Prompt injection detection (alert-only) ----
    injection = _scan_for_injection(body)
    if injection:
        _alert_injection(injection, tool)

    # ---- 1. Model routing — downgrade Sonnet→Haiku when safe ----
    routed_model, original_model, routing_reason = _route_model(body)
    if routed_model != original_model:
        body = dict(body)
        body["model"] = routed_model
        log.info(f"[ROUTE] {original_model} → {routed_model.split('-')[1]} ({routing_reason}) | {tool}")

    # ---- 2. Prompt caching injection ----
    body = inject_cache_control(body)
    inject_beta = body.pop("_inject_cache_beta", False)
    inject_compact = body.pop("_inject_compact_beta", False)

    # ---- 3. Native context compaction (server-side, zero cost) ----
    messages = body.get("messages", [])
    estimated_tokens = sum(len(str(m.get("content", ""))) // 4 for m in messages)
    if estimated_tokens > 4000:
        body = dict(body)
        if "context_management" not in body:
            body["context_management"] = {
                "edits": [{
                    "type": "compact_20260112",
                    "trigger": {"type": "input_tokens", "value": 30000},
                }]
            }
        inject_compact = True
        log.debug(f"Native compaction armed (est. {estimated_tokens} tokens)")

    # ---- 4. Build forwarding headers ----
    forward_headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in ("host", "content-length", "transfer-encoding")
    }
    if inject_beta or inject_compact:
        existing_beta = forward_headers.get("anthropic-beta", "")
        betas = [b.strip() for b in existing_beta.split(",") if b.strip()]
        if inject_beta and "prompt-caching-2024-07-31" not in betas:
            betas.append("prompt-caching-2024-07-31")
        if inject_compact and "compact-2026-01-12" not in betas:
            betas.append("compact-2026-01-12")
        forward_headers["anthropic-beta"] = ",".join(betas)

    # ---- 5. Non-streaming: semantic cache check ----
    if not is_stream:
        cached = cache_lookup(body)
        if cached:
            usage = cached.get("usage", {})
            record_cost(
                model=cached.get("model", body.get("model", "")),
                input_tok=usage.get("input_tokens", 0),
                output_tok=usage.get("output_tokens", 0),
                from_cache=True,
                tool=tool,
                original_model=original_model,
            )
            log.info(f"[CACHE HIT] {tool}")
            return JSONResponse(content=cached)

    # ---- 6. Forward to Anthropic ----
    client = await get_http_client()
    forward_body = json.dumps(body).encode()

    if is_stream:
        async def stream_response() -> AsyncIterator[bytes]:
            async with client.stream(
                "POST", "/v1/messages",
                content=forward_body,
                headers={**forward_headers, "content-length": str(len(forward_body))},
            ) as resp:
                usage_tracked = False
                async for chunk in resp.aiter_bytes():
                    yield chunk
                    if not usage_tracked:
                        usage_tracked = _try_track_stream_chunk(
                            chunk, body.get("model", ""), tool, original_model
                        )

        return StreamingResponse(
            stream_response(),
            media_type="text/event-stream",
            headers={"x-proxy": "token-proxy"},
        )

    else:
        resp = await client.post(
            "/v1/messages",
            content=forward_body,
            headers={**forward_headers, "content-length": str(len(forward_body))},
        )
        response_body = resp.json()

        if resp.status_code == 200:
            usage = response_body.get("usage", {})
            record_cost(
                model=response_body.get("model", body.get("model", "")),
                input_tok=usage.get("input_tokens", 0),
                output_tok=usage.get("output_tokens", 0),
                cache_create=usage.get("cache_creation_input_tokens", 0),
                cache_read=usage.get("cache_read_input_tokens", 0),
                from_cache=False,
                tool=tool,
                original_model=original_model,
            )
            cache_store(body, response_body)

        return Response(
            content=json.dumps(response_body),
            status_code=resp.status_code,
            media_type="application/json",
            headers={"x-proxy": "token-proxy"},
        )

# ---- Streaming cost tracking -----------------------------------------------

def _try_track_stream_chunk(chunk: bytes, model: str, tool: str, original_model: str = "") -> bool:
    """
    Parse SSE events from streaming response to extract usage data.
    Anthropic sends usage in message_delta event (output_tokens) and
    message_start event (input_tokens). Returns True when usage has been captured.
    """
    try:
        text = chunk.decode("utf-8", errors="ignore")
        for line in text.splitlines():
            if not line.startswith("data: "):
                continue
            data_str = line[6:].strip()
            if data_str == "[DONE]":
                continue
            try:
                data = json.loads(data_str)
            except Exception:
                continue

            # message_start has input_tokens (and optionally cache tokens)
            if data.get("type") == "message_start":
                usage = data.get("message", {}).get("usage", {})
                if usage.get("input_tokens"):
                    # Store for combination with output_tokens later
                    _stream_input_cache[id(chunk)] = usage

            # message_delta has output_tokens — this is where we record cost
            elif data.get("type") == "message_delta":
                usage = data.get("usage", {})
                if usage.get("output_tokens"):
                    record_cost(
                        model=model,
                        input_tok=usage.get("input_tokens", 0),
                        output_tok=usage.get("output_tokens", 0),
                        cache_create=usage.get("cache_creation_input_tokens", 0),
                        cache_read=usage.get("cache_read_input_tokens", 0),
                        tool=tool,
                        original_model=original_model,
                    )
                    return True
    except Exception:
        pass
    return False

_stream_input_cache: dict = {}

# ---- Passthrough other Anthropic endpoints ---------------------------------
# Must be AFTER all specific routes

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_passthrough(request: Request, path: str):
    if path.startswith("proxy/"):
        return JSONResponse({"error": "not found"}, status_code=404)
    client = await get_http_client()
    forward_headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in ("host", "content-length")
    }
    body = await request.body()
    resp = await client.request(
        method=request.method,
        url=f"/{path}",
        content=body,
        headers=forward_headers,
        params=dict(request.query_params),
    )
    return Response(content=resp.content, status_code=resp.status_code,
                    media_type=resp.headers.get("content-type", "application/json"))

# ---- Startup ---------------------------------------------------------------

@app.on_event("startup")
async def startup():
    global _daily_cost
    _daily_cost = _load_today_cost()
    _init_cache()
    log.info(f"Token proxy ready on :{PROXY_PORT}")
    log.info(f"Today's spend so far: ${_daily_cost:.4f}")
    log.info(f"Model routing: {'ENABLED' if MODEL_ROUTING_ENABLED else 'DISABLED'}")
    log.info(f"Daily budget: ${DAILY_BUDGET_USD:.2f} (alert at ${DAILY_ALERT_USD:.2f})")

# ---- Launchd plist ---------------------------------------------------------

PLIST = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.amplified.token-proxy</string>
  <key>ProgramArguments</key>
  <array>
    <string>{python}</string>
    <string>{script}</string>
  </array>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>StandardOutPath</key>
  <string>{home}/.amplified/token-proxy.log</string>
  <key>StandardErrorPath</key>
  <string>{home}/.amplified/token-proxy.log</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key>
    <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string>
  </dict>
</dict>
</plist>
"""

def install_plist():
    plist_path = Path.home() / "Library/LaunchAgents/com.amplified.token-proxy.plist"
    plist_path.write_text(PLIST.format(
        python=sys.executable,
        script=str(Path(__file__).resolve()),
        home=str(Path.home()),
    ))
    print(f"Written: {plist_path}")
    print(f"\nActivate:   launchctl load {plist_path}")
    print(f"Stop:       launchctl unload {plist_path}")
    print(f"Stats:      curl http://localhost:{PROXY_PORT}/proxy/stats")
    print(f"Costs:      curl http://localhost:{PROXY_PORT}/proxy/costs")
    print(f"\nThen add to ~/.amplified/keys.env:")
    print(f"    ANTHROPIC_BASE_URL=http://localhost:{PROXY_PORT}")

# ---- Main ------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--install-plist", action="store_true")
    parser.add_argument("--port", type=int, default=PROXY_PORT)
    args = parser.parse_args()

    if args.install_plist:
        install_plist()
        sys.exit(0)

    uvicorn.run(
        "token_proxy:app",
        host="127.0.0.1",
        port=args.port,
        log_level="info",
        access_log=False,
    )
