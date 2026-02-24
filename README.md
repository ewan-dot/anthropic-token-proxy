# anthropic-token-proxy

A lightweight local proxy that sits between your tools and the Anthropic API. Every call gets cheaper. Nothing changes in your code.

```
Your tools → localhost:8088 → Anthropic API
```

Built because we were spending more than we needed to. Given back because that's the deal.

---

## What it does

**On every API call, automatically:**

| Feature | How | Saving |
|---------|-----|--------|
| Prompt caching | Injects `cache_control: ephemeral` on system prompts | ~90% on cached tokens |
| Native context compaction | Arms `compact-2026-01-12` beta on long conversations | ~70-85% on long histories |
| Semantic caching | Qdrant vector similarity — returns cached response if ≥95% match | 100% (zero API call) |
| **Model routing** | **Classifies prompts, downgrades Sonnet→Haiku when safe** | **Up to 12x on routed calls** |
| Cost tracking | Logs every call to `~/.amplified/cost-log.jsonl` | Awareness |
| Budget control | Daily spend limit — forces Haiku above threshold | Hard ceiling |
| Telegram alerts | Daily alert when spend exceeds threshold | No surprises |

**Model routing** is the new addition. Haiku costs $0.25/1M tokens; Sonnet costs $3.00/1M — a 12x difference. The proxy classifies every incoming request and downgrades Sonnet→Haiku when the task is clearly extractive, classificatory, or simple. Complex tasks (interviews, generation, reasoning) stay on Sonnet. The classifier is conservative: when in doubt, it keeps the requested model.

**Native context compaction** remains the big win on conversation costs. Anthropic released server-side context management in January 2026 (`compact-2026-01-12` beta). The proxy arms it automatically on conversations over ~4,000 tokens — the Anthropic server compacts the conversation history using Claude's own understanding, then continues. Zero cost on your side. Zero quality loss.

---

## Numbers

From the research behind this project:

- Full conversation history at 25 turns: **~$0.28/interview**
- With context compaction active: **~$0.10/interview**
- At 1,000 interviews/month: **$180/month saved**
- At 10,000 interviews/month: **$1,800/month saved**

Model routing adds further savings for any multi-step pipeline where extraction, classification, and summarisation steps were previously sent to Sonnet by default.

Semantic caching adds further savings for any repeated or near-identical queries (FAQ bots, classification tasks, repeated prompts in multi-agent systems).

---

## Setup

### 1. Install

```bash
git clone https://github.com/ewan-dot/anthropic-token-proxy
cd anthropic-token-proxy
pip install fastapi uvicorn httpx qdrant-client sentence-transformers anthropic
```

### 2. Run Qdrant (for semantic caching)

```bash
# Docker
docker run -p 6333:6333 qdrant/qdrant

# Or native binary — see https://qdrant.tech/documentation/quick-start/
```

### 3. Start the proxy

```bash
python token_proxy.py
# Proxy running on http://localhost:8088
```

### 4. Point your SDK at it

```bash
export ANTHROPIC_BASE_URL=http://localhost:8088
```

Or set it permanently in your environment. The Anthropic Python SDK reads `ANTHROPIC_BASE_URL` automatically — nothing else changes.

```python
import anthropic
# This now routes through the proxy transparently
client = anthropic.Anthropic()
response = client.messages.create(...)
```

### 5. Check it's working

```bash
curl http://localhost:8088/proxy/stats
# Shows routing stats, cache status, today's spend

curl "http://localhost:8088/proxy/costs?days=7&group_by=model"
# Historical breakdown by model
```

---

## Run as a background service (macOS)

```bash
python token_proxy.py --install-plist
launchctl load ~/Library/LaunchAgents/com.amplified.token-proxy.plist
```

KeepAlive — restarts automatically if it crashes. RunAtLoad — starts on every boot. You never need to start it manually.

---

## Daily Cost Report

`daily_cost_report.py` — sends a Telegram message at 8am with yesterday's spend breakdown.

```bash
# Install as daily launchd job (runs at 08:00 every day)
python daily_cost_report.py --install-plist
launchctl load ~/Library/LaunchAgents/com.amplified.daily-cost-report.plist

# Test it
python daily_cost_report.py --dry-run
```

Report includes: total spend (USD + GBP), breakdown by model and by agent/tool, cache hit rate, routing savings, trend vs 7-day average. Also saves a markdown copy to your vault.

Set `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` (or `EWAN_TELEGRAM_CHAT_ID`) in your environment.

---

## Context Compressor

`context_compressor.py` — Haiku-based conversation history compression. Useful if you need context management without the native Anthropic beta, or for non-Anthropic contexts.

```python
from context_compressor import ContextCompressor

compressor = ContextCompressor()
messages = compressor.compress(messages)  # O(1) token cost regardless of history length
```

Uses Haiku ($0.25/1M input) to extract structured facts from old turns, keeps last 4 turns verbatim. Works anywhere, no beta required. The proxy uses native compaction first; this is the fallback.

---

## Configuration

| Environment variable | Default | Description |
|---------------------|---------|-------------|
| `TOKEN_PROXY_PORT` | `8088` | Port to listen on |
| `QDRANT_HOST` | `localhost` | Qdrant host |
| `QDRANT_PORT` | `6333` | Qdrant port |
| `QDRANT_API_KEY` | — | Qdrant API key (if auth enabled) |
| `COST_ALERT_THRESHOLD_USD` | `5.0` | Daily spend alert threshold |
| `DAILY_BUDGET_USD` | `50.0` | Hard daily budget — forces Haiku above this |
| `MODEL_ROUTING_ENABLED` | `true` | Set to `false` to disable model routing |
| `TELEGRAM_BOT_TOKEN` | — | For daily cost alerts |
| `TELEGRAM_CHAT_ID` | — | Your Telegram chat ID |

---

## Model routing details

The classifier is conservative by design. It only downgrades when it's certain:

**Routes to Haiku (fast, cheap):**
- Prompts containing: `extract`, `classify`, `yes or no`, `return only JSON/number/boolean`, `format this`, `convert`, `normalise`, `parse`, `true or false`, `sentiment`, `summarise this`, `categorise`
- Very short prompts (< 80 chars) with no conversation history
- When daily budget is exceeded (hard fallback)

**Stays on Sonnet (quality-sensitive):**
- Any prompt containing: `interview`, `next question`, `analyse`, `recommend`, `strategy`, `write`, `draft`, `explain`, `how do/does/can/should`, `why` questions
- Long conversations (> 8 messages)
- Long prompts (> 2,000 chars)
- Opus requests (always respected)
- Already-Haiku requests (not touched)

Override globally: `MODEL_ROUTING_ENABLED=false` in environment.

---

## What this doesn't do

- **Multi-provider routing** — if you need OpenAI, Gemini, Bedrock etc., use [LiteLLM](https://github.com/BerriAI/litellm) (33k stars, production-grade, the right choice for multi-provider)
- **Rate limiting** — this is a local proxy, not a gateway for multiple users
- **Streaming model routing** — routing decisions apply to both streaming and non-streaming calls (routing decision made before forwarding)

---

## Why we built this instead of adopting LiteLLM

LiteLLM is excellent and you should use it if you need multi-provider support. We looked at it seriously (see [research](https://github.com/ewan-dot/anthropic-token-proxy/blob/main/research/)).

We built this because:
1. We're Anthropic-only and wanted something with zero overhead for that case
2. We wanted Qdrant-backed semantic caching (not Redis)
3. We wanted native context compaction wired in automatically
4. We wanted model routing that understands Anthropic's model tier economics specifically
5. We wanted to understand every line of what sits between our tools and the API

If you're Anthropic-only and running locally, this is lighter. If you need multi-provider, use LiteLLM.

---

## Companion modules

These live in the repo and are independently usable:

- **`semantic_cache.py`** — drop-in wrapper for `anthropic.messages.create()` with Qdrant caching
- **`model_router.py`** — classify prompts and route to Haiku vs Sonnet based on task type
- **`cost_monitor.py`** — per-call cost tracking, JSONL logging, Telegram alerts
- **`context_compressor.py`** — Haiku-based conversation history compression
- **`daily_cost_report.py`** — daily Telegram cost summary with model/agent breakdown

---

## Licence

MIT. Take it. Use it. Improve it. We took from open source; this is the giving back.

---

Built by [Amplified Partners](https://amplifiedpartners.ai) — AI-powered consultancy for UK small businesses.
