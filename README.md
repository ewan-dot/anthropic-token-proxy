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
| Cost tracking | Logs every call to `~/.amplified/cost-log.jsonl` | Awareness |
| Telegram alerts | Daily alert when spend exceeds threshold | No surprises |

**Native context compaction** is the big one. Anthropic released server-side context management in January 2026 (`compact-2026-01-12` beta). The proxy arms it automatically on conversations over ~4,000 tokens — the Anthropic server compacts the conversation history using Claude's own understanding, then continues. Zero cost on your side. Zero quality loss. The proxy just needs to include the right beta header and trigger config.

---

## Numbers

From the research behind this project:

- Full conversation history at 25 turns: **~$0.28/interview**
- With context compaction active: **~$0.10/interview**
- At 1,000 interviews/month: **$180/month saved**
- At 10,000 interviews/month: **$1,800/month saved**

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
# {"session_cost_usd": 0.000023, "session_calls": 2, "cache_enabled": true, ...}
```

---

## Run as a background service (macOS)

```bash
python token_proxy.py --install-plist
launchctl load ~/Library/LaunchAgents/com.amplified.token-proxy.plist
```

KeepAlive — restarts automatically if it crashes.

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
| `TELEGRAM_BOT_TOKEN` | — | For daily cost alerts |
| `TELEGRAM_CHAT_ID` | — | Your Telegram chat ID |

---

## What this doesn't do

- **Multi-provider routing** — if you need OpenAI, Gemini, Bedrock etc., use [LiteLLM](https://github.com/BerriAI/litellm) (33k stars, production-grade, the right choice for multi-provider)
- **Model routing** — the `model_router.py` module exists but isn't wired into the proxy yet
- **Rate limiting** — this is a local proxy, not a gateway for multiple users

---

## Why we built this instead of adopting LiteLLM

LiteLLM is excellent and you should use it if you need multi-provider support. We looked at it seriously (see [research](https://github.com/ewan-dot/anthropic-token-proxy/blob/main/research/)).

We built this because:
1. We're Anthropic-only and wanted something with zero overhead for that case
2. We wanted Qdrant-backed semantic caching (not Redis)
3. We wanted native context compaction wired in automatically
4. We wanted to understand every line of what sits between our tools and the API

If you're Anthropic-only and running locally, this is lighter. If you need multi-provider, use LiteLLM.

---

## Companion modules

These live in the repo and are independently usable:

- **`semantic_cache.py`** — drop-in wrapper for `anthropic.messages.create()` with Qdrant caching
- **`model_router.py`** — classify prompts and route to Haiku vs Sonnet based on task type
- **`cost_monitor.py`** — per-call cost tracking, JSONL logging, Telegram alerts

---

## Licence

MIT. Take it. Use it. Improve it. We took from open source; this is the giving back.

---

Built by [Amplified Partners](https://amplifiedpartners.ai) — AI-powered consultancy for UK small businesses.
