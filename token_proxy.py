#!/usr/bin/env python3
"""
Token Proxy — Universal Anthropic API proxy for this Mac.

Sits between every tool (CRM, dashboard, Telegram bots, OpenClaw, Claude Code)
and the Anthropic API. Applies cost optimisations transparently.

What it does on every call:
  1. Prompt caching    — injects cache_control on system prompts automatically (90% savings)
  2. Native compaction — injects compact-2026-01-12 beta on long conversations (server-side,
                         zero cost, Claude compacts its own context at 30k token trigger)
  3. Semantic caching  — Qdrant lookup before calling API for repeated queries (non-streaming)
  4. Cost tracking     — logs every call to ~/.amplified/cost-log.jsonl + Telegram alerts
  5. Model routing     — config-driven (UNFINISHED — see model_router.py)

Activation (zero changes to any other tool):
  Add to ~/.amplified/keys.env:
      ANTHROPIC_BASE_URL=http://localhost:8088

  The Anthropic Python SDK reads ANTHROPIC_BASE_URL automatically.
  Every tool that sources keys.env routes through this proxy.

Run:      python3 token_proxy.py
Daemon:   python3 token_proxy.py --install-plist && launchctl load ...
Port:     8088 (configurable via TOKEN_PROXY_PORT env var)

Status:
  DONE — core proxy, prompt caching injection, cost tracking, streaming passthrough
  DONE — semantic caching (non-streaming calls)
  DONE — launchd plist installer
  UNFINISHED — model routing (config exists, classifier not yet wired)
  UNFINISHED — context compression (see context_compressor.py, not yet integrated here)
"""

import os
import sys
import json
import time
import hashlib
import asyncio
import argparse
import logging
from pathlib import Path
from datetime import datetime, date
from typing import Optional, AsyncIterator

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

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("proxy")

# ---- Model pricing (USD per token) ----------------------------------------

PRICING = {
    "claude-haiku-4-5-20251001": {"in": 0.25e-6, "out": 1.25e-6},
    "claude-haiku-4-5":          {"in": 0.25e-6, "out": 1.25e-6},
    "claude-sonnet-4-6":         {"in": 3.00e-6, "out": 15.00e-6},
    "claude-sonnet-4-5":         {"in": 3.00e-6, "out": 15.00e-6},
    "claude-opus-4-6":           {"in": 15.00e-6, "out": 75.00e-6},
}

def calc_cost(model: str, input_tok: int, output_tok: int,
              cache_create: int = 0, cache_read: int = 0) -> float:
    p = PRICING.get(model, PRICING["claude-sonnet-4-6"])
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
                from_cache: bool = False, tool: str = "unknown"):
    global _daily_cost, _session_cost, _session_calls
    cost = calc_cost(model, input_tok, output_tok, cache_create, cache_read)
    _daily_cost += cost
    _session_cost += cost
    _session_calls += 1

    entry = {
        "ts": datetime.utcnow().isoformat(),
        "date": date.today().isoformat(),
        "model": model,
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

    log.info(
        f"${cost:.5f} | {model.split('-')[1][:6]} | "
        f"in={input_tok} out={output_tok} "
        f"cache_r={cache_read} cache_w={cache_create} "
        f"{'[HIT]' if from_cache else ''} | "
        f"session=${_session_cost:.4f} today=${_daily_cost:.4f}"
    )

    if _daily_cost >= DAILY_ALERT_USD:
        _maybe_telegram_alert()

def _maybe_telegram_alert():
    flag = Path.home() / f".amplified/cost-alert-{date.today().isoformat()}.flag"
    if flag.exists():
        return
    flag.touch()
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not (token and chat_id):
        return
    msg = (f"⚠️ Daily AI spend: ${_daily_cost:.4f} (threshold ${DAILY_ALERT_USD:.2f})")
    try:
        import urllib.request
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        data = json.dumps({"chat_id": chat_id, "text": msg}).encode()
        urllib.request.urlopen(
            urllib.request.Request(url, data, {"Content-Type": "application/json"}),
            timeout=5
        )
    except Exception:
        pass

# ---- Prompt caching injection ----------------------------------------------

def inject_cache_control(body: dict) -> dict:
    """
    Automatically add cache_control: ephemeral to system prompt if not present.
    Anthropic caches system prompts that have this — cached reads cost 10% of normal.
    Min 1024 tokens to qualify; typical system prompts are 500-3000 tokens.
    Only applied if system is a string (we convert to content block).
    """
    system = body.get("system")
    if isinstance(system, str) and len(system) > 100:
        body = dict(body)
        body["system"] = [
            {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
        ]
        # Add beta header so Anthropic processes cache_control
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
        from qdrant_client.models import ScoredPoint
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
        # TTL check
        cached_at = payload.get("cached_at", "")
        try:
            from datetime import timedelta
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

def _tool_from_headers(headers) -> str:
    ua = headers.get("user-agent", "")
    if "claude-code" in ua.lower():
        return "claude-code"
    if "openclaw" in ua.lower():
        return "openclaw"
    if "python-httpx" in ua.lower():
        return "python-sdk"
    return ua[:40] or "unknown"

@app.post("/v1/messages")
async def proxy_messages(request: Request):
    """
    Main proxy endpoint. Handles both streaming and non-streaming.

    Non-streaming: semantic cache check → inject prompt caching → forward → cache store
    Streaming: inject prompt caching → forward (SSE passthrough)
    """
    raw_body = await request.body()
    try:
        body = json.loads(raw_body)
    except Exception:
        return Response(content=raw_body, status_code=400)

    is_stream = body.get("stream", False)
    tool = _tool_from_headers(request.headers)

    # Inject prompt caching headers
    body = inject_cache_control(body)
    inject_beta = body.pop("_inject_cache_beta", False)
    inject_compact = body.pop("_inject_compact_beta", False)

    # Build forwarding headers
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

    # ---- Context management -----------------------------------------------
    # Strategy 1 (preferred): Anthropic native compaction beta.
    #   Server-side, zero cost, Claude compacts its own context.
    #   Triggers at 30k tokens — well before the 200k limit.
    # Strategy 2 (fallback): Haiku-based summarisation for non-beta clients
    #   or when native compaction is not appropriate.

    messages = body.get("messages", [])
    estimated_tokens = sum(len(str(m.get("content", ""))) // 4 for m in messages)

    if estimated_tokens > 4000:  # Only bother for conversations worth compacting
        # Inject Anthropic native compaction (compact-2026-01-12 beta)
        body = dict(body)
        if "context_management" not in body:
            body["context_management"] = {
                "edits": [{
                    "type": "compact_20260112",
                    "trigger": {"type": "input_tokens", "value": 30000},
                }]
            }
        # Ensure the beta header is included
        body["_inject_compact_beta"] = True
        log.debug(f"Native compaction armed (est. {estimated_tokens} tokens in history)")

    # ---- Non-streaming: check semantic cache first ----
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
            )
            log.info(f"[CACHE HIT] {tool}")
            return JSONResponse(content=cached)

    # ---- Forward to Anthropic ----
    client = await get_http_client()
    forward_body = json.dumps(body).encode()

    if is_stream:
        # Streaming: SSE passthrough
        async def stream_response() -> AsyncIterator[bytes]:
            async with client.stream(
                "POST", "/v1/messages",
                content=forward_body,
                headers={**forward_headers, "content-length": str(len(forward_body))},
            ) as resp:
                async for chunk in resp.aiter_bytes():
                    yield chunk
                    # Parse cost from the final [DONE] event if present
                    # (best-effort — streaming cost tracking via message_delta events)
                    _try_track_stream_chunk(chunk, body.get("model", ""), tool)

        return StreamingResponse(
            stream_response(),
            media_type="text/event-stream",
            headers={"x-proxy": "token-proxy"},
        )

    else:
        # Non-streaming: get full response, cache it, return it
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
            )
            cache_store(body, response_body)

        return Response(
            content=json.dumps(response_body),
            status_code=resp.status_code,
            media_type="application/json",
            headers={"x-proxy": "token-proxy"},
        )


# Streaming cost tracking — best-effort parse of SSE events
_stream_buffers: dict = {}

def _try_track_stream_chunk(chunk: bytes, model: str, tool: str):
    """Parse message_stop event from stream to get final usage."""
    try:
        text = chunk.decode("utf-8", errors="ignore")
        for line in text.splitlines():
            if line.startswith("data: ") and "message_stop" in line:
                # The usage comes in message_delta event, not message_stop
                pass
            if line.startswith("data: ") and "usage" in line:
                data_str = line[6:].strip()
                if data_str == "[DONE]":
                    return
                data = json.loads(data_str)
                usage = data.get("usage", {})
                if usage.get("output_tokens"):
                    record_cost(
                        model=model,
                        input_tok=usage.get("input_tokens", 0),
                        output_tok=usage.get("output_tokens", 0),
                        cache_create=usage.get("cache_creation_input_tokens", 0),
                        cache_read=usage.get("cache_read_input_tokens", 0),
                        tool=tool,
                    )
    except Exception:
        pass


@app.get("/proxy/stats")
async def proxy_stats():
    return {
        "session_cost_usd": round(_session_cost, 6),
        "session_calls": _session_calls,
        "today_cost_usd": round(_daily_cost, 6),
        "daily_alert_threshold_usd": DAILY_ALERT_USD,
        "cache_enabled": _qdrant is not None,
        "port": PROXY_PORT,
    }

# Passthrough other Anthropic endpoints (models list, etc.)
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


@app.on_event("startup")
async def startup():
    global _daily_cost
    _daily_cost = _load_today_cost()
    _init_cache()
    log.info(f"Token proxy ready on :{PROXY_PORT}")
    log.info(f"Today's spend so far: ${_daily_cost:.4f}")
    log.info(f"Set ANTHROPIC_BASE_URL=http://localhost:{PROXY_PORT} in keys.env to activate")

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
