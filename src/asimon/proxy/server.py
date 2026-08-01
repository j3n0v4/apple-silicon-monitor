"""Ollama transparent proxy server.

An aiohttp-based reverse proxy that sits between clients and Ollama.
Intercepts /api/generate and /api/chat to capture per-request inference
metrics (tokens/sec, duration, token counts) and records them to SQLite
and Prometheus. All other endpoints are passed through unchanged.
"""

import asyncio
import json
import logging
from typing import Any

import aiohttp
from aiohttp import web

from asimon.config import Settings
from asimon.proxy.metrics import record_inference_run
from asimon.storage.db import Database

logger = logging.getLogger(__name__)

# Headers that must NOT be forwarded to Ollama (set by aiohttp or the proxy)
_FORWARD_EXCLUDE_HEADERS = frozenset(
    {
        "host",
        "content-length",
        "content-encoding",
        "transfer-encoding",
    }
)


def create_proxy_app(settings: Settings, db: Database) -> web.Application:
    """Create the aiohttp application for the Ollama transparent proxy.

    Args:
        settings: Application settings (ollama_url, etc.).
        db: Database instance for recording inference runs.

    Returns:
        A configured aiohttp web application ready to run.
    """
    app = web.Application()

    async def _proxy_handler(request: web.Request) -> web.StreamResponse:
        """Catch-all handler that forwards every request to Ollama."""
        path = request.match_info.get("path", "")
        target_url = f"{settings.ollama_url.rstrip('/')}/{path.lstrip('/')}"
        is_inference = path in ("api/generate", "api/chat")

        try:
            body = await request.read()
            headers = {
                k: v
                for k, v in request.headers.items()
                if k.lower() not in _FORWARD_EXCLUDE_HEADERS
            }
            async with (
                aiohttp.ClientSession() as session,
                session.request(
                    method=request.method,
                    url=target_url,
                    data=body,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=300),
                ) as ollama_resp,
            ):
                resp = web.StreamResponse(status=ollama_resp.status)
                resp.headers["X-Asimon-Proxy"] = "true"

                # Copy response headers from Ollama (excluding hop-by-hop)
                for key, value in ollama_resp.headers.items():
                    if key.lower() not in _FORWARD_EXCLUDE_HEADERS:
                        resp.headers[key] = value

                await resp.prepare(request)

                if is_inference and ollama_resp.status == 200:
                    # Inference endpoints: stream + accumulate stats
                    buffer = bytearray()
                    async for chunk, _ in ollama_resp.content.iter_chunks():
                        if chunk:
                            buffer.extend(chunk)
                            await resp.write(chunk)

                    await resp.write_eof()

                    # Extract stats from the accumulated NDJSON buffer
                    _extract_and_record(buffer, db)
                else:
                    # Non-inference endpoints: simple pass-through
                    async for chunk, _ in ollama_resp.content.iter_chunks():
                        if chunk:
                            await resp.write(chunk)

                    await resp.write_eof()

                return resp

        except aiohttp.ClientConnectorError:
            logger.error(
                "Cannot connect to Ollama at %s (is it running?)",
                settings.ollama_url,
            )
            return web.Response(
                status=502,
                body=b'{"error":"Ollama not reachable"}',
                content_type="application/json",
                headers={"X-Asimon-Proxy": "true"},
            )

    app.router.add_route("*", "/{path:.*}", _proxy_handler)
    return app


def _extract_and_record(buffer: bytearray, db: Database) -> None:
    """Parse the accumulated NDJSON buffer and record the inference run.

    Finds the last complete JSON line in the buffer (the final response
    object with ``done: true``) and fires off a background task to record
    the metrics.

    Args:
        buffer: Accumulated response body bytes.
        db: Database instance for recording.
    """
    try:
        text = buffer.decode("utf-8")
    except UnicodeDecodeError:
        logger.warning("Failed to decode proxy response buffer as UTF-8")
        return

    # Find the last non-empty line
    last_line: str | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            last_line = stripped

    if not last_line:
        return

    try:
        final_obj: dict[str, Any] = json.loads(last_line)
    except json.JSONDecodeError:
        logger.warning("Failed to parse final NDJSON line as JSON")
        return

    # Check if this is a final inference response
    if not final_obj.get("done") and not final_obj.get("done_reason"):
        return

    model = final_obj.get("model", "unknown")
    eval_count = final_obj.get("eval_count")
    prompt_eval_count = final_obj.get("prompt_eval_count")
    total_duration = final_obj.get("total_duration")
    load_duration = final_obj.get("load_duration")
    eval_duration = final_obj.get("eval_duration")
    prompt_eval_duration = final_obj.get("prompt_eval_duration", 0)

    # Calculate tokens per second
    tokens_per_second_value: float | None = None
    if eval_count is not None and eval_duration is not None and eval_duration > 0:
        tokens_per_second_value = eval_count / (eval_duration / 1_000_000_000)

    # Fire off the database write as a background task
    asyncio.create_task(
        record_inference_run(
            model=model,
            prompt_tokens=prompt_eval_count,
            completion_tokens=eval_count,
            total_duration_ns=total_duration,
            load_duration_ns=load_duration,
            eval_duration_ns=eval_duration,
            prompt_eval_duration_ns=prompt_eval_duration,
            tokens_per_second_value=tokens_per_second_value,
            api_endpoint="/api/generate",
            db=db,
        )
    )
