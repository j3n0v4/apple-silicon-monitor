"""Tests for the Ollama transparent proxy.

Uses a mock Ollama backend (aiohttp test server) to verify:
- Non-inference endpoints pass through correctly
- Streaming /api/generate responses are proxied with stats extraction
- Inference metrics are updated after completed runs
- Database records are created for completed inference runs
- 502 response when Ollama is not reachable
"""

import asyncio
import json

import aiohttp
import pytest
from aiohttp import web

from asimon.config import Settings
from asimon.proxy.server import create_proxy_app
from asimon.storage.db import Database

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ndjson_chunks(objects: list[dict]) -> bytes:
    """Convert a list of dicts to NDJSON bytes (newline-delimited JSON)."""
    return b"\n".join(json.dumps(obj).encode() for obj in objects) + b"\n"


# Sample NDJSON for /api/generate (streaming)
GENERATE_CHUNKS = [
    {"model": "qwen3-coder:30b", "response": "Hello", "done": False},
    {"model": "qwen3-coder:30b", "response": " world", "done": False},
    {
        "model": "qwen3-coder:30b",
        "response": "",
        "done": True,
        "done_reason": "stop",
        "eval_count": 2,
        "eval_duration": 2000000000,
        "total_duration": 3000000000,
        "load_duration": 100000000,
        "prompt_eval_count": 15,
        "prompt_eval_duration": 500000000,
    },
]

# Sample NDJSON for /api/chat (streaming)
CHAT_CHUNKS = [
    {
        "model": "hermes3:8b",
        "message": {"role": "assistant", "content": "Hi"},
        "done": False,
    },
    {
        "model": "hermes3:8b",
        "message": {"role": "assistant", "content": " there"},
        "done": False,
    },
    {
        "model": "hermes3:8b",
        "message": {"role": "assistant", "content": ""},
        "done": True,
        "done_reason": "stop",
        "eval_count": 2,
        "eval_duration": 1500000000,
        "total_duration": 2500000000,
        "load_duration": 50000000,
        "prompt_eval_count": 10,
        "prompt_eval_duration": 300000000,
    },
]

# /api/tags response (non-streaming)
TAGS_RESPONSE = {
    "models": [
        {
            "name": "qwen3-coder:30b",
            "model": "qwen3-coder:30b",
            "size": 18556700761,
            "digest": "abc123",
            "details": {
                "parent_model": "",
                "format": "gguf",
                "family": "qwen3moe",
                "families": ["qwen3moe"],
                "parameter_size": "30.5B",
                "quantization_level": "Q4_K_M",
            },
        }
    ]
}


def _make_mock_ollama_app() -> web.Application:
    """Create a mock Ollama backend for testing."""

    app = web.Application()

    async def handle_generate(request: web.Request) -> web.StreamResponse:
        resp = web.StreamResponse()
        resp.headers["Content-Type"] = "application/x-ndjson"
        await resp.prepare(request)
        for chunk in GENERATE_CHUNKS:
            await resp.write(json.dumps(chunk).encode() + b"\n")
            await asyncio.sleep(0.001)
        await resp.write_eof()
        return resp

    async def handle_chat(request: web.Request) -> web.StreamResponse:
        resp = web.StreamResponse()
        resp.headers["Content-Type"] = "application/x-ndjson"
        await resp.prepare(request)
        for chunk in CHAT_CHUNKS:
            await resp.write(json.dumps(chunk).encode() + b"\n")
            await asyncio.sleep(0.001)
        await resp.write_eof()
        return resp

    async def handle_tags(request: web.Request) -> web.Response:
        return web.json_response(TAGS_RESPONSE)

    app.router.add_post("/api/generate", handle_generate)
    app.router.add_post("/api/chat", handle_chat)
    app.router.add_get("/api/tags", handle_tags)

    return app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def mock_ollama_server():
    """Start a mock Ollama backend server and yield its URL."""
    app = _make_mock_ollama_app()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()

    # Get the actual port
    port = site._server.sockets[0].getsockname()[1]
    url = f"http://127.0.0.1:{port}"

    yield url

    await runner.cleanup()


@pytest.fixture
async def db():
    """Create an in-memory database for testing."""
    database = Database(db_path=":memory:")
    await database.init_db()
    yield database
    await database.close()


@pytest.fixture
def settings(mock_ollama_server: str) -> Settings:
    """Create settings pointing at the mock Ollama server."""
    return Settings(
        ollama_url=mock_ollama_server,
        proxy_port=0,  # random port
        db_path=":memory:",
    )


@pytest.fixture
async def proxy_app(settings: Settings, db: Database):
    """Create the proxy app and start it on a random port, yield the URL."""
    app = create_proxy_app(settings, db)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()

    port = site._server.sockets[0].getsockname()[1]
    url = f"http://127.0.0.1:{port}"

    yield url, db

    await runner.cleanup()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestProxyNonInference:
    """Test that non-inference endpoints pass through correctly."""

    @pytest.mark.asyncio
    async def test_tags_passthrough(self, proxy_app):
        """Test that /api/tags is passed through unchanged."""
        url, _ = proxy_app

        async with (
            aiohttp.ClientSession() as session,
            session.get(f"{url}/api/tags") as resp,
        ):
            assert resp.status == 200
            assert resp.headers.get("X-Asimon-Proxy") == "true"
            data = await resp.json()
            assert data == TAGS_RESPONSE

    @pytest.mark.asyncio
    async def test_root_passthrough(self, proxy_app, mock_ollama_server):
        """Test that root path is passed through."""
        url, _ = proxy_app

        async with aiohttp.ClientSession() as session, session.get(f"{url}/") as resp:
            # Mock server has no root handler, so expect 404 from mock
            assert resp.status == 404


class TestProxyInferenceGenerate:
    """Test that /api/generate is proxied with stats extraction."""

    @pytest.mark.asyncio
    async def test_generate_streams_through(self, proxy_app):
        """Test that /api/generate response is streamed through."""
        url, _ = proxy_app

        async with (
            aiohttp.ClientSession() as session,
            session.post(
                f"{url}/api/generate",
                json={"model": "qwen3-coder:30b", "prompt": "hello"},
            ) as resp,
        ):
            assert resp.status == 200
            assert resp.headers.get("X-Asimon-Proxy") == "true"
            text = await resp.text()

        # Verify all chunks came through
        assert "Hello" in text
        assert "world" in text
        assert '"done":true' in text or '"done": true' in text

    @pytest.mark.asyncio
    async def test_generate_records_inference(self, proxy_app):
        """Test that /api/generate creates a database record."""
        url, db = proxy_app

        async with (
            aiohttp.ClientSession() as session,
            session.post(
                f"{url}/api/generate",
                json={"model": "qwen3-coder:30b", "prompt": "hello"},
            ) as resp,
        ):
            assert resp.status == 200
            # Consume the response
            await resp.text()

        # Give the background task time to complete
        await asyncio.sleep(0.1)

        # Check the database
        conn = db._conn
        cursor = await conn.execute(
            "SELECT model_name, eval_count, prompt_eval_count, "
            "tokens_per_second, streaming FROM inference_runs"
        )
        rows = await cursor.fetchall()
        assert len(rows) == 1
        row = rows[0]
        assert row["model_name"] == "qwen3-coder:30b"
        assert row["eval_count"] == 2
        assert row["prompt_eval_count"] == 15
        assert row["streaming"] == 1
        # 2 tokens / (2_000_000_000 ns / 1e9) = 1.0 tok/s
        assert row["tokens_per_second"] == pytest.approx(1.0, rel=0.01)


class TestProxyInferenceChat:
    """Test that /api/chat is proxied with stats extraction."""

    @pytest.mark.asyncio
    async def test_chat_streams_through(self, proxy_app):
        """Test that /api/chat response is streamed through."""
        url, _ = proxy_app

        async with (
            aiohttp.ClientSession() as session,
            session.post(
                f"{url}/api/chat",
                json={
                    "model": "hermes3:8b",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            ) as resp,
        ):
            assert resp.status == 200
            assert resp.headers.get("X-Asimon-Proxy") == "true"
            text = await resp.text()

        assert "Hi" in text
        assert "there" in text
        assert '"done":true' in text or '"done": true' in text

    @pytest.mark.asyncio
    async def test_chat_records_inference(self, proxy_app):
        """Test that /api/chat creates a database record."""
        url, db = proxy_app

        async with (
            aiohttp.ClientSession() as session,
            session.post(
                f"{url}/api/chat",
                json={
                    "model": "hermes3:8b",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            ) as resp,
        ):
            assert resp.status == 200
            await resp.text()

        # Give the background task time to complete
        await asyncio.sleep(0.1)

        conn = db._conn
        cursor = await conn.execute(
            "SELECT model_name, eval_count, prompt_eval_count, "
            "tokens_per_second, streaming FROM inference_runs"
        )
        rows = await cursor.fetchall()
        assert len(rows) == 1
        row = rows[0]
        assert row["model_name"] == "hermes3:8b"
        assert row["eval_count"] == 2
        assert row["prompt_eval_count"] == 10
        assert row["streaming"] == 1
        # 2 tokens / (1_500_000_000 ns / 1e9) ≈ 1.333 tok/s
        assert row["tokens_per_second"] == pytest.approx(1.333, rel=0.05)


class TestProxyConnectionError:
    """Test proxy behavior when Ollama is not running."""

    @pytest.mark.asyncio
    async def test_returns_502_when_ollama_down(self):
        """Test that proxy returns 502 when Ollama is unreachable."""
        settings = Settings(
            ollama_url="http://127.0.0.1:1",  # unlikely to be running
            proxy_port=0,
            db_path=":memory:",
        )
        db = Database(db_path=":memory:")
        await db.init_db()

        app = create_proxy_app(settings, db)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()

        port = site._server.sockets[0].getsockname()[1]
        url = f"http://127.0.0.1:{port}"

        async with (
            aiohttp.ClientSession() as session,
            session.post(
                f"{url}/api/generate",
                json={"model": "test", "prompt": "hello"},
            ) as resp,
        ):
            assert resp.status == 502
            data = await resp.json()
            assert "error" in data
            assert "Ollama not reachable" in data["error"]
            assert resp.headers.get("X-Asimon-Proxy") == "true"

        await runner.cleanup()
        await db.close()


class TestProxyMetrics:
    """Test that Prometheus metrics are updated after inference."""

    @pytest.mark.asyncio
    async def test_metrics_updated_after_generate(self, proxy_app):
        """Test that Prometheus gauges are updated after a generate request."""
        url, _ = proxy_app

        # Import the gauges directly
        from asimon.exporters.prometheus import (
            inference_duration_seconds,
            tokens_per_second,
        )

        async with (
            aiohttp.ClientSession() as session,
            session.post(
                f"{url}/api/generate",
                json={"model": "qwen3-coder:30b", "prompt": "hello"},
            ) as resp,
        ):
            assert resp.status == 200
            await resp.text()

        # Give the background task time to complete
        await asyncio.sleep(0.1)

        # Check Prometheus metrics
        tok_val = tokens_per_second.labels(model="qwen3-coder:30b")._value.get()
        dur_val = inference_duration_seconds.labels(
            model="qwen3-coder:30b"
        )._value.get()

        assert tok_val == pytest.approx(1.0, rel=0.01)
        assert dur_val == pytest.approx(3.0, rel=0.01)  # 3_000_000_000 ns = 3.0s


class TestProxyExtractAndRecord:
    """Test the _extract_and_record function directly."""

    @pytest.mark.asyncio
    async def test_extract_from_generate_buffer(self, db):
        """Test extracting stats from a generate NDJSON buffer."""
        from asimon.proxy.server import _extract_and_record

        buffer = _make_ndjson_chunks(GENERATE_CHUNKS)
        _extract_and_record(buffer, db)

        await asyncio.sleep(0.05)

        conn = db._conn
        cursor = await conn.execute(
            "SELECT model_name, eval_count, tokens_per_second FROM inference_runs"
        )
        rows = await cursor.fetchall()
        assert len(rows) == 1
        assert rows[0]["model_name"] == "qwen3-coder:30b"
        assert rows[0]["eval_count"] == 2

    @pytest.mark.asyncio
    async def test_extract_from_chat_buffer(self, db):
        """Test extracting stats from a chat NDJSON buffer."""
        from asimon.proxy.server import _extract_and_record

        buffer = _make_ndjson_chunks(CHAT_CHUNKS)
        _extract_and_record(buffer, db)

        await asyncio.sleep(0.05)

        conn = db._conn
        cursor = await conn.execute(
            "SELECT model_name, eval_count, tokens_per_second FROM inference_runs"
        )
        rows = await cursor.fetchall()
        assert len(rows) == 1
        assert rows[0]["model_name"] == "hermes3:8b"
        assert rows[0]["eval_count"] == 2

    @pytest.mark.asyncio
    async def test_no_done_does_not_record(self, db):
        """Test that a buffer without done:true does not create a record."""
        from asimon.proxy.server import _extract_and_record

        chunks = [
            {"model": "test", "response": "hello", "done": False},
        ]
        buffer = _make_ndjson_chunks(chunks)
        _extract_and_record(buffer, db)

        await asyncio.sleep(0.05)

        conn = db._conn
        cursor = await conn.execute("SELECT COUNT(*) FROM inference_runs")
        count = await cursor.fetchone()
        assert count[0] == 0

    @pytest.mark.asyncio
    async def test_empty_buffer_does_not_error(self, db):
        """Test that an empty buffer is handled gracefully."""
        from asimon.proxy.server import _extract_and_record

        _extract_and_record(bytearray(), db)

        await asyncio.sleep(0.05)

        conn = db._conn
        cursor = await conn.execute("SELECT COUNT(*) FROM inference_runs")
        count = await cursor.fetchone()
        assert count[0] == 0

    @pytest.mark.asyncio
    async def test_invalid_json_does_not_error(self, db):
        """Test that invalid JSON in buffer is handled gracefully."""
        from asimon.proxy.server import _extract_and_record

        _extract_and_record(b"not valid json\n", db)

        await asyncio.sleep(0.05)

        conn = db._conn
        cursor = await conn.execute("SELECT COUNT(*) FROM inference_runs")
        count = await cursor.fetchone()
        assert count[0] == 0
