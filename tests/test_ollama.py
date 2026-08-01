"""Tests for the Ollama collector."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from asimon.collectors.ollama import (
    OllamaCollector,
    OllamaPsResponse,
    OllamaTagsResponse,
)

# Real /api/ps response (empty — no models loaded)
REAL_PS_EMPTY = {"models": []}

# Real /api/tags response (abbreviated — one model)
REAL_TAGS_RESPONSE = {
    "models": [
        {
            "name": "hermes3:8b",
            "model": "hermes3:8b",
            "modified_at": "2026-07-30T23:59:25.07201974+02:00",
            "size": 4661227243,
            "digest": "4f6b83f30b62bc3d0cf9be09266db222805ee815c8fd7d8b38f863f655be78b7",
            "details": {
                "parent_model": "",
                "format": "gguf",
                "family": "llama",
                "families": ["llama"],
                "parameter_size": "8.0B",
                "quantization_level": "Q4_0",
            },
        }
    ]
}

# Simulated /api/ps with a loaded model
REAL_PS_LOADED = {
    "models": [
        {
            "name": "qwen3-coder:30b",
            "model": "qwen3-coder:30b",
            "size": 18556700761,
            "size_vram": 18556700761,
            "digest": "06c1097efce0431c2045fe7b2e5108366e43bee1b4603a7aded8f21689e90bca",
            "details": {
                "parent_model": "",
                "format": "gguf",
                "family": "qwen3moe",
                "families": ["qwen3moe"],
                "parameter_size": "30.5B",
                "quantization_level": "Q4_K_M",
            },
            "expires_at": "2026-08-01T17:36:35.366245+00:00",
            "context_length": 262144,
        }
    ]
}


class TestOllamaModelParsing:
    """Test that Ollama API responses parse correctly."""

    def test_parse_ps_empty(self):
        """Parse an empty /api/ps response."""
        parsed = OllamaPsResponse.model_validate(REAL_PS_EMPTY)
        assert len(parsed.models) == 0

    def test_parse_ps_loaded(self):
        """Parse a /api/ps response with a loaded model."""
        parsed = OllamaPsResponse.model_validate(REAL_PS_LOADED)
        assert len(parsed.models) == 1
        model = parsed.models[0]
        assert model.name == "qwen3-coder:30b"
        assert model.size == 18556700761
        assert model.size_vram == 18556700761
        assert model.details.family == "qwen3moe"
        assert model.details.parameter_size == "30.5B"
        assert model.details.quantization_level == "Q4_K_M"
        assert model.context_length == 262144
        assert model.expires_at == "2026-08-01T17:36:35.366245+00:00"

    def test_parse_tags(self):
        """Parse a /api/tags response."""
        parsed = OllamaTagsResponse.model_validate(REAL_TAGS_RESPONSE)
        assert len(parsed.models) == 1
        model = parsed.models[0]
        assert model.name == "hermes3:8b"
        assert model.size == 4661227243
        assert model.details.family == "llama"
        assert model.details.parameter_size == "8.0B"


def _make_mock_get(response_data: dict, status: int = 200):
    """Create a mock for aiohttp session.get() that works as async context manager."""
    mock_resp = AsyncMock()
    mock_resp.status = status
    mock_resp.json = AsyncMock(return_value=response_data)
    # session.get() returns an async context manager
    mock_get = MagicMock()
    mock_get.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_get.__aexit__ = AsyncMock(return_value=False)
    return mock_get


class TestOllamaCollector:
    """Test the OllamaCollector class using mocked aiohttp sessions."""

    @pytest.mark.asyncio
    async def test_get_loaded_models_empty(self):
        """Test fetching loaded models when none are loaded."""
        collector = OllamaCollector(base_url="http://localhost:11434")
        mock_get = _make_mock_get(REAL_PS_EMPTY)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_get)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "asimon.collectors.ollama.aiohttp.ClientSession", return_value=mock_session
        ):
            models = await collector.get_loaded_models()

        assert len(models) == 0

    @pytest.mark.asyncio
    async def test_get_loaded_models_with_data(self):
        """Test fetching loaded models with a loaded model."""
        collector = OllamaCollector(base_url="http://localhost:11434")
        mock_get = _make_mock_get(REAL_PS_LOADED)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_get)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "asimon.collectors.ollama.aiohttp.ClientSession", return_value=mock_session
        ):
            models = await collector.get_loaded_models()

        assert len(models) == 1
        assert models[0].name == "qwen3-coder:30b"
        assert models[0].size_vram == 18556700761

    @pytest.mark.asyncio
    async def test_get_available_models(self):
        """Test fetching available models."""
        collector = OllamaCollector(base_url="http://localhost:11434")
        mock_get = _make_mock_get(REAL_TAGS_RESPONSE)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_get)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "asimon.collectors.ollama.aiohttp.ClientSession", return_value=mock_session
        ):
            models = await collector.get_available_models()

        assert len(models) == 1
        assert models[0].name == "hermes3:8b"
        assert models[0].size == 4661227243

    @pytest.mark.asyncio
    async def test_connection_error(self):
        """Test graceful handling when Ollama is not running."""
        collector = OllamaCollector(base_url="http://localhost:11434")

        with patch(
            "asimon.collectors.ollama.aiohttp.ClientSession",
            side_effect=ConnectionRefusedError("Connection refused"),
        ):
            models = await collector.get_loaded_models()

        assert models == []

    @pytest.mark.asyncio
    async def test_non_200_response(self):
        """Test graceful handling of non-200 HTTP responses."""
        collector = OllamaCollector(base_url="http://localhost:11434")
        mock_get = _make_mock_get({}, status=503)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_get)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "asimon.collectors.ollama.aiohttp.ClientSession", return_value=mock_session
        ):
            models = await collector.get_loaded_models()

        assert models == []
