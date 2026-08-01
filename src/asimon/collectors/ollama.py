"""Ollama collector — polls the Ollama API for model state.

Polls /api/ps for currently loaded models and /api/tags for available models.
"""

import logging

import aiohttp
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class OllamaModelDetails(BaseModel):
    """Details about an Ollama model."""

    parent_model: str = Field(default="", alias="parent_model")
    format: str = Field(default="", alias="format")
    family: str = Field(default="", alias="family")
    families: list[str] | None = Field(default=None, alias="families")
    parameter_size: str = Field(default="", alias="parameter_size")
    quantization_level: str = Field(default="", alias="quantization_level")


class OllamaLoadedModel(BaseModel):
    """A model currently loaded in Ollama (from /api/ps)."""

    name: str = Field(..., alias="name")
    model: str = Field(..., alias="model")
    size: int = Field(..., alias="size")
    size_vram: int = Field(..., alias="size_vram")
    digest: str = Field(..., alias="digest")
    details: OllamaModelDetails = Field(..., alias="details")
    expires_at: str = Field(..., alias="expires_at")
    context_length: int = Field(..., alias="context_length")

    model_config = {"populate_by_name": True}


class OllamaAvailableModel(BaseModel):
    """A model available in Ollama (from /api/tags)."""

    name: str = Field(..., alias="name")
    model: str = Field(..., alias="model")
    size: int = Field(..., alias="size")
    digest: str = Field(..., alias="digest")
    details: OllamaModelDetails = Field(..., alias="details")

    model_config = {"populate_by_name": True}


class OllamaPsResponse(BaseModel):
    """Response from /api/ps."""

    models: list[OllamaLoadedModel] = Field(default_factory=list, alias="models")

    model_config = {"populate_by_name": True}


class OllamaTagsResponse(BaseModel):
    """Response from /api/tags."""

    models: list[OllamaAvailableModel] = Field(default_factory=list, alias="models")

    model_config = {"populate_by_name": True}


class OllamaCollector:
    """Collects model state from the Ollama API."""

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip("/")

    async def get_loaded_models(self) -> list[OllamaLoadedModel]:
        """Fetch currently loaded models from /api/ps.

        Returns an empty list if Ollama is not reachable.
        """
        try:
            async with aiohttp.ClientSession() as session:  # noqa: SIM117
                async with session.get(
                    f"{self.base_url}/api/ps",
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    if resp.status != 200:
                        logger.warning("Ollama /api/ps returned status %d", resp.status)
                        return []
                    data = await resp.json()
                    parsed = OllamaPsResponse.model_validate(data)
                    return parsed.models
        except aiohttp.ClientConnectorError:
            logger.warning(
                "Cannot connect to Ollama at %s (is it running?)", self.base_url
            )
            return []
        except TimeoutError:
            logger.warning("Ollama /api/ps timed out")
            return []
        except Exception as e:  # noqa: BLE001
            logger.error("Unexpected error fetching loaded models: %s", e)
            return []

    async def get_available_models(self) -> list[OllamaAvailableModel]:
        """Fetch all available models from /api/tags.

        Returns an empty list if Ollama is not reachable.
        """
        try:
            async with aiohttp.ClientSession() as session:  # noqa: SIM117
                async with session.get(
                    f"{self.base_url}/api/tags",
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    if resp.status != 200:
                        logger.warning(
                            "Ollama /api/tags returned status %d", resp.status
                        )
                        return []
                    data = await resp.json()
                    parsed = OllamaTagsResponse.model_validate(data)
                    return parsed.models
        except aiohttp.ClientConnectorError:
            logger.warning(
                "Cannot connect to Ollama at %s (is it running?)", self.base_url
            )
            return []
        except TimeoutError:
            logger.warning("Ollama /api/tags timed out")
            return []
        except Exception as e:  # noqa: BLE001
            logger.error("Unexpected error fetching available models: %s", e)
            return []
