"""Configuration management for Apple Silicon Monitor.

Settings are loaded from environment variables with the ASIMON_ prefix,
or from a YAML config file. Environment variables take precedence.
"""

import os
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from env vars (ASIMON_ prefix) or defaults."""

    polling_interval: float = Field(
        default=1.0,
        description="Seconds between hardware metric samples",
    )
    ollama_url: str = Field(
        default="http://localhost:11434",
        description="Base URL for the Ollama API",
    )
    proxy_port: int = Field(
        default=11435,
        description="Port for the Ollama transparent proxy",
    )
    metrics_port: int = Field(
        default=9100,
        description="Port for the Prometheus /metrics endpoint",
    )
    retention_days: int = Field(
        default=7,
        description="Days to keep data before automatic cleanup",
    )
    db_path: str = Field(
        default="~/.asimon/data.db",
        description="Path to the SQLite database file",
    )

    model_config = {
        "env_prefix": "ASIMON_",
        "env_file": ".env",
        "extra": "ignore",
    }

    @property
    def resolved_db_path(self) -> Path:
        """Return the db_path with ~ expanded to the user's home directory."""
        return Path(os.path.expanduser(self.db_path))
