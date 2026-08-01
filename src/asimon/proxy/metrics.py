"""Inference metric recording for the Ollama transparent proxy.

Provides a single function to record an inference run: update Prometheus
gauges and write to SQLite. This keeps the proxy server clean and focused
on request forwarding.
"""

import logging
from datetime import UTC, datetime

from asimon.exporters.prometheus import (
    inference_duration_seconds,
    tokens_per_second,
)
from asimon.storage.db import Database

logger = logging.getLogger(__name__)


async def record_inference_run(
    model: str,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    total_duration_ns: int | None,
    load_duration_ns: int | None,
    eval_duration_ns: int | None,
    prompt_eval_duration_ns: int | None,
    tokens_per_second_value: float | None,
    api_endpoint: str,
    db: Database,
) -> None:
    """Record an inference run: update Prometheus metrics and write to SQLite.

    Args:
        model: The model name (e.g. "qwen3-coder:30b").
        prompt_tokens: Number of prompt tokens (eval_count).
        completion_tokens: Number of completion tokens (eval_count).
        total_duration_ns: Total request duration in nanoseconds.
        load_duration_ns: Model load duration in nanoseconds.
        eval_duration_ns: Evaluation duration in nanoseconds.
        prompt_eval_duration_ns: Prompt evaluation duration in nanoseconds.
        tokens_per_second_value: Calculated tokens per second.
        api_endpoint: The API endpoint used ("/api/generate" or "/api/chat").
        db: Database instance for SQLite storage.
    """
    # Update Prometheus gauges
    if tokens_per_second_value is not None:
        tokens_per_second.labels(model=model).set(tokens_per_second_value)

    if total_duration_ns is not None:
        inference_duration_seconds.labels(model=model).set(
            total_duration_ns / 1_000_000_000
        )

    # Write to SQLite
    try:
        await db.insert_inference_run(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "model_name": model,
                "prompt_eval_count": prompt_tokens,
                "eval_count": completion_tokens,
                "total_duration_ns": total_duration_ns,
                "eval_duration_ns": eval_duration_ns,
                "prompt_eval_duration_ns": prompt_eval_duration_ns,
                "load_duration_ns": load_duration_ns,
                "tokens_per_second": tokens_per_second_value,
                "streaming": 1,
            }
        )
    except Exception:
        logger.exception("Failed to record inference run to database")
