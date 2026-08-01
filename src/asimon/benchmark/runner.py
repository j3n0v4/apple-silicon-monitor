"""Benchmark runner — orchestrates LLM inference benchmarks via the asimon proxy.

Records baseline memory state, runs inference requests through the proxy
for each model/prompt combination, then queries the SQLite database for
the recorded metrics and outputs them in the requested format.
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import aiohttp

from asimon.benchmark.prompts import ALL_LENGTHS, PROMPTS, PromptLength
from asimon.config import Settings
from asimon.storage.db import Database

logger = logging.getLogger(__name__)

OutputFormat = Literal["json", "table", "csv"]

DEFAULT_MODELS = [
    "hermes3:8b",
    "gemma4:12b-nvfp4",
    "gemma4:26b-mlx",
    "qwen3.6:35b-a3b-nvfp4",
]


@dataclass
class MemorySnapshot:
    """A snapshot of system memory state."""

    timestamp: str
    vm_stat: dict[str, int] = field(default_factory=dict)
    swap_usage: str = ""
    free_memory_pct: float = 0.0


@dataclass
class BenchmarkResult:
    """Results for a single inference run."""

    model: str
    prompt_length: str
    eval_count: int | None = None
    eval_duration_ns: int | None = None
    tokens_per_second: float | None = None
    load_duration_ns: int | None = None
    total_duration_ns: int | None = None
    error: str | None = None


def _get_vm_stat() -> dict[str, int]:
    """Run vm_stat and parse the output into a dict of page counts."""
    try:
        result = subprocess.run(
            ["vm_stat"], capture_output=True, text=True, timeout=10, check=False
        )
        stats: dict[str, int] = {}
        for line in result.stdout.splitlines():
            if ":" in line:
                key, val = line.split(":", 1)
                val = val.strip().rstrip(".")
                try:
                    stats[key.strip()] = int(val)
                except ValueError:
                    pass
        return stats
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        logger.warning("Failed to run vm_stat: %s", e)
        return {}


def _get_swap_usage() -> str:
    """Run sysctl vm.swapusage and return the raw output."""
    try:
        result = subprocess.run(
            ["sysctl", "vm.swapusage"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        logger.warning("Failed to run sysctl vm.swapusage: %s", e)
        return ""


def _get_free_memory_pct() -> float:
    """Estimate free memory percentage from vm_stat."""
    stats = _get_vm_stat()
    if not stats:
        return 0.0
    # Pages are 16384 bytes on Apple Silicon
    free = stats.get("Pages free", 0)
    active = stats.get("Pages active", 0)
    inactive = stats.get("Pages inactive", 0)
    wired = stats.get("Pages wired down", 0)
    compressed = stats.get("Pages occupied by compressor", 0)
    total = free + active + inactive + wired + compressed
    if total == 0:
        return 0.0
    return (free / total) * 100


def _take_snapshot() -> MemorySnapshot:
    """Take a full memory snapshot."""
    return MemorySnapshot(
        timestamp=datetime.now(UTC).isoformat(),
        vm_stat=_get_vm_stat(),
        swap_usage=_get_swap_usage(),
        free_memory_pct=_get_free_memory_pct(),
    )


async def _check_serve_health(metrics_port: int) -> bool:
    """Check if asimon serve is running by hitting the /health endpoint."""
    try:
        async with (
            aiohttp.ClientSession() as session,
            session.get(f"http://localhost:{metrics_port}/health", timeout=5) as resp,
        ):
            return resp.status == 200
    except (TimeoutError, aiohttp.ClientError):
        return False


async def _check_proxy_running(proxy_port: int) -> bool:
    """Check if the asimon proxy is running by hitting /api/tags."""
    try:
        async with (
            aiohttp.ClientSession() as session,
            session.get(f"http://localhost:{proxy_port}/api/tags", timeout=5) as resp,
        ):
            return resp.status == 200
    except (TimeoutError, aiohttp.ClientError):
        return False


async def _start_proxy(proxy_port: int, db_path: str) -> subprocess.Popen | None:
    """Start the asimon proxy as a subprocess.

    Returns the Popen object or None if startup fails.
    """
    logger.info("Starting asimon proxy on port %d...", proxy_port)
    try:
        proc = subprocess.Popen(  # noqa: ASYNC220
            [
                sys.executable,
                "-m",
                "asimon",
                "proxy",
                "--port",
                str(proxy_port),
                "--db-path",
                db_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        # Wait for proxy to be ready
        for _ in range(30):
            if await _check_proxy_running(proxy_port):
                logger.info("Proxy started successfully")
                return proc
            await asyncio.sleep(0.5)
        logger.error("Proxy failed to start within 15 seconds")
        proc.kill()
        return None
    except OSError as e:
        logger.error("Failed to start proxy: %s", e)
        return None


async def _run_inference(
    proxy_port: int, model: str, prompt: str, timeout: int = 300
) -> dict[str, Any]:
    """Send an inference request through the proxy and return the response JSON."""
    url = f"http://localhost:{proxy_port}/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}
    async with (
        aiohttp.ClientSession() as session,
        session.post(
            url,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as resp,
    ):
        if resp.status != 200:
            text = await resp.text()
            raise RuntimeError(f"Inference request failed (HTTP {resp.status}): {text}")
        return await resp.json()


def _format_bytes(b: int) -> str:
    """Format bytes as a human-readable string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if b < 1024:
            return f"{b:.1f} {unit}"
        b /= 1024
    return f"{b:.1f} PB"


def _format_duration_ns(ns: int | None) -> str:
    """Format nanoseconds as a human-readable duration."""
    if ns is None:
        return "N/A"
    seconds = ns / 1_000_000_000
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.2f}s"
    return f"{seconds / 60:.1f}m"


def _output_json(
    results: list[BenchmarkResult], baseline: MemorySnapshot, end: MemorySnapshot
) -> str:
    """Format results as JSON."""
    output = {
        "baseline": {
            "timestamp": baseline.timestamp,
            "free_memory_pct": round(baseline.free_memory_pct, 1),
            "swap_usage": baseline.swap_usage,
        },
        "end": {
            "timestamp": end.timestamp,
            "free_memory_pct": round(end.free_memory_pct, 1),
            "swap_usage": end.swap_usage,
        },
        "results": [
            {
                "model": r.model,
                "prompt_length": r.prompt_length,
                "eval_count": r.eval_count,
                "eval_duration_ns": r.eval_duration_ns,
                "tokens_per_second": round(r.tokens_per_second, 2)
                if r.tokens_per_second is not None
                else None,
                "load_duration_ns": r.load_duration_ns,
                "total_duration_ns": r.total_duration_ns,
                "error": r.error,
            }
            for r in results
        ],
    }
    return json.dumps(output, indent=2)


def _output_table(
    results: list[BenchmarkResult], baseline: MemorySnapshot, end: MemorySnapshot
) -> str:
    """Format results as a human-readable table."""
    lines: list[str] = []
    lines.append("=" * 120)
    lines.append("  Apple Silicon Monitor — Benchmark Results")
    lines.append("=" * 120)
    lines.append("")
    lines.append(f"  Baseline:  {baseline.timestamp}")
    lines.append(f"    Free memory:  {baseline.free_memory_pct:.1f}%")
    lines.append(f"    Swap:         {baseline.swap_usage}")
    lines.append("")
    lines.append(f"  End state: {end.timestamp}")
    lines.append(f"    Free memory:  {end.free_memory_pct:.1f}%")
    lines.append(f"    Swap:         {end.swap_usage}")
    lines.append("")
    lines.append("-" * 120)
    header = f"{'Model':<30} {'Prompt':<8} {'Tokens':>8} {'Tok/s':>8} {'Eval':>10} {'Load':>10} {'Total':>10}"
    lines.append(header)
    lines.append("-" * 120)
    for r in results:
        if r.error:
            lines.append(
                f"{r.model:<30} {r.prompt_length:<8} {'ERROR':>8} {r.error:<30}"
            )
        else:
            tok_s = (
                f"{r.tokens_per_second:.1f}"
                if r.tokens_per_second is not None
                else "N/A"
            )
            lines.append(
                f"{r.model:<30} {r.prompt_length:<8} "
                f"{r.eval_count or 0:>8} {tok_s:>8} "
                f"{_format_duration_ns(r.eval_duration_ns):>10} "
                f"{_format_duration_ns(r.load_duration_ns):>10} "
                f"{_format_duration_ns(r.total_duration_ns):>10}"
            )
    lines.append("-" * 120)
    return "\n".join(lines)


def _output_csv(
    results: list[BenchmarkResult], baseline: MemorySnapshot, end: MemorySnapshot
) -> str:
    """Format results as CSV."""
    import csv
    import io

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(
        [
            "model",
            "prompt_length",
            "eval_count",
            "tokens_per_second",
            "eval_duration_ns",
            "load_duration_ns",
            "total_duration_ns",
            "error",
        ]
    )
    for r in results:
        writer.writerow(
            [
                r.model,
                r.prompt_length,
                r.eval_count,
                round(r.tokens_per_second, 2)
                if r.tokens_per_second is not None
                else "",
                r.eval_duration_ns,
                r.load_duration_ns,
                r.total_duration_ns,
                r.error or "",
            ]
        )
    return output.getvalue()


async def run_benchmark(
    models: list[str] | None = None,
    prompt_lengths: list[PromptLength] | None = None,
    proxy_port: int = 11435,
    output_format: OutputFormat = "table",
    metrics_port: int = 9100,
    db_path: str | None = None,
) -> str:
    """Run the full benchmark suite.

    Args:
        models: List of model names to benchmark. Defaults to DEFAULT_MODELS.
        prompt_lengths: List of prompt lengths to test. Defaults to ALL_LENGTHS.
        proxy_port: Port the asimon proxy is (or should be) running on.
        output_format: Output format (json, table, csv).
        metrics_port: Port the asimon metrics server is on.
        db_path: Path to the SQLite database. Defaults to Settings().db_path.

    Returns:
        Formatted benchmark results as a string.
    """
    settings = Settings()
    if models is None:
        models = DEFAULT_MODELS
    if prompt_lengths is None:
        prompt_lengths = ALL_LENGTHS
    resolved_db_path = Path(os.path.expanduser(db_path or settings.db_path))

    # Step a: Check if asimon serve is running
    logger.info("Checking asimon serve health on port %d...", metrics_port)
    if not await _check_serve_health(metrics_port):
        return (
            "ERROR: asimon serve is not running on port {metrics_port}.\n"
            "Start it with: asimon serve\n"
            "Then re-run the benchmark."
        )

    # Step b: Check if proxy is running, start it if not
    logger.info("Checking asimon proxy on port %d...", proxy_port)
    proxy_proc: subprocess.Popen | None = None
    if not await _check_proxy_running(proxy_port):
        logger.info("Proxy not running, starting it...")
        proxy_proc = await _start_proxy(proxy_port, str(resolved_db_path))
        if proxy_proc is None:
            return (
                f"ERROR: Failed to start asimon proxy on port {proxy_port}.\n"
                "Start it manually with: asimon proxy\n"
                "Then re-run the benchmark."
            )

    try:
        # Step c: Record baseline
        logger.info("Recording baseline memory state...")
        baseline = _take_snapshot()

        # Step d: Run benchmarks
        results: list[BenchmarkResult] = []
        for model in models:
            logger.info("Benchmarking model: %s", model)
            for plen in prompt_lengths:
                prompt = PROMPTS[plen]
                logger.info("  Prompt: %s", plen)
                result = BenchmarkResult(model=model, prompt_length=plen)
                try:
                    resp = await _run_inference(proxy_port, model, prompt)
                    result.eval_count = resp.get("eval_count")
                    result.eval_duration_ns = resp.get("eval_duration")
                    result.load_duration_ns = resp.get("load_duration")
                    result.total_duration_ns = resp.get("total_duration")
                    eval_dur = resp.get("eval_duration", 0)
                    eval_cnt = resp.get("eval_count", 0)
                    if eval_cnt and eval_dur:
                        result.tokens_per_second = eval_cnt / (eval_dur / 1_000_000_000)
                except Exception as e:  # noqa: BLE001
                    result.error = str(e)
                    logger.warning("  Error: %s", e)
                results.append(result)

                # Wait between prompts
                await asyncio.sleep(3)

            # Wait between models
            await asyncio.sleep(10)

        # Step e: Record end state
        logger.info("Recording end memory state...")
        end = _take_snapshot()

        # Step f: Query SQLite for inference runs from this session
        logger.info("Querying SQLite for recorded inference runs...")
        try:
            db = Database(resolved_db_path)
            await db.init_db()
            # Get all inference runs from the last hour (covers the benchmark window)
            conn = db._conn
            cursor = await conn.execute(
                """
                SELECT model_name, eval_count, eval_duration_ns,
                       load_duration_ns, total_duration_ns, tokens_per_second
                FROM inference_runs
                WHERE timestamp > ?
                ORDER BY id
                """,
                (baseline.timestamp,),
            )
            db_rows = await cursor.fetchall()
            if db_rows:
                logger.info("Found %d inference runs in database", len(db_rows))
            await db.close()
        except Exception as e:  # noqa: BLE001
            logger.warning("Could not query database: %s", e)

        # Step g: Output results
        if output_format == "json":
            return _output_json(results, baseline, end)
        elif output_format == "csv":
            return _output_csv(results, baseline, end)
        else:
            return _output_table(results, baseline, end)

    finally:
        # Clean up proxy if we started it
        if proxy_proc is not None:
            logger.info("Stopping proxy subprocess...")
            proxy_proc.terminate()
            try:
                proxy_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proxy_proc.kill()
