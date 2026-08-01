"""Click CLI for Apple Silicon Monitor."""

import asyncio
import json
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import click
from aiohttp import web

from asimon.alerts.checker import AlertChecker
from asimon.benchmark.prompts import ALL_LENGTHS, PromptLength
from asimon.benchmark.runner import run_benchmark
from asimon.collectors.macmon import MacmonCollector
from asimon.collectors.ollama import OllamaCollector
from asimon.config import Settings
from asimon.exporters.prometheus import (
    start_metrics_server,
    update_hardware_metrics,
    update_inference_metrics,
    update_model_metrics,
)
from asimon.storage.db import Database

logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version="0.1.0", prog_name="asimon")
@click.option("-v", "--verbose", count=True, help="Increase log verbosity")
def main(verbose: int) -> None:
    """Apple Silicon Monitor — Lean hardware + inference monitoring for Apple Silicon LLMs."""
    level = logging.WARNING
    if verbose == 1:
        level = logging.INFO
    elif verbose >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@main.command()
@click.option(
    "--count", default=1, type=int, help="Number of samples to collect (0 = infinite)"
)
@click.option(
    "--interval", default=None, type=float, help="Polling interval in seconds"
)
def collect(count: int, interval: float | None) -> None:
    """Collect hardware metrics from macmon."""
    settings = Settings()
    if interval is not None:
        settings.polling_interval = interval

    collector = MacmonCollector(macmon_path="/opt/homebrew/bin/macmon")

    async def _run() -> None:
        if count == 0:
            async for sample in collector.collect_stream(
                interval=settings.polling_interval
            ):
                print(json.dumps(sample.model_dump(), indent=2))
                sys.stdout.flush()
        else:
            async for sample in collector.collect_stream(
                interval=settings.polling_interval, max_samples=count
            ):
                print(json.dumps(sample.model_dump(), indent=2))
                sys.stdout.flush()

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass


@main.command()
@click.option("--port", default=9100, type=int, help="Prometheus metrics port")
@click.option("--db-path", default=None, type=str, help="Path to SQLite database file")
@click.option(
    "--retention-days",
    default=None,
    type=int,
    help="Days to keep data before automatic cleanup (default: from config, 7)",
)
def serve(port: int, db_path: str | None, retention_days: int | None) -> None:
    """Run headless collector + Prometheus exporter."""
    settings = Settings()
    if port != 9100:
        settings.metrics_port = port
    if db_path is not None:
        settings.db_path = db_path
    if retention_days is not None:
        settings.retention_days = retention_days

    async def _run() -> None:
        # 1. Initialize database
        db = Database(settings.resolved_db_path)
        await db.init_db()
        logger.info("Database initialized at %s", settings.resolved_db_path)

        # 2. Create collectors and alert checker
        macmon_collector = MacmonCollector(macmon_path="/opt/homebrew/bin/macmon")
        ollama_collector = OllamaCollector(base_url=settings.ollama_url)
        alert_checker = AlertChecker(db)

        # 3. Start metrics server
        await start_metrics_server(settings.metrics_port)
        logger.info("Metrics server started on port %d", settings.metrics_port)

        # 4. Run initial cleanup
        cleanup_results = await db.cleanup_old_records(days=settings.retention_days)
        total_cleaned = sum(cleanup_results.values())
        if total_cleaned > 0:
            logger.info(
                "Initial cleanup removed %d old records (retention: %d days)",
                total_cleaned,
                settings.retention_days,
            )

        # 5. Schedule hourly cleanup
        shutdown_event = asyncio.Event()

        async def hourly_cleanup() -> None:
            """Run cleanup every hour."""
            while not shutdown_event.is_set():
                try:
                    await asyncio.sleep(3600)
                    results = await db.cleanup_old_records(days=settings.retention_days)
                    total = sum(results.values())
                    if total > 0:
                        logger.info("Hourly cleanup removed %d old records", total)
                except Exception:
                    logger.exception("Error in hourly cleanup")

        asyncio.create_task(hourly_cleanup())

        # 6. Polling loop
        async def poll_loop() -> None:
            """Poll macmon and ollama, update metrics, check alerts, and store samples."""
            while not shutdown_event.is_set():
                try:
                    # Collect hardware sample
                    sample = await macmon_collector.collect()
                    if sample is not None:
                        # Update Prometheus metrics
                        update_hardware_metrics(sample)

                        # Store in SQLite
                        hw_data = {
                            "timestamp": sample.timestamp,
                            "cpu_power": sample.cpu_power,
                            "gpu_power": sample.gpu_power,
                            "ane_power": sample.ane_power,
                            "sys_power": sample.sys_power,
                            "cpu_temp_avg": sample.temp.cpu_temp_avg,
                            "gpu_temp_avg": sample.temp.gpu_temp_avg,
                            "gpu_freq_mhz": sample.gpu_freq_mhz,
                            "pcpu_freq_mhz": sample.pcpu_freq_mhz,
                            "ecpu_freq_mhz": sample.ecpu_freq_mhz,
                            "ram_usage_bytes": sample.memory.ram_usage,
                            "ram_total_bytes": sample.memory.ram_total,
                            "swap_usage_bytes": sample.memory.swap_usage,
                            "swap_total_bytes": sample.memory.swap_total,
                            "fan0_rpm": sample.fans[0].rpm
                            if len(sample.fans) > 0
                            else None,
                            "fan1_rpm": sample.fans[1].rpm
                            if len(sample.fans) > 1
                            else None,
                        }
                        await db.insert_hardware_sample(hw_data)

                    # Collect model state
                    loaded_models = await ollama_collector.get_loaded_models()
                    if loaded_models is not None:
                        update_model_metrics(loaded_models)

                        # Store model loads in SQLite
                        now_iso = datetime.now(UTC).isoformat()
                        for model in loaded_models:
                            model_data = {
                                "timestamp": now_iso,
                                "model_name": model.name,
                                "model_size_bytes": model.size,
                                "size_vram_bytes": model.size_vram,
                                "action": "running",
                                "context_length": model.context_length,
                                "expires_at": model.expires_at,
                            }
                            await db.insert_model_load(model_data)

                    # Read latest inference metrics from SQLite and update Prometheus gauges
                    inference_rows = await db.get_latest_inference_metrics()
                    update_inference_metrics(inference_rows)

                    # Check alerts if we have a sample
                    if sample is not None:
                        alerts = await alert_checker.check_all(
                            sample, loaded_models or []
                        )
                        for alert in alerts:
                            logger.log(
                                logging.CRITICAL
                                if alert.severity == "critical"
                                else logging.WARNING
                                if alert.severity == "warning"
                                else logging.INFO,
                                "ALERT [%s] %s",
                                alert.severity.upper(),
                                alert.message,
                            )

                except Exception:
                    logger.exception("Error in poll loop iteration")

                await asyncio.sleep(settings.polling_interval)

        try:
            await poll_loop()
        except asyncio.CancelledError:
            pass
        finally:
            shutdown_event.set()
            await db.close()
            logger.info("Shutdown complete")

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        logger.info("Received shutdown signal, stopping...")


@main.command()
@click.option(
    "--retention-days",
    default=None,
    type=int,
    help="Days to keep data before cleanup (default: from config, 7)",
)
@click.option("--db-path", default=None, type=str, help="Path to SQLite database file")
def cleanup(retention_days: int | None, db_path: str | None) -> None:
    """Remove records older than the retention period."""
    settings = Settings()
    if db_path is not None:
        settings.db_path = db_path
    if retention_days is not None:
        settings.retention_days = retention_days

    async def _run() -> None:
        db = Database(settings.resolved_db_path)
        await db.init_db()
        logger.info(
            "Cleaning up records older than %d days from %s",
            settings.retention_days,
            settings.resolved_db_path,
        )
        results = await db.cleanup_old_records(days=settings.retention_days)
        total = sum(results.values())
        for table, count in results.items():
            logger.info("  %s: %d records removed", table, count)
        logger.info("Total: %d records removed", total)
        await db.close()

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass


@main.command()
def dashboard() -> None:
    """Print instructions for the Grafana dashboard."""
    click.echo()
    click.echo("  ╔══════════════════════════════════════════════════════════╗")
    click.echo("  ║           Apple Silicon Monitor — Grafana Dashboard     ║")
    click.echo("  ╚══════════════════════════════════════════════════════════╝")
    click.echo()
    click.echo("  Quick start:")
    click.echo()
    click.echo("  1. Start the metrics exporter (in one terminal):")
    click.echo("     $ asimon serve")
    click.echo()
    click.echo("  2. Start Prometheus + Grafana (in another terminal):")
    click.echo("     $ docker compose up -d")
    click.echo()
    click.echo("  3. Open Grafana:")
    click.echo("     http://localhost:3000")
    click.echo("     Login: admin / admin")
    click.echo()
    click.echo("  4. The 'Apple Silicon Monitor' dashboard is auto-provisioned.")
    click.echo()
    click.echo("  ── Services ──────────────────────────────────────────────")
    click.echo(f"  asimon metrics:  http://localhost:{Settings().metrics_port}/metrics")
    click.echo("  Prometheus:      http://localhost:9090")
    click.echo("  Grafana:         http://localhost:3000")
    click.echo()


@main.command()
@click.option(
    "--port", default=None, type=int, help="Proxy port (default: from config or 11435)"
)
@click.option("--db-path", default=None, type=str, help="Path to SQLite database file")
def proxy(port: int | None, db_path: str | None) -> None:
    """Run the Ollama transparent proxy."""
    from asimon.proxy.server import create_proxy_app

    settings = Settings()
    proxy_port = port or settings.proxy_port
    db_path_str = db_path or settings.db_path

    async def _run() -> None:
        # 1. Initialize database
        db = Database(Path(os.path.expanduser(db_path_str)))
        await db.init_db()
        logger.info("Database initialized at %s", db_path_str)

        # 2. Create proxy app
        app = create_proxy_app(settings, db)

        # 3. Start the proxy server
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", proxy_port)
        await site.start()
        logger.info(
            "Ollama proxy listening on 0.0.0.0:%d, forwarding to %s",
            proxy_port,
            settings.ollama_url,
        )

        # 4. Wait for shutdown
        shutdown_event = asyncio.Event()

        try:
            await shutdown_event.wait()
        except (asyncio.CancelledError, KeyboardInterrupt):
            logger.info("Received shutdown signal, stopping...")
        finally:
            shutdown_event.set()
            await db.close()
            await runner.cleanup()
            logger.info("Proxy shutdown complete")

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        logger.info("Received shutdown signal, stopping...")


@main.command()
@click.option(
    "--models",
    default=None,
    type=str,
    help="Comma-separated list of models to benchmark",
)
@click.option(
    "--prompt",
    default="all",
    type=str,
    help="Prompt length: short, medium, long, or all",
)
@click.option(
    "--proxy-port",
    default=11435,
    type=int,
    help="Port for the asimon proxy",
)
@click.option(
    "--output",
    "output_format",
    default="table",
    type=click.Choice(["json", "table", "csv"]),
    help="Output format",
)
def benchmark(
    models: str | None,
    prompt: str,
    proxy_port: int,
    output_format: str,
) -> None:
    """Run LLM inference benchmarks through the asimon proxy.

    Measures tokens/sec, eval duration, and memory impact for each model/prompt
    combination. Requires asimon serve to be running.
    """
    model_list: list[str] | None = None
    if models:
        model_list = [m.strip() for m in models.split(",") if m.strip()]

    prompt_lengths: list[PromptLength] | None = None
    if prompt == "all":
        prompt_lengths = ALL_LENGTHS
    elif prompt in ("short", "medium", "long"):
        prompt_lengths = [prompt]  # type: ignore[list-item]
    else:
        click.echo(f"Unknown prompt length: {prompt}. Use short, medium, long, or all.")
        raise SystemExit(1)

    result = asyncio.run(
        run_benchmark(
            models=model_list,
            prompt_lengths=prompt_lengths,
            proxy_port=proxy_port,
            output_format=output_format,  # type: ignore[arg-type]
        )
    )
    click.echo(result)


@main.command()
@click.option("--stop-ollama", is_flag=True, help="Stop Ollama before purging memory")
@click.option("--verbose", is_flag=True, help="Show detailed output")
def clean(stop_ollama: bool, verbose: bool) -> None:
    """Purge system memory and display before/after comparison.

    Runs sudo purge to clear inactive memory pages, then shows the
    memory state before and after the operation.
    """
    import subprocess
    import time

    def _run_cmd(cmd: list[str]) -> str:
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30, check=False
            )
            return result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
            return f"Error: {e}"

    def _get_memory_info() -> dict[str, str]:
        info: dict[str, str] = {}
        info["vm_stat"] = _run_cmd(["vm_stat"])
        info["swap_usage"] = _run_cmd(["/usr/sbin/sysctl", "vm.swapusage"])
        # Free memory %
        stats = _run_cmd(["vm_stat"])
        free_pct = "N/A"
        for line in stats.splitlines():
            if "Pages free" in line:
                try:
                    free_val = int(line.split(":")[1].strip().rstrip("."))
                    total_val = 0
                    for l in stats.splitlines():
                        if ":" in l:
                            try:
                                total_val += int(l.split(":")[1].strip().rstrip("."))
                            except (ValueError, IndexError):
                                pass
                    if total_val > 0:
                        free_pct = f"{(free_val / total_val) * 100:.1f}%"
                except (ValueError, IndexError):
                    pass
                break
        info["free_memory_pct"] = free_pct
        return info

    click.echo()
    click.echo("  ╔══════════════════════════════════════════════════════════╗")
    click.echo("  ║           Apple Silicon Monitor — Memory Clean         ║")
    click.echo("  ╚══════════════════════════════════════════════════════════╝")
    click.echo()

    # Step a: Optionally stop Ollama
    if stop_ollama:
        click.echo("  Stopping Ollama...")
        _run_cmd(["pkill", "ollama"])
        click.echo("  Waiting 3 seconds for Ollama to release memory...")
        time.sleep(3)

    # Record before
    click.echo("  Recording memory state BEFORE purge...")
    before = _get_memory_info()
    if verbose:
        click.echo()
        click.echo("  ── Before ──")
        click.echo(f"  {before['vm_stat']}")
        click.echo(f"  {before['swap_usage']}")
        click.echo(f"  Free memory: {before['free_memory_pct']}")
        click.echo()

    # Step b: Run sudo purge
    click.echo("  Running sudo purge (you may be prompted for your password)...")
    try:
        purge_result = subprocess.run(
            ["sudo", "purge"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if purge_result.returncode != 0:
            stderr = purge_result.stderr.strip()
            if stderr:
                click.echo(f"  Warning: purge returned: {stderr}")
            else:
                click.echo("  Warning: purge may not have completed successfully")
        else:
            click.echo("  Purge completed successfully")
    except FileNotFoundError:
        click.echo("  Error: purge command not found. This is a macOS-only command.")
        raise SystemExit(1)
    except subprocess.TimeoutExpired:
        click.echo("  Error: purge timed out after 30 seconds")
        raise SystemExit(1)

    # Step c: Wait for memory to settle
    click.echo("  Waiting 5 seconds for memory to settle...")
    time.sleep(5)

    # Step d: Record after
    click.echo("  Recording memory state AFTER purge...")
    after = _get_memory_info()

    # Step e: Show comparison
    click.echo()
    click.echo("  ── Before vs After ──")
    click.echo(
        f"  Free memory:  {before['free_memory_pct']}  →  {after['free_memory_pct']}"
    )
    click.echo(f"  Swap:         {before['swap_usage']}")
    click.echo(f"                {after['swap_usage']}")
    click.echo()

    if verbose:
        click.echo("  ── After vm_stat ──")
        click.echo(f"  {after['vm_stat']}")
        click.echo()

    click.echo("  Done.")


if __name__ == "__main__":
    main()
