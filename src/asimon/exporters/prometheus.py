"""Prometheus metrics exporter for Apple Silicon Monitor.

Registers 18 Prometheus gauges covering hardware, inference, and health metrics.
Provides update functions and an async HTTP server with a /metrics endpoint.
"""

import asyncio
import logging
import time

from aiohttp import web
from prometheus_client import Gauge, generate_latest

from asimon.collectors.macmon import MacmonSample
from asimon.collectors.ollama import OllamaLoadedModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hardware metrics (from macmon)
# ---------------------------------------------------------------------------

gpu_temp = Gauge(
    "asimon_gpu_temp_celsius",
    "GPU temperature in Celsius",
)
cpu_temp = Gauge(
    "asimon_cpu_temp_celsius",
    "CPU temperature in Celsius",
)
gpu_freq = Gauge(
    "asimon_gpu_freq_mhz",
    "GPU frequency in MHz",
)
gpu_power = Gauge(
    "asimon_gpu_power_watts",
    "GPU power draw in watts",
)
ane_power = Gauge(
    "asimon_ane_power_watts",
    "ANE power draw in watts",
)
cpu_power = Gauge(
    "asimon_cpu_power_watts",
    "CPU power draw in watts",
)
sys_power = Gauge(
    "asimon_sys_power_watts",
    "System power draw in watts",
)
ram_usage = Gauge(
    "asimon_ram_usage_bytes",
    "RAM usage in bytes",
)
ram_total = Gauge(
    "asimon_ram_total_bytes",
    "Total RAM in bytes",
)
swap_usage = Gauge(
    "asimon_swap_usage_bytes",
    "Swap usage in bytes",
)
swap_total = Gauge(
    "asimon_swap_total_bytes",
    "Total swap in bytes",
)
fan_rpm = Gauge(
    "asimon_fan_rpm",
    "Fan speed in RPM",
    labelnames=["fan"],
)

# ---------------------------------------------------------------------------
# Power source and thermal pressure metrics
# ---------------------------------------------------------------------------

power_source = Gauge(
    "asimon_power_source",
    "1 for current power source (AC Power or Battery Power)",
    labelnames=["source"],
)
battery_pct = Gauge(
    "asimon_battery_pct",
    "Battery percentage (0-100), or 100 if on AC power",
)
thermal_warning = Gauge(
    "asimon_thermal_warning",
    "1 if macOS thermal warning is active",
)
performance_warning = Gauge(
    "asimon_performance_warning",
    "1 if macOS performance warning is active",
)

# ---------------------------------------------------------------------------
# Inference metrics (from Ollama proxy — registered now, populated later)
# ---------------------------------------------------------------------------

tokens_per_second = Gauge(
    "asimon_tokens_per_second",
    "Tokens per second for a model",
    labelnames=["model"],
)
inference_duration_seconds = Gauge(
    "asimon_inference_duration_seconds",
    "Inference duration in seconds",
    labelnames=["model"],
)
loaded_models_count = Gauge(
    "asimon_loaded_models_count",
    "Number of models currently loaded in Ollama",
)
model_size_bytes = Gauge(
    "asimon_model_size_bytes",
    "Model size in bytes",
    labelnames=["model"],
)
model_vram_bytes = Gauge(
    "asimon_model_vram_bytes",
    "Model VRAM usage in bytes",
    labelnames=["model"],
)
model_loaded = Gauge(
    "asimon_model_loaded",
    "Binary indicator: 1 if model is loaded, 0 if absent",
    labelnames=["model"],
)

# ---------------------------------------------------------------------------
# Health metrics
# ---------------------------------------------------------------------------

thermal_throttling = Gauge(
    "asimon_thermal_throttling",
    "1 if macOS thermal or performance warning is active (from pmset -g therm), 0 otherwise",
)
uptime_seconds = Gauge(
    "asimon_uptime_seconds",
    "Seconds since the collector started",
)

# Track collector start time for uptime
_start_time: float = time.monotonic()


# ---------------------------------------------------------------------------
# Update functions
# ---------------------------------------------------------------------------


def update_hardware_metrics(sample: MacmonSample) -> None:
    """Update all hardware-related Prometheus gauges from a MacmonSample."""
    gpu_temp.set(sample.temp.gpu_temp_avg)
    cpu_temp.set(sample.temp.cpu_temp_avg)
    gpu_freq.set(sample.gpu_freq_mhz)
    gpu_power.set(sample.gpu_power)
    ane_power.set(sample.ane_power)
    cpu_power.set(sample.cpu_power)
    sys_power.set(sample.sys_power)
    ram_usage.set(sample.memory.ram_usage)
    ram_total.set(sample.memory.ram_total)
    swap_usage.set(sample.memory.swap_usage)
    swap_total.set(sample.memory.swap_total)

    # Fan metrics — one gauge per fan with label
    for i, fan in enumerate(sample.fans):
        fan_rpm.labels(fan=str(i)).set(fan.rpm)

    # Thermal throttling: 1 if macOS thermal or performance warning is active (from pmset -g therm)
    thermal_throttling.set(
        1 if (sample.thermal_warning or sample.performance_warning) else 0
    )

    # Power source: clear all labels, set the active one
    power_source.clear()
    power_source.labels(source=sample.power_source).set(1)

    # Battery percentage: 100 if on AC, actual pct if on battery
    battery_pct.set(sample.battery_pct if sample.battery_pct is not None else 100.0)

    # Thermal pressure warnings
    thermal_warning.set(1 if sample.thermal_warning else 0)
    performance_warning.set(1 if sample.performance_warning else 0)

    # Uptime
    uptime_seconds.set(time.monotonic() - _start_time)


def update_model_metrics(models: list[OllamaLoadedModel]) -> None:
    """Update all model-related Prometheus gauges from a list of loaded models."""
    loaded_models_count.set(len(models))

    # Track which models are currently loaded so we can clear stale labels
    loaded_names: set[str] = set()

    # Set metrics for each loaded model
    for model in models:
        loaded_names.add(model.name)
        model_size_bytes.labels(model=model.name).set(model.size)
        model_vram_bytes.labels(model=model.name).set(model.size_vram)
        model_loaded.labels(model=model.name).set(1)

    # Clear metrics for models that are no longer loaded
    # Iterate over all known labels and set stale ones to 0
    for sample in model_loaded.collect():
        for s in sample.samples:
            m = s.labels.get("model", "")
            if m and m not in loaded_names:
                model_loaded.labels(model=m).set(0)
                # Also clear inference gauges for unloaded models
                tokens_per_second.labels(model=m).set(0)
                inference_duration_seconds.labels(model=m).set(0)


def update_inference_metrics(
    rows: list[dict],
) -> None:
    """Update inference-related Prometheus gauges from SQLite inference data.

    Args:
        rows: List of dicts with keys model_name, tokens_per_second, eval_duration_seconds
              (as returned by Database.get_latest_inference_metrics()).
    """
    for row in rows:
        model = row.get("model_name", "")
        tok_s = row.get("tokens_per_second")
        dur_s = row.get("eval_duration_seconds")
        if model and tok_s is not None:
            tokens_per_second.labels(model=model).set(tok_s)
        if model and dur_s is not None:
            inference_duration_seconds.labels(model=model).set(dur_s)


# ---------------------------------------------------------------------------
# Metrics HTTP server
# ---------------------------------------------------------------------------


async def _handle_metrics(request: web.Request) -> web.Response:
    """Handle /metrics requests by generating the latest Prometheus output."""
    loop = asyncio.get_running_loop()
    data = await loop.run_in_executor(None, generate_latest)
    return web.Response(
        body=data,
        content_type="text/plain",
        charset="utf-8",
    )


async def start_metrics_server(port: int) -> web.TCPSite:
    """Start an aiohttp web server with a /metrics endpoint.

    Returns the TCPSite so the caller can await site.start() and later
    stop the server gracefully.
    """
    app = web.Application()
    app.router.add_get("/metrics", _handle_metrics)
    app.router.add_get("/health", _handle_health)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    logger.info("Prometheus metrics server started on port %d", port)
    return site


async def _handle_health(request: web.Request) -> web.Response:
    """Simple health check endpoint."""
    return web.json_response({"status": "ok"})
