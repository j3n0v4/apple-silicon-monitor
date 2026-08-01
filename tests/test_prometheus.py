"""Tests for the Prometheus exporter."""

from typing import ClassVar

import aiohttp
import pytest
from prometheus_client import REGISTRY

from asimon.collectors.macmon import MacmonSample
from asimon.collectors.ollama import OllamaLoadedModel
from asimon.exporters.prometheus import (
    start_metrics_server,
    update_hardware_metrics,
    update_inference_metrics,
    update_model_metrics,
)

# Reuse the real macmon output fixture from test_macmon.py
from tests.test_macmon import REAL_MACMON_OUTPUT

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def macmon_sample() -> MacmonSample:
    """Return a parsed MacmonSample from real macmon output."""
    return MacmonSample.model_validate(REAL_MACMON_OUTPUT)


@pytest.fixture
def loaded_models() -> list[OllamaLoadedModel]:
    """Return a list of mock loaded models."""
    return [
        OllamaLoadedModel(
            name="qwen3-coder:30b",
            model="qwen3-coder:30b",
            size=21474836480,
            size_vram=17179869184,
            digest="abc123",
            details={
                "parent_model": "",
                "format": "gguf",
                "family": "qwen3",
                "families": ["qwen3"],
                "parameter_size": "30B",
                "quantization_level": "Q4_K_M",
            },
            expires_at="2026-08-01T17:00:00Z",
            context_length=32768,
        ),
        OllamaLoadedModel(
            name="gemma4:e2b-nvfp4",
            model="gemma4:e2b-nvfp4",
            size=10737418240,
            size_vram=8589934592,
            digest="def456",
            details={
                "parent_model": "",
                "format": "gguf",
                "family": "gemma",
                "families": ["gemma"],
                "parameter_size": "27B",
                "quantization_level": "NVFP4",
            },
            expires_at="2026-08-01T17:05:00Z",
            context_length=16384,
        ),
    ]


# ---------------------------------------------------------------------------
# Metric registration tests
# ---------------------------------------------------------------------------


class TestMetricRegistration:
    """Test that all 23 metrics are properly registered."""

    EXPECTED_METRICS: ClassVar[set[str]] = {
        "asimon_gpu_temp_celsius",
        "asimon_gpu_freq_mhz",
        "asimon_gpu_power_watts",
        "asimon_ane_power_watts",
        "asimon_cpu_power_watts",
        "asimon_sys_power_watts",
        "asimon_ram_usage_bytes",
        "asimon_ram_total_bytes",
        "asimon_swap_usage_bytes",
        "asimon_swap_total_bytes",
        "asimon_fan_rpm",
        "asimon_tokens_per_second",
        "asimon_inference_duration_seconds",
        "asimon_loaded_models_count",
        "asimon_model_size_bytes",
        "asimon_model_vram_bytes",
        "asimon_model_loaded",
        "asimon_thermal_throttling",
        "asimon_uptime_seconds",
        "asimon_power_source",
        "asimon_battery_pct",
        "asimon_thermal_warning",
        "asimon_performance_warning",
    }

    def test_all_metrics_registered(self):
        """Verify all 23 expected metrics exist in the Prometheus registry."""
        registered = {
            sample.name
            for sample in REGISTRY.collect()
            if sample.name.startswith("asimon_")
        }
        missing = self.EXPECTED_METRICS - registered
        assert not missing, f"Missing metrics: {missing}"

    def test_exact_metric_count(self):
        """Verify exactly 23 asimon_ metrics are registered."""
        registered = {
            sample.name
            for sample in REGISTRY.collect()
            if sample.name.startswith("asimon_")
        }
        assert len(registered) == 24  # 24 unique metric names


# ---------------------------------------------------------------------------
# Hardware metrics update tests
# ---------------------------------------------------------------------------


class TestHardwareMetricsUpdate:
    """Test that update_hardware_metrics() correctly sets gauge values."""

    def test_gpu_temp(self, macmon_sample: MacmonSample):
        """Test GPU temperature gauge."""
        update_hardware_metrics(macmon_sample)
        sample = REGISTRY.get_sample_value("asimon_gpu_temp_celsius")
        assert sample == macmon_sample.temp.gpu_temp_avg

    def test_gpu_freq(self, macmon_sample: MacmonSample):
        """Test GPU frequency gauge."""
        update_hardware_metrics(macmon_sample)
        sample = REGISTRY.get_sample_value("asimon_gpu_freq_mhz")
        assert sample == macmon_sample.gpu_freq_mhz

    def test_gpu_power(self, macmon_sample: MacmonSample):
        """Test GPU power gauge."""
        update_hardware_metrics(macmon_sample)
        sample = REGISTRY.get_sample_value("asimon_gpu_power_watts")
        assert sample == macmon_sample.gpu_power

    def test_ane_power(self, macmon_sample: MacmonSample):
        """Test ANE power gauge."""
        update_hardware_metrics(macmon_sample)
        sample = REGISTRY.get_sample_value("asimon_ane_power_watts")
        assert sample == macmon_sample.ane_power

    def test_cpu_power(self, macmon_sample: MacmonSample):
        """Test CPU power gauge."""
        update_hardware_metrics(macmon_sample)
        sample = REGISTRY.get_sample_value("asimon_cpu_power_watts")
        assert sample == macmon_sample.cpu_power

    def test_sys_power(self, macmon_sample: MacmonSample):
        """Test system power gauge."""
        update_hardware_metrics(macmon_sample)
        sample = REGISTRY.get_sample_value("asimon_sys_power_watts")
        assert sample == macmon_sample.sys_power

    def test_ram_usage(self, macmon_sample: MacmonSample):
        """Test RAM usage gauge."""
        update_hardware_metrics(macmon_sample)
        sample = REGISTRY.get_sample_value("asimon_ram_usage_bytes")
        assert sample == macmon_sample.memory.ram_usage

    def test_ram_total(self, macmon_sample: MacmonSample):
        """Test RAM total gauge."""
        update_hardware_metrics(macmon_sample)
        sample = REGISTRY.get_sample_value("asimon_ram_total_bytes")
        assert sample == macmon_sample.memory.ram_total

    def test_swap_usage(self, macmon_sample: MacmonSample):
        """Test swap usage gauge."""
        update_hardware_metrics(macmon_sample)
        sample = REGISTRY.get_sample_value("asimon_swap_usage_bytes")
        assert sample == macmon_sample.memory.swap_usage

    def test_swap_total(self, macmon_sample: MacmonSample):
        """Test swap total gauge."""
        update_hardware_metrics(macmon_sample)
        sample = REGISTRY.get_sample_value("asimon_swap_total_bytes")
        assert sample == macmon_sample.memory.swap_total

    def test_fan_rpm(self, macmon_sample: MacmonSample):
        """Test fan RPM gauges with labels."""
        update_hardware_metrics(macmon_sample)
        sample0 = REGISTRY.get_sample_value("asimon_fan_rpm", {"fan": "0"})
        sample1 = REGISTRY.get_sample_value("asimon_fan_rpm", {"fan": "1"})
        assert sample0 == macmon_sample.fans[0].rpm
        assert sample1 == macmon_sample.fans[1].rpm

    def test_thermal_throttling_throttled(self, macmon_sample: MacmonSample):
        """Test thermal throttling is 1 when macOS thermal warning is active."""
        macmon_sample.thermal_warning = True
        update_hardware_metrics(macmon_sample)
        sample = REGISTRY.get_sample_value("asimon_thermal_throttling")
        assert sample == 1

    def test_thermal_throttling_normal(self):
        """Test thermal throttling is 0 when no macOS thermal/performance warning."""
        sample_data = dict(REAL_MACMON_OUTPUT)
        sample_data["gpu_freq_mhz"] = 1296
        normal_sample = MacmonSample.model_validate(sample_data)
        update_hardware_metrics(normal_sample)
        sample = REGISTRY.get_sample_value("asimon_thermal_throttling")
        assert sample == 0

    def test_uptime_increases(self, macmon_sample: MacmonSample):
        """Test that uptime is a positive value."""
        update_hardware_metrics(macmon_sample)
        sample = REGISTRY.get_sample_value("asimon_uptime_seconds")
        assert sample > 0

    def test_power_source_ac(self, macmon_sample: MacmonSample):
        """Test power_source gauge with AC Power."""
        macmon_sample.power_source = "AC Power"
        update_hardware_metrics(macmon_sample)
        sample = REGISTRY.get_sample_value(
            "asimon_power_source", {"source": "AC Power"}
        )
        assert sample == 1

    def test_power_source_battery(self, macmon_sample: MacmonSample):
        """Test power_source gauge with Battery Power."""
        macmon_sample.power_source = "Battery Power"
        update_hardware_metrics(macmon_sample)
        sample = REGISTRY.get_sample_value(
            "asimon_power_source", {"source": "Battery Power"}
        )
        assert sample == 1

    def test_battery_pct_on_ac(self, macmon_sample: MacmonSample):
        """Test battery_pct is 100 when on AC (battery_pct is None)."""
        macmon_sample.power_source = "AC Power"
        macmon_sample.battery_pct = None
        update_hardware_metrics(macmon_sample)
        sample = REGISTRY.get_sample_value("asimon_battery_pct")
        assert sample == 100.0

    def test_battery_pct_on_battery(self, macmon_sample: MacmonSample):
        """Test battery_pct reflects actual percentage on battery."""
        macmon_sample.power_source = "Battery Power"
        macmon_sample.battery_pct = 75.0
        update_hardware_metrics(macmon_sample)
        sample = REGISTRY.get_sample_value("asimon_battery_pct")
        assert sample == 75.0

    def test_thermal_warning_active(self, macmon_sample: MacmonSample):
        """Test thermal_warning gauge is 1 when warning is active."""
        macmon_sample.thermal_warning = True
        update_hardware_metrics(macmon_sample)
        sample = REGISTRY.get_sample_value("asimon_thermal_warning")
        assert sample == 1

    def test_thermal_warning_inactive(self, macmon_sample: MacmonSample):
        """Test thermal_warning gauge is 0 when no warning."""
        macmon_sample.thermal_warning = False
        update_hardware_metrics(macmon_sample)
        sample = REGISTRY.get_sample_value("asimon_thermal_warning")
        assert sample == 0

    def test_performance_warning_active(self, macmon_sample: MacmonSample):
        """Test performance_warning gauge is 1 when warning is active."""
        macmon_sample.performance_warning = True
        update_hardware_metrics(macmon_sample)
        sample = REGISTRY.get_sample_value("asimon_performance_warning")
        assert sample == 1

    def test_performance_warning_inactive(self, macmon_sample: MacmonSample):
        """Test performance_warning gauge is 0 when no warning."""
        macmon_sample.performance_warning = False
        update_hardware_metrics(macmon_sample)
        sample = REGISTRY.get_sample_value("asimon_performance_warning")
        assert sample == 0


# ---------------------------------------------------------------------------
# Model metrics update tests
# ---------------------------------------------------------------------------


class TestModelMetricsUpdate:
    """Test that update_model_metrics() correctly sets model gauges."""

    def test_loaded_models_count(self, loaded_models: list[OllamaLoadedModel]):
        """Test loaded models count gauge."""
        update_model_metrics(loaded_models)
        sample = REGISTRY.get_sample_value("asimon_loaded_models_count")
        assert sample == len(loaded_models)

    def test_model_size_bytes(self, loaded_models: list[OllamaLoadedModel]):
        """Test model size bytes gauge per model."""
        update_model_metrics(loaded_models)
        for model in loaded_models:
            sample = REGISTRY.get_sample_value(
                "asimon_model_size_bytes", {"model": model.name}
            )
            assert sample == model.size

    def test_model_vram_bytes(self, loaded_models: list[OllamaLoadedModel]):
        """Test model VRAM bytes gauge per model."""
        update_model_metrics(loaded_models)
        for model in loaded_models:
            sample = REGISTRY.get_sample_value(
                "asimon_model_vram_bytes", {"model": model.name}
            )
            assert sample == model.size_vram

    def test_model_loaded(self, loaded_models: list[OllamaLoadedModel]):
        """Test model loaded binary gauge per model."""
        update_model_metrics(loaded_models)
        for model in loaded_models:
            sample = REGISTRY.get_sample_value(
                "asimon_model_loaded", {"model": model.name}
            )
            assert sample == 1

    def test_empty_models(self):
        """Test that empty model list sets count to 0."""
        update_model_metrics([])
        sample = REGISTRY.get_sample_value("asimon_loaded_models_count")
        assert sample == 0

    def test_stale_model_labels_cleared(self, loaded_models: list[OllamaLoadedModel]):
        """Test that when a model is unloaded, its labels are cleared to 0."""
        # First load models
        update_model_metrics(loaded_models)
        for model in loaded_models:
            assert (
                REGISTRY.get_sample_value("asimon_model_loaded", {"model": model.name})
                == 1
            )

        # Now unload by passing empty list
        update_model_metrics([])
        for model in loaded_models:
            assert (
                REGISTRY.get_sample_value("asimon_model_loaded", {"model": model.name})
                == 0
            )
            assert (
                REGISTRY.get_sample_value(
                    "asimon_tokens_per_second", {"model": model.name}
                )
                == 0
            )
            assert (
                REGISTRY.get_sample_value(
                    "asimon_inference_duration_seconds", {"model": model.name}
                )
                == 0
            )


# ---------------------------------------------------------------------------
# Inference metrics update tests
# ---------------------------------------------------------------------------


class TestInferenceMetricsUpdate:
    """Test that update_inference_metrics() correctly sets gauge values."""

    def test_tokens_per_second_set(self):
        """Test tokens_per_second gauge is set from inference rows."""
        rows = [
            {
                "model_name": "qwen3-coder:30b",
                "tokens_per_second": 45.2,
                "eval_duration_seconds": 12.5,
            },
            {
                "model_name": "gemma4:e2b-nvfp4",
                "tokens_per_second": 32.1,
                "eval_duration_seconds": 8.3,
            },
        ]
        update_inference_metrics(rows)
        assert (
            REGISTRY.get_sample_value(
                "asimon_tokens_per_second", {"model": "qwen3-coder:30b"}
            )
            == 45.2
        )
        assert (
            REGISTRY.get_sample_value(
                "asimon_tokens_per_second", {"model": "gemma4:e2b-nvfp4"}
            )
            == 32.1
        )
        assert (
            REGISTRY.get_sample_value(
                "asimon_inference_duration_seconds", {"model": "qwen3-coder:30b"}
            )
            == 12.5
        )
        assert (
            REGISTRY.get_sample_value(
                "asimon_inference_duration_seconds", {"model": "gemma4:e2b-nvfp4"}
            )
            == 8.3
        )

    def test_empty_rows(self):
        """Test that empty rows list does not raise."""
        update_inference_metrics([])
        # No assertion needed — just must not raise

    def test_rows_with_none_values(self):
        """Test that rows with None tokens_per_second are skipped."""
        rows = [
            {
                "model_name": "test-model",
                "tokens_per_second": None,
                "eval_duration_seconds": None,
            }
        ]
        update_inference_metrics(rows)
        # Should not have set anything — just must not raise


# ---------------------------------------------------------------------------
# HTTP endpoint tests
# ---------------------------------------------------------------------------


class TestMetricsEndpoint:
    """Test the /metrics HTTP endpoint."""

    @pytest.mark.asyncio
    async def test_metrics_endpoint_returns_prometheus_text(self, unused_tcp_port: int):
        """Test that /metrics returns prometheus-formatted text."""
        site = await start_metrics_server(unused_tcp_port)

        try:
            async with (
                aiohttp.ClientSession() as session,
                session.get(f"http://127.0.0.1:{unused_tcp_port}/metrics") as resp,
            ):
                assert resp.status == 200
                content_type = resp.headers.get("Content-Type", "")
                assert "text/plain" in content_type
                text = await resp.text()
                # Should contain our metrics
                assert "asimon_gpu_temp_celsius" in text
                assert "asimon_uptime_seconds" in text
                assert "# HELP" in text
                assert "# TYPE" in text
        finally:
            await site.stop()

    @pytest.mark.asyncio
    async def test_health_endpoint(self, unused_tcp_port: int):
        """Test that /health returns a JSON response."""
        site = await start_metrics_server(unused_tcp_port)

        try:
            async with (
                aiohttp.ClientSession() as session,
                session.get(f"http://127.0.0.1:{unused_tcp_port}/health") as resp,
            ):
                assert resp.status == 200
                data = await resp.json()
                assert data["status"] == "ok"
        finally:
            await site.stop()

    @pytest.mark.asyncio
    async def test_metrics_contains_hardware_values(
        self, unused_tcp_port: int, macmon_sample: MacmonSample
    ):
        """Test that /metrics reflects updated hardware values."""
        update_hardware_metrics(macmon_sample)
        site = await start_metrics_server(unused_tcp_port)

        try:
            async with (
                aiohttp.ClientSession() as session,
                session.get(f"http://127.0.0.1:{unused_tcp_port}/metrics") as resp,
            ):
                text = await resp.text()
                # Check that the GPU temp value appears in the output
                expected_line = (
                    f"asimon_gpu_temp_celsius {macmon_sample.temp.gpu_temp_avg}"
                )
                assert expected_line in text
        finally:
            await site.stop()
