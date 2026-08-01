"""Tests for the macmon collector."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from asimon.collectors.macmon import (
    MacmonCollector,
    MacmonSample,
)

# Real macmon JSON output captured from M1 Max
REAL_MACMON_OUTPUT = {
    "all_power": 2.5006771087646484,
    "ane_power": 0.0,
    "cpu_active_ratio": 0.1903354674577713,
    "cpu_power": 1.3708608150482178,
    "cpu_usage_pct": 0.11301743984222412,
    "cpu_usage_ratio": 0.11301743984222412,
    "ecpu_active_ratio": 0.2199309766292572,
    "ecpu_cores": [
        {
            "active_ratio": 0.23338988423347473,
            "core_id": 0,
            "die_id": 0,
            "freq_mhz": 1399,
            "usage_ratio": 0.1582687348127365,
        },
        {
            "active_ratio": 0.20647208392620087,
            "core_id": 10,
            "die_id": 0,
            "freq_mhz": 1420,
            "usage_ratio": 0.14210067689418793,
        },
    ],
    "ecpu_freq_mhz": 1408,
    "ecpu_usage": [1408, 0.15018470585346222],
    "ecpu_usage_ratio": 0.15018470585346222,
    "fans": [
        {"max_rpm": 5779, "name": "fan0", "rpm": 2311},
        {"max_rpm": 6241, "name": "fan1", "rpm": 2514},
    ],
    "gpu_active_ratio": 0.6921621561050415,
    "gpu_freq_mhz": 388,
    "gpu_power": 1.1298161745071411,
    "gpu_ram_power": 0.040669530630111694,
    "gpu_usage": [388, 0.2072213888168335],
    "gpu_usage_ratio": 0.2072213888168335,
    "memory": {
        "ram_total": 68719476736,
        "ram_usage": 28706603008,
        "swap_total": 7516192768,
        "swap_usage": 5701894144,
    },
    "pcpu_active_ratio": 0.18293659389019012,
    "pcpu_cores": [
        {
            "active_ratio": 0.5556453466415405,
            "core_id": 0,
            "die_id": 0,
            "freq_mhz": 1805,
            "usage_ratio": 0.310747891664505,
        },
    ],
    "pcpu_freq_mhz": 1829,
    "pcpu_usage": [1829, 0.1037256270647049],
    "pcpu_usage_ratio": 0.1037256270647049,
    "ram_power": 1.7497817277908325,
    "sys_power": 18.952476501464844,
    "temp": {
        "cpu_temp_avg": 55.03165054321289,
        "gpu_temp_avg": 52.1805419921875,
    },
    "timestamp": "2026-08-01T16:36:35.366245+00:00",
}


class TestMacmonSampleParsing:
    """Test that macmon JSON parses correctly into Pydantic models."""

    def test_parse_real_macmon_output(self):
        """Parse the real macmon JSON output."""
        sample = MacmonSample.model_validate(REAL_MACMON_OUTPUT)

        # Top-level fields
        assert sample.cpu_power == 1.3708608150482178
        assert sample.gpu_power == 1.1298161745071411
        assert sample.ane_power == 0.0
        assert sample.sys_power == 18.952476501464844
        assert sample.gpu_freq_mhz == 388
        assert sample.pcpu_freq_mhz == 1829
        assert sample.ecpu_freq_mhz == 1408
        assert sample.timestamp == "2026-08-01T16:36:35.366245+00:00"

        # Nested temp
        assert sample.temp.cpu_temp_avg == 55.03165054321289
        assert sample.temp.gpu_temp_avg == 52.1805419921875

        # Nested memory
        assert sample.memory.ram_usage == 28706603008
        assert sample.memory.ram_total == 68719476736
        assert sample.memory.swap_usage == 5701894144
        assert sample.memory.swap_total == 7516192768

        # Fans
        assert len(sample.fans) == 2
        assert sample.fans[0].rpm == 2311
        assert sample.fans[0].max_rpm == 5779
        assert sample.fans[1].rpm == 2514
        assert sample.fans[1].max_rpm == 6241

        # New power/thermal fields have defaults
        assert sample.power_source == "Unknown"
        assert sample.battery_pct is None
        assert sample.thermal_warning is False
        assert sample.performance_warning is False

    def test_model_dump_roundtrip(self):
        """Test that model_dump() produces expected output."""
        sample = MacmonSample.model_validate(REAL_MACMON_OUTPUT)
        dumped = sample.model_dump()

        assert dumped["cpu_power"] == 1.3708608150482178
        assert dumped["gpu_power"] == 1.1298161745071411
        assert dumped["temp"]["cpu_temp_avg"] == 55.03165054321289
        assert dumped["memory"]["ram_usage"] == 28706603008
        assert dumped["fans"][0]["rpm"] == 2311


# ---------------------------------------------------------------------------
# PowerStatus parsing tests
# ---------------------------------------------------------------------------


class TestPowerStatusParsing:
    """Test parsing of pmset -g batt output."""

    def test_ac_power_charged(self):
        """Parse AC Power output with battery charged."""
        output = """Now drawing from 'AC Power'
 -InternalBattery-0 (id=12345678) 100%; charged; 0:00 remaining"""
        status = MacmonCollector._parse_power_status(output)
        assert status.source == "AC Power"
        assert status.battery_pct == 100.0
        assert status.time_remaining == "0:00"
        assert status.charging is False

    def test_battery_power_discharging(self):
        """Parse Battery Power output with discharging battery."""
        output = """Now drawing from 'Battery Power'
 -InternalBattery-0 (id=12345678) 75%; discharging; 3:24 remaining"""
        status = MacmonCollector._parse_power_status(output)
        assert status.source == "Battery Power"
        assert status.battery_pct == 75.0
        assert status.time_remaining == "3:24"
        assert status.charging is False

    def test_battery_charging(self):
        """Parse Battery Power output while charging."""
        output = """Now drawing from 'AC Power'
 -InternalBattery-0 (id=12345678) 82%; charging; 1:15 remaining"""
        status = MacmonCollector._parse_power_status(output)
        assert status.source == "AC Power"
        assert status.battery_pct == 82.0
        assert status.time_remaining == "1:15"
        assert status.charging is True

    def test_battery_no_estimate(self):
        """Parse output with no time estimate."""
        output = """Now drawing from 'Battery Power'
 -InternalBattery-0 (id=12345678) 50%; discharging; (no estimate)"""
        status = MacmonCollector._parse_power_status(output)
        assert status.source == "Battery Power"
        assert status.battery_pct == 50.0
        assert status.time_remaining is None
        assert status.charging is False

    def test_ac_attached_not_charging(self):
        """Parse AC attached but not charging."""
        output = """Now drawing from 'AC Power'
 -InternalBattery-0 (id=12345678) 95%; AC attached; not charging"""
        status = MacmonCollector._parse_power_status(output)
        assert status.source == "AC Power"
        assert status.battery_pct == 95.0
        assert status.time_remaining is None
        assert status.charging is False

    def test_empty_output(self):
        """Parse empty output."""
        status = MacmonCollector._parse_power_status("")
        assert status.source == "Unknown"
        assert status.battery_pct is None
        assert status.time_remaining is None
        assert status.charging is False


# ---------------------------------------------------------------------------
# ThermalPressure parsing tests
# ---------------------------------------------------------------------------


class TestThermalPressureParsing:
    """Test parsing of pmset -g therm output."""

    def test_no_thermal_warning(self):
        """Parse output with no thermal warning recorded."""
        output = """Note: No thermal warning level has been recorded.
Note: No performance warning level has been recorded.
Note: CPU Power status has been recorded."""
        pressure = MacmonCollector._parse_thermal_pressure(output)
        assert pressure.thermal_warning is False
        assert pressure.performance_warning is False
        assert pressure.cpu_power_warning is False

    def test_thermal_warning_active(self):
        """Parse output with active thermal warning."""
        output = """Thermal Warning Level: 1
Performance Warning Level: 0
CPU Power Status: 0"""
        pressure = MacmonCollector._parse_thermal_pressure(output)
        assert pressure.thermal_warning is True
        assert pressure.performance_warning is False
        assert pressure.cpu_power_warning is False

    def test_all_warnings_active(self):
        """Parse output with all warnings active."""
        output = """Thermal Warning Level: 2
Performance Warning Level: 1
CPU Power Status: 1"""
        pressure = MacmonCollector._parse_thermal_pressure(output)
        assert pressure.thermal_warning is True
        assert pressure.performance_warning is True
        assert pressure.cpu_power_warning is True

    def test_empty_output(self):
        """Parse empty output."""
        pressure = MacmonCollector._parse_thermal_pressure("")
        assert pressure.thermal_warning is False
        assert pressure.performance_warning is False
        assert pressure.cpu_power_warning is False


# ---------------------------------------------------------------------------
# MacmonCollector tests
# ---------------------------------------------------------------------------


def _make_mock_proc(returncode: int, stdout: bytes, stderr: bytes = b""):
    """Helper to create a mock subprocess."""
    mock = AsyncMock()
    mock.returncode = returncode
    mock.communicate = AsyncMock(return_value=(stdout, stderr))
    return mock


class TestMacmonCollector:
    """Test the MacmonCollector class."""

    @pytest.mark.asyncio
    async def test_collect_success(self):
        """Test successful collection with mock subprocess."""
        collector = MacmonCollector(macmon_path="/opt/homebrew/bin/macmon")

        # Mock the three subprocess calls: macmon, pmset -g batt, pmset -g therm
        macmon_proc = _make_mock_proc(0, json.dumps(REAL_MACMON_OUTPUT).encode())
        pmset_batt_proc = _make_mock_proc(
            0,
            b"Now drawing from 'AC Power'\n -InternalBattery-0 (id=12345678) 100%; charged; 0:00 remaining",
        )
        pmset_therm_proc = _make_mock_proc(
            0,
            b"Note: No thermal warning level has been recorded.\nNote: No performance warning level has been recorded.",
        )

        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=[macmon_proc, pmset_batt_proc, pmset_therm_proc],
        ):
            sample = await collector.collect()

        assert sample is not None
        assert sample.gpu_power == 1.1298161745071411
        assert sample.temp.gpu_temp_avg == 52.1805419921875
        # Power/thermal enrichment
        assert sample.power_source == "AC Power"
        assert sample.battery_pct == 100.0
        assert sample.thermal_warning is False
        assert sample.performance_warning is False

    @pytest.mark.asyncio
    async def test_collect_macmon_not_found(self):
        """Test graceful handling when macmon binary is missing."""
        collector = MacmonCollector(macmon_path="/nonexistent/macmon")

        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=FileNotFoundError("macmon not found"),
        ):
            sample = await collector.collect()

        assert sample is None

    @pytest.mark.asyncio
    async def test_collect_json_parse_error(self):
        """Test graceful handling of invalid JSON output."""
        collector = MacmonCollector(macmon_path="/opt/homebrew/bin/macmon")

        macmon_proc = _make_mock_proc(0, b"not valid json")
        pmset_batt_proc = _make_mock_proc(
            0,
            b"Now drawing from 'AC Power'\n -InternalBattery-0 (id=12345678) 100%; charged; 0:00 remaining",
        )
        pmset_therm_proc = _make_mock_proc(
            0,
            b"Note: No thermal warning level has been recorded.",
        )

        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=[macmon_proc, pmset_batt_proc, pmset_therm_proc],
        ):
            sample = await collector.collect()

        assert sample is None

    @pytest.mark.asyncio
    async def test_collect_nonzero_exit(self):
        """Test graceful handling when macmon exits with error."""
        collector = MacmonCollector(macmon_path="/opt/homebrew/bin/macmon")

        macmon_proc = _make_mock_proc(1, b"", b"error message")
        pmset_batt_proc = _make_mock_proc(
            0,
            b"Now drawing from 'AC Power'\n -InternalBattery-0 (id=12345678) 100%; charged; 0:00 remaining",
        )
        pmset_therm_proc = _make_mock_proc(
            0,
            b"Note: No thermal warning level has been recorded.",
        )

        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=[macmon_proc, pmset_batt_proc, pmset_therm_proc],
        ):
            sample = await collector.collect()

        assert sample is None

    @pytest.mark.asyncio
    async def test_collect_pmset_not_found(self):
        """Test graceful handling when pmset is not available (Linux)."""
        collector = MacmonCollector(macmon_path="/opt/homebrew/bin/macmon")

        macmon_proc = _make_mock_proc(0, json.dumps(REAL_MACMON_OUTPUT).encode())

        # pmset not found — FileNotFoundError
        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=[
                macmon_proc,
                FileNotFoundError("pmset not found"),
                FileNotFoundError("pmset not found"),
            ],
        ):
            sample = await collector.collect()

        assert sample is not None
        # Defaults should be used when pmset is unavailable
        assert sample.power_source == "Unknown"
        assert sample.battery_pct is None
        assert sample.thermal_warning is False
        assert sample.performance_warning is False

    @pytest.mark.asyncio
    async def test_collect_stream(self):
        """Test the collect_stream async generator."""
        collector = MacmonCollector(macmon_path="/opt/homebrew/bin/macmon")

        macmon_proc = _make_mock_proc(0, json.dumps(REAL_MACMON_OUTPUT).encode())
        pmset_batt_proc = _make_mock_proc(
            0,
            b"Now drawing from 'AC Power'\n -InternalBattery-0 (id=12345678) 100%; charged; 0:00 remaining",
        )
        pmset_therm_proc = _make_mock_proc(
            0,
            b"Note: No thermal warning level has been recorded.",
        )

        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=[macmon_proc, pmset_batt_proc, pmset_therm_proc],
        ):
            samples = []
            async for sample in collector.collect_stream(interval=0.01, max_samples=1):
                samples.append(sample)

        assert len(samples) == 1
        assert samples[0].gpu_power == 1.1298161745071411
        assert samples[0].power_source == "AC Power"

    @pytest.mark.asyncio
    async def test_get_power_status_ac(self):
        """Test get_power_status returns AC Power correctly."""
        collector = MacmonCollector()

        mock_proc = _make_mock_proc(
            0,
            b"Now drawing from 'AC Power'\n -InternalBattery-0 (id=12345678) 100%; charged; 0:00 remaining",
        )

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            status = await collector.get_power_status()

        assert status.source == "AC Power"
        assert status.battery_pct == 100.0
        assert status.time_remaining == "0:00"
        assert status.charging is False

    @pytest.mark.asyncio
    async def test_get_power_status_battery(self):
        """Test get_power_status returns Battery Power correctly."""
        collector = MacmonCollector()

        mock_proc = _make_mock_proc(
            0,
            b"Now drawing from 'Battery Power'\n -InternalBattery-0 (id=12345678) 75%; discharging; 3:24 remaining",
        )

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            status = await collector.get_power_status()

        assert status.source == "Battery Power"
        assert status.battery_pct == 75.0
        assert status.time_remaining == "3:24"
        assert status.charging is False

    @pytest.mark.asyncio
    async def test_get_power_status_not_found(self):
        """Test get_power_status handles missing pmset."""
        collector = MacmonCollector()

        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=FileNotFoundError("pmset not found"),
        ):
            status = await collector.get_power_status()

        assert status.source == "Unknown"
        assert status.battery_pct is None

    @pytest.mark.asyncio
    async def test_get_thermal_pressure_active(self):
        """Test get_thermal_pressure with active warnings."""
        collector = MacmonCollector()

        mock_proc = _make_mock_proc(
            0,
            b"Thermal Warning Level: 1\nPerformance Warning Level: 0\nCPU Power Status: 0",
        )

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            pressure = await collector.get_thermal_pressure()

        assert pressure.thermal_warning is True
        assert pressure.performance_warning is False
        assert pressure.cpu_power_warning is False

    @pytest.mark.asyncio
    async def test_get_thermal_pressure_none(self):
        """Test get_thermal_pressure with no warnings."""
        collector = MacmonCollector()

        mock_proc = _make_mock_proc(
            0,
            b"Note: No thermal warning level has been recorded.\nNote: No performance warning level has been recorded.",
        )

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            pressure = await collector.get_thermal_pressure()

        assert pressure.thermal_warning is False
        assert pressure.performance_warning is False
        assert pressure.cpu_power_warning is False

    @pytest.mark.asyncio
    async def test_get_thermal_pressure_not_found(self):
        """Test get_thermal_pressure handles missing pmset."""
        collector = MacmonCollector()

        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=FileNotFoundError("pmset not found"),
        ):
            pressure = await collector.get_thermal_pressure()

        assert pressure.thermal_warning is False
        assert pressure.performance_warning is False
