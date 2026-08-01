"""Tests for the alert checker."""

from datetime import UTC, datetime, timedelta

import pytest

from asimon.alerts.checker import Alert, AlertChecker
from asimon.collectors.macmon import MacmonMemory, MacmonSample, MacmonTemp
from asimon.collectors.ollama import OllamaLoadedModel, OllamaModelDetails


def _make_sample(
    gpu_freq_mhz: int = 1296,
    gpu_temp: float = 52.0,
    swap_usage: int = 1000000000,
    swap_total: int = 7516192768,
    power_source: str = "AC Power",
    thermal_warning: bool = False,
    performance_warning: bool = False,
) -> MacmonSample:
    """Create a MacmonSample with the given parameters."""
    return MacmonSample(
        cpu_power=1.5,
        gpu_power=28.4,
        ane_power=0.0,
        sys_power=30.0,
        gpu_freq_mhz=gpu_freq_mhz,
        pcpu_freq_mhz=1800,
        ecpu_freq_mhz=1400,
        timestamp="2026-08-01T12:00:00+00:00",
        temp=MacmonTemp(cpu_temp_avg=55.0, gpu_temp_avg=gpu_temp),
        memory=MacmonMemory(
            ram_usage=30000000000,
            ram_total=68719476736,
            swap_usage=swap_usage,
            swap_total=swap_total,
        ),
        fans=[],
        power_source=power_source,
        battery_pct=100.0 if power_source == "AC Power" else 75.0,
        thermal_warning=thermal_warning,
        performance_warning=performance_warning,
    )


def _make_model(
    name: str = "qwen3-coder:30b",
    expires_at: str | None = None,
) -> OllamaLoadedModel:
    """Create an OllamaLoadedModel with the given parameters."""
    if expires_at is None:
        expires_at = (datetime.now(UTC) + timedelta(minutes=5)).isoformat()
    return OllamaLoadedModel(
        name=name,
        model=name,
        size=18556700761,
        size_vram=18556700761,
        digest="abc123",
        details=OllamaModelDetails(
            parent_model="",
            format="gguf",
            family="qwen2",
            families=None,
            parameter_size="30B",
            quantization_level="Q4_K_M",
        ),
        expires_at=expires_at,
        context_length=262144,
    )


class TestAlertModel:
    """Test the Alert Pydantic model."""

    def test_alert_creation(self):
        """Test creating an Alert with all fields."""
        alert = Alert(
            type="thermal_throttle",
            message="GPU throttled to 388 MHz",
            severity="critical",
        )
        assert alert.type == "thermal_throttle"
        assert alert.message == "GPU throttled to 388 MHz"
        assert alert.severity == "critical"

    def test_alert_default_severity(self):
        """Test that severity defaults to 'warning'."""
        alert = Alert(
            type="high_temp",
            message="GPU temp 95.0°C exceeds 90°C",
        )
        assert alert.severity == "warning"


class TestAlertChecker:
    """Test the AlertChecker class."""

    @pytest.mark.asyncio
    async def test_thermal_throttle_alert(self, db):
        """Test thermal throttle alert when macOS thermal warning is active."""
        checker = AlertChecker(db)
        sample = _make_sample(thermal_warning=True)
        alerts = await checker.check_thermal_throttle(sample)
        assert len(alerts) == 1
        assert alerts[0].type == "thermal_throttle"
        assert alerts[0].severity == "critical"
        assert "thermal warning" in alerts[0].message

    @pytest.mark.asyncio
    async def test_no_thermal_throttle_alert(self, db):
        """Test no alert when no thermal/performance warning (even with low GPU freq)."""
        checker = AlertChecker(db)
        sample = _make_sample(gpu_freq_mhz=388)
        alerts = await checker.check_thermal_throttle(sample)
        assert len(alerts) == 0

    @pytest.mark.asyncio
    async def test_high_temp_alert(self, db):
        """Test high temp alert when GPU temp > 90°C."""
        checker = AlertChecker(db)
        sample = _make_sample(gpu_temp=95.0)
        alerts = await checker.check_high_temp(sample)
        assert len(alerts) == 1
        assert alerts[0].type == "high_temp"
        assert alerts[0].severity == "warning"
        assert "95.0" in alerts[0].message

    @pytest.mark.asyncio
    async def test_no_high_temp_alert(self, db):
        """Test no alert when GPU temp is normal."""
        checker = AlertChecker(db)
        sample = _make_sample(gpu_temp=52.0)
        alerts = await checker.check_high_temp(sample)
        assert len(alerts) == 0

    @pytest.mark.asyncio
    async def test_high_swap_alert(self, db):
        """Test high swap alert when swap usage > 5 GB (warning)."""
        checker = AlertChecker(db)
        # 6.5 GB out of 7.5 GB = exceeds 5 GB warning threshold
        sample = _make_sample(swap_usage=6500000000, swap_total=7516192768)
        alerts = await checker.check_high_swap(sample)
        assert len(alerts) == 1
        assert alerts[0].type == "high_swap"
        assert alerts[0].severity == "warning"
        assert "6.1 GB" in alerts[0].message or "6.0 GB" in alerts[0].message

    @pytest.mark.asyncio
    async def test_high_swap_critical(self, db):
        """Test high swap alert when swap usage > 10 GB (critical)."""
        checker = AlertChecker(db)
        # 12 GB out of 16 GB = exceeds 10 GB critical threshold
        sample = _make_sample(swap_usage=12000000000, swap_total=16000000000)
        alerts = await checker.check_high_swap(sample)
        assert len(alerts) == 1
        assert alerts[0].type == "high_swap"
        assert alerts[0].severity == "critical"
        assert "11.2 GB" in alerts[0].message or "11.1 GB" in alerts[0].message

    @pytest.mark.asyncio
    async def test_no_high_swap_alert(self, db):
        """Test no alert when swap usage is below 5 GB."""
        checker = AlertChecker(db)
        # 1 GB out of 7.5 GB = below warning threshold
        sample = _make_sample(swap_usage=1000000000, swap_total=7516192768)
        alerts = await checker.check_high_swap(sample)
        assert len(alerts) == 0

    @pytest.mark.asyncio
    async def test_high_swap_zero_usage(self, db):
        """Test no alert when swap usage is 0."""
        checker = AlertChecker(db)
        sample = _make_sample(swap_usage=0, swap_total=7516192768)
        alerts = await checker.check_high_swap(sample)
        assert len(alerts) == 0

    @pytest.mark.asyncio
    async def test_stuck_model_alert(self, db):
        """Test stuck model alert when model loaded > 30 min."""
        checker = AlertChecker(db)
        # expires_at 45 minutes from now = loaded for a while
        far_expiry = (datetime.now(UTC) + timedelta(minutes=45)).isoformat()
        model = _make_model(expires_at=far_expiry)
        alerts = await checker.check_stuck_model([model])
        assert len(alerts) == 1
        assert alerts[0].type == "stuck_model"
        assert alerts[0].severity == "info"
        assert "qwen3-coder:30b" in alerts[0].message

    @pytest.mark.asyncio
    async def test_no_stuck_model_alert(self, db):
        """Test no alert when model loaded recently."""
        checker = AlertChecker(db)
        # expires_at 5 minutes from now = recently loaded
        near_expiry = (datetime.now(UTC) + timedelta(minutes=5)).isoformat()
        model = _make_model(expires_at=near_expiry)
        alerts = await checker.check_stuck_model([model])
        assert len(alerts) == 0

    @pytest.mark.asyncio
    async def test_stuck_model_no_expiry(self, db):
        """Test no alert when model has no expires_at."""
        checker = AlertChecker(db)
        model = _make_model(expires_at=None)
        alerts = await checker.check_stuck_model([model])
        assert len(alerts) == 0

    @pytest.mark.asyncio
    async def test_battery_power_alert(self, db):
        """Test battery power alert when on battery."""
        checker = AlertChecker(db)
        sample = _make_sample(power_source="Battery Power")
        alerts = await checker.check_power_source(sample)
        assert len(alerts) == 1
        assert alerts[0].type == "battery_power"
        assert alerts[0].severity == "info"
        assert "battery" in alerts[0].message.lower()

    @pytest.mark.asyncio
    async def test_no_battery_power_alert(self, db):
        """Test no alert when on AC power."""
        checker = AlertChecker(db)
        sample = _make_sample(power_source="AC Power")
        alerts = await checker.check_power_source(sample)
        assert len(alerts) == 0

    @pytest.mark.asyncio
    async def test_no_alerts_when_normal(self, db):
        """Test that check_all returns no alerts under normal conditions."""
        checker = AlertChecker(db)
        sample = _make_sample()
        model = _make_model()
        alerts = await checker.check_all(sample, [model])
        assert len(alerts) == 0

    @pytest.mark.asyncio
    async def test_alerts_written_to_db(self, db):
        """Test that check_all writes alerts to the database."""
        checker = AlertChecker(db)
        # Trigger thermal throttle and battery power alerts
        sample = _make_sample(thermal_warning=True, power_source="Battery Power")
        model = _make_model()
        alerts = await checker.check_all(sample, [model])
        assert len(alerts) == 2

        # Verify alerts were written to DB
        conn = db._conn
        cursor = await conn.execute(
            "SELECT alert_name, severity FROM alerts ORDER BY id"
        )
        rows = await cursor.fetchall()
        assert len(rows) == 2
        alert_types = {row["alert_name"] for row in rows}
        assert "thermal_throttle" in alert_types
        assert "battery_power" in alert_types

    @pytest.mark.asyncio
    async def test_multiple_alerts(self, db):
        """Test that multiple alerts can fire simultaneously."""
        checker = AlertChecker(db)
        # Trigger thermal throttle, high temp, high swap, and battery power
        sample = _make_sample(
            thermal_warning=True,
            gpu_temp=95.0,
            swap_usage=7000000000,
            swap_total=7516192768,
            power_source="Battery Power",
        )
        model = _make_model()
        alerts = await checker.check_all(sample, [model])
        # Should have: thermal_throttle, high_temp, high_swap, battery_power
        assert len(alerts) == 4
        alert_types = {a.type for a in alerts}
        assert alert_types == {
            "thermal_throttle",
            "high_temp",
            "high_swap",
            "battery_power",
        }
