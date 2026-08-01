"""Alert checking logic for Apple Silicon Monitor.

Checks hardware and inference conditions and generates alerts
that are stored in the database.
"""

import logging
from datetime import UTC, datetime

from pydantic import BaseModel, Field

from asimon.collectors.macmon import MacmonSample
from asimon.collectors.ollama import OllamaLoadedModel
from asimon.storage.db import Database

logger = logging.getLogger(__name__)


class Alert(BaseModel):
    """A single alert event."""

    type: str = Field(..., description="Alert type identifier")
    message: str = Field(..., description="Human-readable alert message")
    severity: str = Field(
        default="warning", description="Severity level: info, warning, critical"
    )


class AlertChecker:
    """Check hardware and inference conditions and generate alerts."""

    def __init__(self, db: Database):
        self.db = db

    async def check_thermal_throttle(self, sample: MacmonSample) -> list[Alert]:
        """macOS thermal or performance warning active = thermal throttling.

        Uses pmset -g therm thermal/performance warnings (authoritative macOS
        thermal state) instead of GPU frequency, which idles at 100-400 MHz
        on Apple Silicon and is NOT an indicator of throttling.
        """
        if sample.thermal_warning or sample.performance_warning:
            reasons = []
            if sample.thermal_warning:
                reasons.append("thermal warning")
            if sample.performance_warning:
                reasons.append("performance warning")
            return [
                Alert(
                    type="thermal_throttle",
                    message=f"macOS throttling active: {' + '.join(reasons)}",
                    severity="critical",
                )
            ]
        return []

    async def check_high_temp(
        self, sample: MacmonSample, threshold: float = 90.0
    ) -> list[Alert]:
        """GPU temp > threshold."""
        if sample.temp.gpu_temp_avg > threshold:
            return [
                Alert(
                    type="high_temp",
                    message=f"GPU temp {sample.temp.gpu_temp_avg:.1f}°C exceeds {threshold}°C",
                    severity="warning",
                )
            ]
        return []

    async def check_high_swap(
        self, sample: MacmonSample, warning_gb: float = 5.0, critical_gb: float = 10.0
    ) -> list[Alert]:
        """Swap usage > threshold in GB.

        Uses absolute GB thresholds instead of percentage, since swap
        total varies between systems.

        Args:
            sample: The hardware sample to check.
            warning_gb: Warning threshold in GB (default: 5 GB).
            critical_gb: Critical threshold in GB (default: 10 GB).
        """
        if sample.memory.swap_usage <= 0:
            return []

        swap_gb = sample.memory.swap_usage / (1024**3)
        alerts: list[Alert] = []

        if swap_gb > critical_gb:
            alerts.append(
                Alert(
                    type="high_swap",
                    message=f"Swap usage {swap_gb:.1f} GB exceeds critical threshold {critical_gb:.0f} GB",
                    severity="critical",
                )
            )
        elif swap_gb > warning_gb:
            alerts.append(
                Alert(
                    type="high_swap",
                    message=f"Swap usage {swap_gb:.1f} GB exceeds warning threshold {warning_gb:.0f} GB",
                    severity="warning",
                )
            )

        return alerts

    async def check_stuck_model(self, models: list[OllamaLoadedModel]) -> list[Alert]:
        """Model loaded for > 30 minutes = stuck (wastes VRAM).

        Checks if a model's expires_at is more than 30 minutes from now,
        indicating it has been loaded for an extended period without being
        actively used.
        """
        alerts: list[Alert] = []
        now = datetime.now(UTC)
        for model in models:
            if not model.expires_at:
                continue
            try:
                expires = datetime.fromisoformat(model.expires_at)
                # If expires_at is far in the future, the model has been
                # loaded for a while (Ollama extends expires_at on each
                # request, so a far-future expiry means it was loaded
                # long ago and kept alive)
                time_until_expiry = (expires - now).total_seconds()
                if time_until_expiry > 1800:  # 30 minutes = 1800 seconds
                    alerts.append(
                        Alert(
                            type="stuck_model",
                            message=f"Model {model.name} loaded > 30 min — expires in {int(time_until_expiry / 60)} min",
                            severity="info",
                        )
                    )
            except (ValueError, TypeError):
                logger.debug("Could not parse expires_at: %s", model.expires_at)
        return alerts

    async def check_power_source(self, sample: MacmonSample) -> list[Alert]:
        """Running on battery power = potential performance impact."""
        if sample.power_source == "Battery Power":
            return [
                Alert(
                    type="battery_power",
                    message="Running on battery power — performance may be reduced",
                    severity="info",
                )
            ]
        return []

    async def check_all(
        self, sample: MacmonSample, models: list[OllamaLoadedModel]
    ) -> list[Alert]:
        """Run all checks and write alerts to DB."""
        alerts: list[Alert] = []
        alerts.extend(await self.check_thermal_throttle(sample))
        alerts.extend(await self.check_high_temp(sample))
        alerts.extend(await self.check_high_swap(sample))
        alerts.extend(await self.check_stuck_model(models))
        alerts.extend(await self.check_power_source(sample))

        for alert in alerts:
            await self.db.insert_alert(
                {
                    "alert_name": alert.type,
                    "message": alert.message,
                    "severity": alert.severity,
                    "metric_value": None,
                    "threshold": None,
                }
            )

        return alerts
