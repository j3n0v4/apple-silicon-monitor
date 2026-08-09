"""macmon collector — polls macmon pipe for hardware metrics.

Runs `/opt/homebrew/bin/macmon pipe --samples 1` as a subprocess and
parses the JSON output into a Pydantic model.
"""

import asyncio
import json
import logging
from collections.abc import AsyncGenerator

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class MacmonTemp(BaseModel):
    """Temperature readings from macmon."""

    cpu_temp_avg: float = Field(..., alias="cpu_temp_avg")
    gpu_temp_avg: float = Field(..., alias="gpu_temp_avg")


class MacmonMemory(BaseModel):
    """Memory readings from macmon."""

    ram_usage: int = Field(..., alias="ram_usage")
    ram_total: int = Field(..., alias="ram_total")
    swap_usage: int = Field(..., alias="swap_usage")
    swap_total: int = Field(..., alias="swap_total")


class MacmonFan(BaseModel):
    """Fan reading from macmon."""

    rpm: int = Field(..., alias="rpm")
    max_rpm: int = Field(..., alias="max_rpm")


class PowerStatus(BaseModel):
    """Power source status from pmset -g batt."""

    source: str = "Unknown"
    battery_pct: float | None = None
    time_remaining: str | None = None
    charging: bool = False


class ThermalPressure(BaseModel):
    """Thermal pressure status from pmset -g therm."""

    thermal_warning: bool = False
    performance_warning: bool = False
    cpu_power_warning: bool = False


class MacmonSample(BaseModel):
    """A single hardware sample from macmon."""

    cpu_power: float = Field(..., alias="cpu_power")
    gpu_power: float = Field(..., alias="gpu_power")
    ane_power: float = Field(..., alias="ane_power")
    sys_power: float = Field(..., alias="sys_power")
    gpu_freq_mhz: int = Field(..., alias="gpu_freq_mhz")
    pcpu_freq_mhz: int = Field(..., alias="pcpu_freq_mhz")
    ecpu_freq_mhz: int = Field(..., alias="ecpu_freq_mhz")
    timestamp: str = Field(..., alias="timestamp")
    temp: MacmonTemp = Field(..., alias="temp")
    memory: MacmonMemory = Field(..., alias="memory")
    fans: list[MacmonFan] = Field(..., alias="fans")
    power_source: str = "Unknown"
    battery_pct: float | None = None
    thermal_warning: bool = False
    performance_warning: bool = False

    model_config = {"populate_by_name": True}


class MacmonCollector:
    """Collects hardware metrics by running macmon pipe as a subprocess."""

    def __init__(self, macmon_path: str = "/opt/homebrew/bin/macmon"):
        self.macmon_path = macmon_path

    async def collect(self) -> MacmonSample | None:
        """Run macmon pipe once and return a parsed sample.

        Returns None if macmon is not found or JSON parsing fails.
        """
        try:
            proc = await asyncio.create_subprocess_exec(
                self.macmon_path,
                "pipe",
                "--samples",
                "1",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10.0)

            if proc.returncode != 0:
                logger.warning(
                    "macmon exited with code %d: %s",
                    proc.returncode,
                    stderr.decode().strip(),
                )
                return None

            data = json.loads(stdout.decode())
            sample = MacmonSample.model_validate(data)

            # Enrich with power source and thermal pressure data
            # These are best-effort — do not let pmset failures kill the sample
            try:
                power = await self.get_power_status()
                sample.power_source = power.source
                sample.battery_pct = power.battery_pct
            except Exception:  # noqa: BLE001
                logger.debug("Failed to get power status, using defaults")

            try:
                thermal = await self.get_thermal_pressure()
                sample.thermal_warning = thermal.thermal_warning
                sample.performance_warning = thermal.performance_warning
            except Exception:  # noqa: BLE001
                logger.debug("Failed to get thermal pressure, using defaults")

            return sample

        except FileNotFoundError:
            logger.error(
                "macmon not found at %s. Install with: brew install macmon",
                self.macmon_path,
            )
            return None
        except json.JSONDecodeError as e:
            logger.error("Failed to parse macmon JSON output: %s", e)
            return None
        except TimeoutError:
            logger.error("macmon subprocess timed out after 10 seconds")
            return None
        except Exception as e:  # noqa: BLE001
            logger.error("Unexpected error running macmon: %s", e)
            return None

    async def get_power_status(self) -> PowerStatus:
        """Get power source status from pmset -g batt.

        Returns PowerStatus with source, battery percentage, time remaining,
        and charging state. Returns defaults if pmset is not available.
        """
        try:
            proc = await asyncio.create_subprocess_exec(
                "pmset",
                "-g",
                "batt",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=5.0)

            if proc.returncode != 0:
                logger.warning(
                    "pmset exited with code %d: %s",
                    proc.returncode,
                    stderr.decode().strip(),
                )
                return PowerStatus()

            output = stdout.decode().strip()
            return self._parse_power_status(output)

        except FileNotFoundError:
            logger.debug("pmset not found — not a macOS system")
            return PowerStatus()
        except TimeoutError:
            logger.error("pmset subprocess timed out")
            return PowerStatus()
        except Exception as e:  # noqa: BLE001
            logger.error("Unexpected error running pmset -g batt: %s", e)
            return PowerStatus()

    @staticmethod
    def _parse_power_status(output: str) -> PowerStatus:
        """Parse pmset -g batt output into a PowerStatus model."""
        status = PowerStatus()

        for line in output.splitlines():
            line = line.strip()

            # Detect power source
            if "Now drawing from" in line:
                if "Battery Power" in line:
                    status.source = "Battery Power"
                elif "AC Power" in line:
                    status.source = "AC Power"

            # Parse battery percentage and time remaining
            if "InternalBattery" in line or line.startswith("-InternalBattery"):
                # Line format: -InternalBattery-0 (id=12345678) 98%; discharging; 3:24 remaining
                # or: -InternalBattery-0 (id=12345678) 100%; charged; 0:00 remaining
                # or: -InternalBattery-0 (id=12345678) 95%; charging; 2:15 remaining
                # or: -InternalBattery-0 (id=12345678) 95%; AC attached; not charging
                # or: -InternalBattery-0 (id=12345678) 100%; charged; (no estimate)

                # Extract percentage
                import re

                pct_match = re.search(r"(\d+)%", line)
                if pct_match:
                    status.battery_pct = float(pct_match.group(1))

                # Detect charging state
                # "charging" but not "discharging", "not charging", or "charged"
                if (
                    "charging" in line
                    and "discharging" not in line
                    and "not charging" not in line
                    and "charged" not in line
                ):
                    status.charging = True

                # Extract time remaining
                time_match = re.search(r"(\d+:\d+) remaining", line)
                if time_match:
                    status.time_remaining = time_match.group(1)

        return status

    async def get_thermal_pressure(self) -> ThermalPressure:
        """Get thermal pressure status from pmset -g therm.

        Returns ThermalPressure with warning flags.
        Returns defaults if pmset is not available.
        """
        try:
            proc = await asyncio.create_subprocess_exec(
                "pmset",
                "-g",
                "therm",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=5.0)

            if proc.returncode != 0:
                logger.warning(
                    "pmset -g therm exited with code %d: %s",
                    proc.returncode,
                    stderr.decode().strip(),
                )
                return ThermalPressure()

            output = stdout.decode().strip()
            return self._parse_thermal_pressure(output)

        except FileNotFoundError:
            logger.debug("pmset not found — not a macOS system")
            return ThermalPressure()
        except TimeoutError:
            logger.error("pmset -g therm subprocess timed out")
            return ThermalPressure()
        except Exception as e:  # noqa: BLE001
            logger.error("Unexpected error running pmset -g therm: %s", e)
            return ThermalPressure()

    @staticmethod
    def _parse_thermal_pressure(output: str) -> ThermalPressure:
        """Parse pmset -g therm output into a ThermalPressure model."""
        pressure = ThermalPressure()

        # If no thermal warning has been recorded, all fields stay False
        if "No thermal warning level has been recorded" in output:
            return pressure

        import re

        # Check for thermal warning levels
        thermal_match = re.search(r"Thermal Warning Level:\s*(\d+)", output)
        if thermal_match and int(thermal_match.group(1)) > 0:
            pressure.thermal_warning = True

        # Check for performance warning levels
        perf_match = re.search(r"Performance Warning Level:\s*(\d+)", output)
        if perf_match and int(perf_match.group(1)) > 0:
            pressure.performance_warning = True

        # Check for CPU power status
        cpu_match = re.search(r"CPU Power Status:\s*(\d+)", output)
        if cpu_match and int(cpu_match.group(1)) > 0:
            pressure.cpu_power_warning = True

        return pressure

    async def collect_stream(
        self,
        interval: float = 1.0,
        max_samples: int | None = None,
    ) -> AsyncGenerator[MacmonSample, None]:
        """Continuously collect macmon samples at the given interval.

        Yields MacmonSample objects. Stops after max_samples if set,
        otherwise runs indefinitely.
        """
        count = 0
        while max_samples is None or count < max_samples:
            sample = await self.collect()
            if sample is not None:
                yield sample
                count += 1
            await asyncio.sleep(interval)
