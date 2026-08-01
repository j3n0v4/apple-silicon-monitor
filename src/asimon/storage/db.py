"""SQLite storage for Apple Silicon Monitor.

Uses aiosqlite for async access with WAL mode enabled.
Schema matches PLAN.md Section 9 exactly.
"""

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiosqlite

logger = logging.getLogger(__name__)

# SQLite schema — exact from PLAN.md Section 9
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS hardware_samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    cpu_power REAL,
    gpu_power REAL,
    ane_power REAL,
    sys_power REAL,
    cpu_temp_avg REAL,
    gpu_temp_avg REAL,
    gpu_freq_mhz INTEGER,
    pcpu_freq_mhz INTEGER,
    ecpu_freq_mhz INTEGER,
    ram_usage_bytes INTEGER,
    ram_total_bytes INTEGER,
    swap_usage_bytes INTEGER,
    swap_total_bytes INTEGER,
    fan0_rpm INTEGER,
    fan1_rpm INTEGER
);

CREATE TABLE IF NOT EXISTS model_loads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    model_name TEXT NOT NULL,
    model_size_bytes INTEGER,
    size_vram_bytes INTEGER,
    action TEXT NOT NULL,
    context_length INTEGER,
    expires_at TEXT
);

CREATE TABLE IF NOT EXISTS inference_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    model_name TEXT NOT NULL,
    prompt_eval_count INTEGER,
    eval_count INTEGER,
    total_duration_ns INTEGER,
    eval_duration_ns INTEGER,
    prompt_eval_duration_ns INTEGER,
    load_duration_ns INTEGER,
    tokens_per_second REAL,
    streaming INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    alert_name TEXT NOT NULL,
    severity TEXT NOT NULL,
    message TEXT,
    metric_value REAL,
    threshold REAL
);

CREATE INDEX IF NOT EXISTS idx_hw_ts ON hardware_samples(timestamp);
CREATE INDEX IF NOT EXISTS idx_ml_ts ON model_loads(timestamp);
CREATE INDEX IF NOT EXISTS idx_ir_ts ON inference_runs(timestamp);
CREATE INDEX IF NOT EXISTS idx_ir_model ON inference_runs(model_name);
CREATE INDEX IF NOT EXISTS idx_alerts_ts ON alerts(timestamp);
"""


class Database:
    """Async SQLite database for Apple Silicon Monitor."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self._conn: aiosqlite.Connection | None = None

    async def init_db(self) -> None:
        """Initialize the database: create tables, indexes, and enable WAL mode."""
        # Ensure parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = await aiosqlite.connect(str(self.db_path))
        self._conn.row_factory = aiosqlite.Row

        # Enable WAL mode for concurrent reads/writes
        await self._conn.execute("PRAGMA journal_mode=WAL;")
        await self._conn.executescript(SCHEMA_SQL)
        await self._run_migrations()
        await self._conn.commit()

    async def _run_migrations(self) -> None:
        """Run schema migrations for existing databases.

        Adds columns that were added after the initial schema was created.
        """
        conn = self._conn
        cursor = await conn.execute("PRAGMA table_info(hardware_samples)")
        columns = {row[1] for row in await cursor.fetchall()}

        migrations = [
            (
                "power_source",
                "ALTER TABLE hardware_samples ADD COLUMN power_source TEXT",
            ),
            ("battery_pct", "ALTER TABLE hardware_samples ADD COLUMN battery_pct REAL"),
            (
                "thermal_warning",
                "ALTER TABLE hardware_samples ADD COLUMN thermal_warning INTEGER NOT NULL DEFAULT 0",
            ),
            (
                "performance_warning",
                "ALTER TABLE hardware_samples ADD COLUMN performance_warning INTEGER NOT NULL DEFAULT 0",
            ),
        ]

        for col_name, alter_sql in migrations:
            if col_name not in columns:
                logger.info("Running migration: adding column %s", col_name)
                await conn.execute(alter_sql)

    async def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            await self._conn.close()
            self._conn = None

    async def _ensure_connection(self) -> aiosqlite.Connection:
        """Return the connection, raising if not initialized."""
        if self._conn is None:
            raise RuntimeError("Database not initialized. Call init_db() first.")
        return self._conn

    async def insert_hardware_sample(self, data: dict[str, Any]) -> int:
        """Insert a hardware sample record. Returns the row id."""
        conn = await self._ensure_connection()
        cursor = await conn.execute(
            """
            INSERT INTO hardware_samples (
                timestamp, cpu_power, gpu_power, ane_power, sys_power,
                cpu_temp_avg, gpu_temp_avg, gpu_freq_mhz,
                pcpu_freq_mhz, ecpu_freq_mhz,
                ram_usage_bytes, ram_total_bytes,
                swap_usage_bytes, swap_total_bytes,
                fan0_rpm, fan1_rpm,
                power_source, battery_pct,
                thermal_warning, performance_warning
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                data.get("timestamp", datetime.now(UTC).isoformat()),
                data.get("cpu_power"),
                data.get("gpu_power"),
                data.get("ane_power"),
                data.get("sys_power"),
                data.get("cpu_temp_avg"),
                data.get("gpu_temp_avg"),
                data.get("gpu_freq_mhz"),
                data.get("pcpu_freq_mhz"),
                data.get("ecpu_freq_mhz"),
                data.get("ram_usage_bytes"),
                data.get("ram_total_bytes"),
                data.get("swap_usage_bytes"),
                data.get("swap_total_bytes"),
                data.get("fan0_rpm"),
                data.get("fan1_rpm"),
                data.get("power_source"),
                data.get("battery_pct"),
                data.get("thermal_warning", 0),
                data.get("performance_warning", 0),
            ),
        )
        await conn.commit()
        return cursor.lastrowid

    async def insert_model_load(self, data: dict[str, Any]) -> int:
        """Insert a model load/unload event. Returns the row id."""
        conn = await self._ensure_connection()
        cursor = await conn.execute(
            """
            INSERT INTO model_loads (
                timestamp, model_name, model_size_bytes, size_vram_bytes,
                action, context_length, expires_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                data.get("timestamp", datetime.now(UTC).isoformat()),
                data.get("model_name", ""),
                data.get("model_size_bytes"),
                data.get("size_vram_bytes"),
                data.get("action", ""),
                data.get("context_length"),
                data.get("expires_at"),
            ),
        )
        await conn.commit()
        return cursor.lastrowid

    async def insert_inference_run(self, data: dict[str, Any]) -> int:
        """Insert an inference run record. Returns the row id."""
        conn = await self._ensure_connection()
        cursor = await conn.execute(
            """
            INSERT INTO inference_runs (
                timestamp, model_name, prompt_eval_count, eval_count,
                total_duration_ns, eval_duration_ns, prompt_eval_duration_ns,
                load_duration_ns, tokens_per_second, streaming
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                data.get("timestamp", datetime.now(UTC).isoformat()),
                data.get("model_name", ""),
                data.get("prompt_eval_count"),
                data.get("eval_count"),
                data.get("total_duration_ns"),
                data.get("eval_duration_ns"),
                data.get("prompt_eval_duration_ns"),
                data.get("load_duration_ns"),
                data.get("tokens_per_second"),
                data.get("streaming", 0),
            ),
        )
        await conn.commit()
        return cursor.lastrowid

    async def insert_alert(self, data: dict[str, Any]) -> int:
        """Insert an alert event. Returns the row id."""
        conn = await self._ensure_connection()
        cursor = await conn.execute(
            """
            INSERT INTO alerts (
                timestamp, alert_name, severity, message,
                metric_value, threshold
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                data.get("timestamp", datetime.now(UTC).isoformat()),
                data.get("alert_name", ""),
                data.get("severity", ""),
                data.get("message"),
                data.get("metric_value"),
                data.get("threshold"),
            ),
        )
        await conn.commit()
        return cursor.lastrowid

    async def get_latest_inference_metrics(self) -> list[dict[str, Any]]:
        """Get the latest tokens_per_second and duration per model from the last 10 minutes.

        Returns a list of dicts with keys: model_name, tokens_per_second, eval_duration_seconds.
        Returns an empty list if the DB is empty or the table doesn't exist.
        """
        try:
            conn = await self._ensure_connection()
            cursor = await conn.execute(
                """
                SELECT
                    model_name,
                    MAX(tokens_per_second) AS tokens_per_second,
                    MAX(eval_duration_ns) / 1e9 AS eval_duration_seconds
                FROM inference_runs
                WHERE timestamp > datetime('now', '-10 minutes')
                GROUP BY model_name
                """
            )
            rows = await cursor.fetchall()
            return [
                {
                    "model_name": row["model_name"],
                    "tokens_per_second": row["tokens_per_second"],
                    "eval_duration_seconds": row["eval_duration_seconds"],
                }
                for row in rows
                if row["tokens_per_second"] is not None
            ]
        except Exception:
            logger.debug(
                "Could not read inference metrics from DB (may be empty)", exc_info=True
            )
            return []

    async def cleanup_old_records(self, days: int) -> dict[str, int]:
        """Delete records older than N days from all tables.

        Returns a dict of table_name -> deleted_row_count.
        """
        conn = await self._ensure_connection()
        from datetime import timedelta

        cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()
        results = {}

        for table in ["hardware_samples", "model_loads", "inference_runs", "alerts"]:
            cursor = await conn.execute(
                f"DELETE FROM {table} WHERE timestamp < ?",
                (cutoff,),
            )
            results[table] = cursor.rowcount

        await conn.commit()
        return results
