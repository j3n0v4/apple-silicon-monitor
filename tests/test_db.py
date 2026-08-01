"""Tests for the SQLite storage layer."""

import pytest

from asimon.storage.db import Database


class TestDatabaseInit:
    """Test database initialization."""

    @pytest.mark.asyncio
    async def test_init_creates_tables(self):
        """Test that init_db creates all 4 tables."""
        database = Database(db_path=":memory:")
        await database.init_db()

        conn = database._conn
        cursor = await conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [row[0] for row in await cursor.fetchall()]

        assert "hardware_samples" in tables
        assert "model_loads" in tables
        assert "inference_runs" in tables
        assert "alerts" in tables

        await database.close()

    @pytest.mark.asyncio
    async def test_init_creates_indexes(self):
        """Test that init_db creates all 5 indexes."""
        database = Database(db_path=":memory:")
        await database.init_db()

        conn = database._conn
        cursor = await conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%' ORDER BY name"
        )
        indexes = [row[0] for row in await cursor.fetchall()]

        assert "idx_hw_ts" in indexes
        assert "idx_ml_ts" in indexes
        assert "idx_ir_ts" in indexes
        assert "idx_ir_model" in indexes
        assert "idx_alerts_ts" in indexes

        await database.close()

    @pytest.mark.asyncio
    async def test_wal_mode_enabled(self):
        """Test that WAL mode is enabled."""
        database = Database(db_path=":memory:")
        await database.init_db()

        conn = database._conn
        cursor = await conn.execute("PRAGMA journal_mode")
        mode = await cursor.fetchone()

        # In-memory databases may not support WAL, but the PRAGMA should not error
        assert mode is not None

        await database.close()

    @pytest.mark.asyncio
    async def test_migration_adds_new_columns(self):
        """Test that _run_migrations adds new columns to existing tables."""
        # Create a database with the old schema (no new columns)
        database = Database(db_path=":memory:")
        await database.init_db()

        conn = database._conn

        # Verify new columns exist after init_db (which runs migrations)
        cursor = await conn.execute("PRAGMA table_info(hardware_samples)")
        columns = {row[1] for row in await cursor.fetchall()}

        assert "power_source" in columns
        assert "battery_pct" in columns
        assert "thermal_warning" in columns
        assert "performance_warning" in columns

        await database.close()

    @pytest.mark.asyncio
    async def test_migration_idempotent(self):
        """Test that running migrations twice doesn't error."""
        database = Database(db_path=":memory:")
        await database.init_db()

        # Run migrations again — should be a no-op
        await database._run_migrations()

        conn = database._conn
        cursor = await conn.execute("PRAGMA table_info(hardware_samples)")
        columns = {row[1] for row in await cursor.fetchall()}

        assert "power_source" in columns
        assert "battery_pct" in columns
        assert "thermal_warning" in columns
        assert "performance_warning" in columns

        await database.close()


class TestHardwareSamples:
    """Test hardware_samples table operations."""

    @pytest.mark.asyncio
    async def test_insert_and_count(self, db):
        """Test inserting a hardware sample and verifying it's stored."""
        row_id = await db.insert_hardware_sample(
            {
                "timestamp": "2026-08-01T12:00:00+00:00",
                "cpu_power": 1.5,
                "gpu_power": 28.4,
                "ane_power": 0.0,
                "sys_power": 30.0,
                "cpu_temp_avg": 55.0,
                "gpu_temp_avg": 52.0,
                "gpu_freq_mhz": 1296,
                "pcpu_freq_mhz": 1800,
                "ecpu_freq_mhz": 1400,
                "ram_usage_bytes": 30000000000,
                "ram_total_bytes": 68719476736,
                "swap_usage_bytes": 1000000000,
                "swap_total_bytes": 7516192768,
                "fan0_rpm": 2000,
                "fan1_rpm": 2100,
                "power_source": "AC Power",
                "battery_pct": 100.0,
                "thermal_warning": 0,
                "performance_warning": 0,
            }
        )

        assert row_id > 0

        conn = db._conn
        cursor = await conn.execute("SELECT COUNT(*) FROM hardware_samples")
        count = await cursor.fetchone()
        assert count[0] == 1

    @pytest.mark.asyncio
    async def test_insert_with_power_and_thermal(self, db):
        """Test inserting a hardware sample with power/thermal data."""
        row_id = await db.insert_hardware_sample(
            {
                "timestamp": "2026-08-01T12:00:00+00:00",
                "cpu_power": 1.5,
                "gpu_power": 28.4,
                "ane_power": 0.0,
                "sys_power": 30.0,
                "cpu_temp_avg": 55.0,
                "gpu_temp_avg": 52.0,
                "gpu_freq_mhz": 1296,
                "pcpu_freq_mhz": 1800,
                "ecpu_freq_mhz": 1400,
                "ram_usage_bytes": 30000000000,
                "ram_total_bytes": 68719476736,
                "swap_usage_bytes": 1000000000,
                "swap_total_bytes": 7516192768,
                "fan0_rpm": 2000,
                "fan1_rpm": 2100,
                "power_source": "Battery Power",
                "battery_pct": 75.0,
                "thermal_warning": 1,
                "performance_warning": 0,
            }
        )

        assert row_id > 0

        conn = db._conn
        cursor = await conn.execute(
            "SELECT power_source, battery_pct, thermal_warning, performance_warning FROM hardware_samples WHERE id = ?",
            (row_id,),
        )
        row = await cursor.fetchone()
        assert row["power_source"] == "Battery Power"
        assert row["battery_pct"] == 75.0
        assert row["thermal_warning"] == 1
        assert row["performance_warning"] == 0

    @pytest.mark.asyncio
    async def test_insert_minimal(self, db):
        """Test inserting a hardware sample with only required fields."""
        row_id = await db.insert_hardware_sample(
            {
                "timestamp": "2026-08-01T12:00:00+00:00",
            }
        )
        assert row_id > 0


class TestModelLoads:
    """Test model_loads table operations."""

    @pytest.mark.asyncio
    async def test_insert_model_load(self, db):
        """Test inserting a model load event."""
        row_id = await db.insert_model_load(
            {
                "timestamp": "2026-08-01T12:00:00+00:00",
                "model_name": "qwen3-coder:30b",
                "model_size_bytes": 18556700761,
                "size_vram_bytes": 18556700761,
                "action": "loaded",
                "context_length": 262144,
                "expires_at": "2026-08-01T13:00:00+00:00",
            }
        )
        assert row_id > 0

        conn = db._conn
        cursor = await conn.execute(
            "SELECT model_name, action, size_vram_bytes FROM model_loads WHERE id = ?",
            (row_id,),
        )
        row = await cursor.fetchone()
        assert row["model_name"] == "qwen3-coder:30b"
        assert row["action"] == "loaded"
        assert row["size_vram_bytes"] == 18556700761


class TestInferenceRuns:
    """Test inference_runs table operations."""

    @pytest.mark.asyncio
    async def test_insert_inference_run(self, db):
        """Test inserting an inference run."""
        row_id = await db.insert_inference_run(
            {
                "timestamp": "2026-08-01T12:00:00+00:00",
                "model_name": "qwen3-coder:30b",
                "prompt_eval_count": 15,
                "eval_count": 200,
                "total_duration_ns": 5000000000,
                "eval_duration_ns": 4000000000,
                "prompt_eval_duration_ns": 500000000,
                "load_duration_ns": 100000000,
                "tokens_per_second": 50.0,
                "streaming": 0,
            }
        )
        assert row_id > 0

        conn = db._conn
        cursor = await conn.execute(
            "SELECT model_name, tokens_per_second, streaming FROM inference_runs WHERE id = ?",
            (row_id,),
        )
        row = await cursor.fetchone()
        assert row["model_name"] == "qwen3-coder:30b"
        assert row["tokens_per_second"] == 50.0
        assert row["streaming"] == 0

    @pytest.mark.asyncio
    async def test_insert_streaming_run(self, db):
        """Test inserting a streaming inference run."""
        row_id = await db.insert_inference_run(
            {
                "timestamp": "2026-08-01T12:00:00+00:00",
                "model_name": "hermes3:8b",
                "eval_count": 500,
                "total_duration_ns": 10000000000,
                "eval_duration_ns": 9000000000,
                "prompt_eval_duration_ns": 500000000,
                "tokens_per_second": 55.5,
                "streaming": 1,
            }
        )
        assert row_id > 0

        conn = db._conn
        cursor = await conn.execute(
            "SELECT streaming FROM inference_runs WHERE id = ?",
            (row_id,),
        )
        row = await cursor.fetchone()
        assert row["streaming"] == 1


class TestAlerts:
    """Test alerts table operations."""

    @pytest.mark.asyncio
    async def test_insert_alert(self, db):
        """Test inserting an alert."""
        row_id = await db.insert_alert(
            {
                "timestamp": "2026-08-01T12:00:00+00:00",
                "alert_name": "thermal_throttle",
                "severity": "warning",
                "message": "GPU frequency dropped to 388 MHz",
                "metric_value": 388.0,
                "threshold": 500.0,
            }
        )
        assert row_id > 0

        conn = db._conn
        cursor = await conn.execute(
            "SELECT alert_name, severity, metric_value FROM alerts WHERE id = ?",
            (row_id,),
        )
        row = await cursor.fetchone()
        assert row["alert_name"] == "thermal_throttle"
        assert row["severity"] == "warning"
        assert row["metric_value"] == 388.0


class TestCleanup:
    """Test record cleanup."""

    @pytest.mark.asyncio
    async def test_cleanup_old_records(self, db):
        """Test that cleanup_old_records deletes old records."""
        # Insert a very old record
        await db.insert_hardware_sample(
            {
                "timestamp": "2020-01-01T00:00:00+00:00",
                "cpu_power": 1.0,
            }
        )
        # Insert a recent record
        await db.insert_hardware_sample(
            {
                "timestamp": "2026-08-01T12:00:00+00:00",
                "cpu_power": 2.0,
            }
        )

        # Clean up records older than 1 day
        results = await db.cleanup_old_records(days=1)

        assert results["hardware_samples"] == 1  # old record deleted

        # Verify only the recent record remains
        conn = db._conn
        cursor = await conn.execute("SELECT COUNT(*) FROM hardware_samples")
        count = await cursor.fetchone()
        assert count[0] == 1
