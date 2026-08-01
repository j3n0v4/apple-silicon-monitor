"""Tests for the benchmark runner module."""

from asimon.benchmark.runner import (
    DEFAULT_MODELS,
    BenchmarkResult,
    MemorySnapshot,
    _format_bytes,
    _format_duration_ns,
    _output_csv,
    _output_json,
    _output_table,
)


class TestFormatHelpers:
    """Test formatting helper functions."""

    def test_format_bytes(self):
        """Test byte formatting."""
        assert "1.0 B" in _format_bytes(1)
        assert "1.0 KB" in _format_bytes(1024)
        assert "1.0 MB" in _format_bytes(1024**2)
        assert "1.0 GB" in _format_bytes(1024**3)
        assert "1.0 TB" in _format_bytes(1024**4)

    def test_format_duration_ns(self):
        """Test duration formatting."""
        assert "0ms" in _format_duration_ns(0)
        assert "500ms" in _format_duration_ns(500_000_000)
        assert "1.00s" in _format_duration_ns(1_000_000_000)
        assert "1.0m" in _format_duration_ns(60_000_000_000)
        assert "N/A" == _format_duration_ns(None)


class TestDefaultModels:
    """Test default model list."""

    def test_default_models(self):
        """Test that DEFAULT_MODELS contains the expected models."""
        assert len(DEFAULT_MODELS) == 4
        assert "hermes3:8b" in DEFAULT_MODELS
        assert "gemma4:12b-nvfp4" in DEFAULT_MODELS
        assert "gemma4:26b-mlx" in DEFAULT_MODELS
        assert "qwen3.6:35b-a3b-nvfp4" in DEFAULT_MODELS


class TestMemorySnapshot:
    """Test MemorySnapshot dataclass."""

    def test_snapshot_creation(self):
        """Test creating a MemorySnapshot."""
        snap = MemorySnapshot(
            timestamp="2026-08-01T12:00:00",
            vm_stat={"Pages free": 1000},
            swap_usage="total = 0.00M  used = 0.00M  free = 0.00M",
            free_memory_pct=50.0,
        )
        assert snap.timestamp == "2026-08-01T12:00:00"
        assert snap.vm_stat["Pages free"] == 1000
        assert snap.free_memory_pct == 50.0


class TestBenchmarkResult:
    """Test BenchmarkResult dataclass."""

    def test_result_creation(self):
        """Test creating a BenchmarkResult."""
        result = BenchmarkResult(
            model="hermes3:8b",
            prompt_length="short",
            eval_count=100,
            eval_duration_ns=1_000_000_000,
            tokens_per_second=100.0,
        )
        assert result.model == "hermes3:8b"
        assert result.prompt_length == "short"
        assert result.eval_count == 100
        assert result.tokens_per_second == 100.0
        assert result.error is None

    def test_result_with_error(self):
        """Test creating a BenchmarkResult with an error."""
        result = BenchmarkResult(
            model="hermes3:8b",
            prompt_length="short",
            error="Connection refused",
        )
        assert result.error == "Connection refused"
        assert result.eval_count is None


class TestOutputFormats:
    """Test output formatting functions."""

    def test_output_json(self):
        """Test JSON output format."""
        results = [
            BenchmarkResult(
                model="test-model",
                prompt_length="short",
                eval_count=100,
                eval_duration_ns=1_000_000_000,
                tokens_per_second=100.0,
            )
        ]
        baseline = MemorySnapshot(
            timestamp="2026-01-01T00:00:00",
            vm_stat={},
            swap_usage="total = 0.00M  used = 0.00M  free = 0.00M",
            free_memory_pct=50.0,
        )
        end = MemorySnapshot(
            timestamp="2026-01-01T00:01:00",
            vm_stat={},
            swap_usage="total = 0.00M  used = 0.00M  free = 0.00M",
            free_memory_pct=45.0,
        )
        output = _output_json(results, baseline, end)
        assert '"model": "test-model"' in output
        assert '"tokens_per_second": 100.0' in output
        assert '"free_memory_pct": 50.0' in output

    def test_output_table(self):
        """Test table output format."""
        results = [
            BenchmarkResult(
                model="test-model",
                prompt_length="short",
                eval_count=100,
                eval_duration_ns=1_000_000_000,
                tokens_per_second=100.0,
            )
        ]
        baseline = MemorySnapshot(
            timestamp="2026-01-01T00:00:00",
            vm_stat={},
            swap_usage="total = 0.00M  used = 0.00M  free = 0.00M",
            free_memory_pct=50.0,
        )
        end = MemorySnapshot(
            timestamp="2026-01-01T00:01:00",
            vm_stat={},
            swap_usage="total = 0.00M  used = 0.00M  free = 0.00M",
            free_memory_pct=45.0,
        )
        output = _output_table(results, baseline, end)
        assert "test-model" in output
        assert "100.0" in output
        assert "Benchmark Results" in output

    def test_output_csv(self):
        """Test CSV output format."""
        results = [
            BenchmarkResult(
                model="test-model",
                prompt_length="short",
                eval_count=100,
                eval_duration_ns=1_000_000_000,
                tokens_per_second=100.0,
            )
        ]
        baseline = MemorySnapshot(
            timestamp="2026-01-01T00:00:00",
            vm_stat={},
            swap_usage="total = 0.00M  used = 0.00M  free = 0.00M",
            free_memory_pct=50.0,
        )
        end = MemorySnapshot(
            timestamp="2026-01-01T00:01:00",
            vm_stat={},
            swap_usage="total = 0.00M  used = 0.00M  free = 0.00M",
            free_memory_pct=45.0,
        )
        output = _output_csv(results, baseline, end)
        assert "test-model" in output
        assert "100.0" in output
        assert "model,prompt_length" in output
