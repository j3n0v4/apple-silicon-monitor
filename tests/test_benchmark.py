"""Tests for the benchmark module."""

from asimon.benchmark.prompts import ALL_LENGTHS, PROMPTS


class TestPrompts:
    """Test the benchmark prompts module."""

    def test_all_lengths(self):
        """Test that ALL_LENGTHS contains the expected prompt lengths."""
        assert ALL_LENGTHS == ["short", "medium", "long"]

    def test_short_prompt(self):
        """Test that the short prompt is defined and non-empty."""
        assert "short" in PROMPTS
        assert len(PROMPTS["short"]) > 10

    def test_medium_prompt(self):
        """Test that the medium prompt is defined and non-empty."""
        assert "medium" in PROMPTS
        assert len(PROMPTS["medium"]) > 50

    def test_long_prompt(self):
        """Test that the long prompt is defined and non-empty."""
        assert "long" in PROMPTS
        assert len(PROMPTS["long"]) > 200

    def test_prompt_uniqueness(self):
        """Test that all prompts are distinct."""
        assert len(set(PROMPTS.values())) == 3
