"""Shared pytest fixtures for Apple Silicon Monitor tests."""

import pytest

from asimon.storage.db import Database


@pytest.fixture
async def db():
    """Create an in-memory database for testing."""
    database = Database(db_path=":memory:")
    await database.init_db()
    yield database
    await database.close()
