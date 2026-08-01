.PHONY: install test lint build

VENV := .venv/bin
PYTHON := $(VENV)/python
PIP := $(VENV)/pip
PYTEST := $(VENV)/pytest
RUFF := $(VENV)/ruff

install:
	$(PIP) install -e ".[dev]"

test:
	PYTHONPATH="src:." $(PYTEST) -v --cov=src/asimon

lint:
	$(RUFF) check src/ tests/
	$(RUFF) format --check src/ tests/

build:
	$(PIP) install build
	$(PYTHON) -m build