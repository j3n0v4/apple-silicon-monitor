# AGENTS.md

Guidance for AI agents (and humans) working in this repository.

## Project

`apple-silicon-monitor` (`asimon`) — real-time hardware + inference monitoring for
Apple Silicon LLMs. Collects GPU/CPU temps, power draw, swap, and Ollama inference
metrics and exposes them as Prometheus metrics on port 9100, with a transparent
Ollama proxy on port 11435 and a Grafana dashboard.

## Repository layout

- `src/asimon/` — package source (src layout)
  - `cli.py` — Click CLI entry point (`asimon`)
  - `collectors/` — `macmon` (hardware) and `ollama` (inference) collectors
  - `proxy/` — transparent Ollama proxy + request metrics
  - `exporters/` — Prometheus exporter
  - `storage/` — SQLite persistence
  - `alerts/` — alert/check logic
  - `benchmark/` — benchmark suite and prompts
- `tests/` — pytest suite
- `docs/` — architecture diagram, dashboard docs
- `grafana/` — Grafana provisioning/dashboards
- `data/` — sample benchmark data (local CSV/TXT, gitignored)

## Commands

- `make lint` — run ruff
- `make test` — run pytest (asyncio_mode = auto)
- `asimon serve` — headless collector + Prometheus exporter (port 9100)
- `asimon collect` — collect hardware metrics from macmon (stdout)
- `asimon proxy` — Ollama transparent proxy (port 11435)
- `asimon benchmark` — run benchmark suite
- `asimon clean` — purge system memory

## Conventions

- Python 3.11+, src layout (`where = ["src"]`).
- Configuration via `ASIMON_`-prefixed environment variables (see `config.py`).
- Keep PRs focused; run `make lint` and `make test` before opening one.

## License

MIT. See `LICENSE` (Copyright (c) 2026 JD Cordero).
