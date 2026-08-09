<h1 align="center">apple-silicon-monitor</h1>

<p align="center">Real-time hardware + inference monitoring for Apple Silicon LLMs.</p>

<p align="center">
  <a href="#getting-started">Getting Started</a> ·
  <a href="#architecture">Architecture</a> ·
  <a href="#cli-commands">CLI</a> ·
  <a href="#what-it-monitors">Metrics</a> ·
  <a href="#grafana-dashboard">Dashboard</a> ·
  <a href="#contributing">Contributing</a> ·
  <a href="#license">License</a>
</p>

<p align="center">
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-blue?style=flat"></a>
  <a href="pyproject.toml"><img alt="Python 3.11+" src="https://img.shields.io/badge/python-3.11+-blue?style=flat"></a>
  <a href="https://www.apple.com/macos/"><img alt="Platform: macOS" src="https://img.shields.io/badge/platform-macOS-333333?style=flat"></a>
  <a href="https://prometheus.io/"><img alt="Metrics: Prometheus" src="https://img.shields.io/badge/metrics-Prometheus-E6522C?style=flat"></a>
  <a href="https://grafana.com/"><img alt="Dashboard: Grafana" src="https://img.shields.io/badge/dashboard-Grafana-F46800?style=flat"></a>
  <a href="https://www.sqlite.org/"><img alt="Storage: SQLite" src="https://img.shields.io/badge/storage-SQLite-003B57?style=flat"></a>
</p>

**Why this exists:**

> I wanted to see what my M1 Max was actually doing during LLM inference. `macmon` gives a terminal snapshot, but I wanted time-series data. Prometheus + Grafana give me that, and `asimon` is the glue.

**What it does:**

Collects GPU/CPU temps, power draw, swap, and Ollama inference metrics from Apple Silicon — then pipes them into a Grafana dashboard with 26 panels and alerting.

## Getting started

```bash
pip install apple-silicon-monitor

# Start the metrics collector
asimon serve

# In another terminal, start the monitoring stack
brew install prometheus grafana
brew services start prometheus
brew services start grafana

# Open the dashboard
open http://localhost:3000
# Login: admin / admin
# Dashboard: "Apple Silicon Monitor" (auto-provisioned)
```

The dashboard shows GPU temp, power draw, tok/s by model, swap pressure, and more — all updating in real time.

## Architecture

![Architecture diagram](docs/architecture.svg)

The collector (`asimon serve`) polls `macmon` for hardware metrics and the Ollama API for inference stats, stores them in SQLite, and exposes them as Prometheus metrics on port 9100. The transparent proxy (`asimon proxy`) sits between Ollama and your client on port 11435, capturing tok/s and model load data without changing anything.

## CLI commands

| Command | Description |
|---------|-------------|
| `asimon serve` | Start headless collector + Prometheus exporter on port 9100 |
| `asimon collect` | Collect hardware metrics from macmon (stdout) |
| `asimon proxy` | Run the Ollama transparent proxy on port 11435 |
| `asimon clean` | Purge system memory and show before/after comparison |
| `asimon benchmark` | Run a benchmark suite against loaded models |

## What it monitors

| Category | Metrics |
|----------|---------|
| **Temperature** | GPU temp, CPU temp, thermal throttling, thermal/performance warnings |
| **Power** | GPU/CPU/ANE/System power draw, power source (AC/Battery), battery % |
| **Memory** | RAM usage %, swap usage (GB), RAM+swap time series |
| **Inference** | Tokens/s by model, loaded models count, model size, model VRAM |
| **System** | Uptime, GPU frequency, fan speed (RPM) |

## Grafana dashboard

The dashboard is auto-provisioned when Grafana starts. 26 panels across 5 sections:

![Dashboard preview](docs/dashboard-demo.png)

Dashboard UID: `asimon-main` (for programmatic reference).

## Benchmark data

This tool collects the data. For benchmark results and analysis, see the [apple-silicon-llm-guide](https://github.com/j3n0v4/apple-silicon-llm-guide) project.

Sample data from an M1 Max (64 GB) system is in the [data/](data/) directory, including:

- Clean baseline benchmarks (4 models × 3 prompts)
- Swap impact analysis (dirty vs. clean memory)

## Test hardware

| Component | Spec |
|-----------|------|
| Machine | MacBook Pro (16-inch, 2021) |
| SoC | Apple M1 Max |
| GPU cores | 32 |
| RAM | 64 GB unified |
| OS | macOS Sequoia 26.6 |
| Stack | asimon v0.1.0 + Prometheus + Grafana (Homebrew) |

The included data covers M1 Max 64 GB only. On a different config, run `asimon benchmark` and open a PR with your results.

## Configuration

All settings are controlled via environment variables with the `ASIMON_` prefix. Copy `.env.example` to `.env` and adjust:

| Variable | Default | Description |
|----------|---------|-------------|
| `ASIMON_POLLING_INTERVAL` | `1` | Seconds between samples |
| `ASIMON_OLLAMA_URL` | `http://localhost:11434` | Ollama API base URL |
| `ASIMON_PROXY_PORT` | `11435` | Ollama proxy port |
| `ASIMON_METRICS_PORT` | `9100` | Prometheus metrics port |
| `ASIMON_RETENTION_DAYS` | `7` | Data retention period |
| `ASIMON_DB_PATH` | `~/.asimon/data.db` | SQLite database path |

## Contributing

Contributions are welcome! This is a community project.

- **Benchmark data** — If you have different hardware (M2, M3, M4, different RAM configs), run `asimon benchmark` and open a PR with your results.
- **Bug reports** — Open an issue. Include your macOS version, hardware, and the output of `asimon collect`.
- **PRs** — Keep it focused. Run `make lint` and `make test` before opening.

## License

[MIT](LICENSE) &copy; 2026 JD Cordero
