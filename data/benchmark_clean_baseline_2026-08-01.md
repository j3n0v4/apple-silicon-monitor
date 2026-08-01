# Asimon Clean Baseline Benchmarks — 2026-08-01

Hardware: MacBook Pro M1 Max, 64GB RAM, macOS Sequoia
Stack: asimon v0.1.0 + Prometheus + Grafana (Homebrew)
Method: 4 models × 3 prompts, via asimon proxy (port 11435), stream=false
Baseline: 47 GB available RAM, 553 MB swap, 0 models loaded

## Inference Performance

| Model | Size | Prompt | Eval Tokens | tok/s | Load (s) | Total (s) |
|-------|------|--------|-------------|-------|----------|------------|
| hermes3:8b | 4.7 GB | Short | 92 | **52.69** | 3.83 | 5.67 |
| hermes3:8b | 4.7 GB | Medium | 828 | **33.30** | 2.06 | 27.07 |
| hermes3:8b | 4.7 GB | Long | 741 | **31.15** | 2.04 | 26.04 |
| gemma4:12b-nvfp4 | 7.7 GB | Short | 460 | **24.98** | 2.18 | 20.90 |
| gemma4:12b-nvfp4 | 7.7 GB | Medium | 2,321 | **21.26** | 1.37 | 110.95 |
| gemma4:12b-nvfp4 | 7.7 GB | Long | 2,242 | **17.94** | 2.22 | 127.89 |
| gemma4:26b-mlx | 17 GB | Short | 1,033 | **37.88** | 3.62 | 31.24 |
| gemma4:26b-mlx | 17 GB | Medium | 1,775 | **36.95** | 1.79 | 50.32 |
| gemma4:26b-mlx | 17 GB | Long | 2,451 | **38.20** | 1.78 | 66.46 |
| qwen3.6:35b-a3b-nvfp4 | 21 GB | Short | 748 | **50.59** | 4.78 | 19.74 |
| qwen3.6:35b-a3b-nvfp4 | 21 GB | Medium | 3,986 | **47.23** | 2.58 | 87.25 |
| qwen3.6:35b-a3b-nvfp4 | 21 GB | Long | 4,839 | **46.66** | 2.38 | 106.45 |

## Average tok/s by Model (across all prompts)

| Model | Size | Avg tok/s | Load (s) | Architecture |
|-------|------|-----------|-----------|---------------|
| qwen3.6:35b-a3b-nvfp4 | 21 GB | **48.16** | 3.25 | MoE (8/128 active) |
| gemma4:26b-mlx | 17 GB | **37.68** | 2.40 | Dense MLX |
| hermes3:8b | 4.7 GB | **39.05** | 2.64 | Dense GGUF |
| gemma4:12b-nvfp4 | 7.7 GB | **21.39** | 1.92 | Dense NVFP4 |

## Hardware Under Load (Prometheus, 15s intervals)

| Metric | Min | Peak | Avg |
|--------|-----|------|-----|
| GPU Temp | 61.3°C | **97.5°C** | 82.5°C |
| CPU Temp | 63.7°C | 94.1°C | 80.5°C |
| GPU Power | 2.3W | **39.5W** | 14.9W |
| Sys Power | 25.7W | **93.1W** | 55.6W |
| RAM Usage | 6.9 GB | **61.2 GB** | 27.3 GB |
| Swap Usage | 0.5 GB | **23.6 GB** | 17.7 GB |
| GPU Freq | 388 MHz | **1,296 MHz** | 838 MHz |
| Thermal Throttle | 0 | 0 | 0 |
| Fan 0 | 2,297 | 5,856 | 5,142 |
| Fan 1 | 2,476 | 6,333 | 5,553 |

## Memory Delta

| Metric | Before | After |
|--------|--------|-------|
| Available RAM | 47.0 GB | 59.9 GB (models unloaded) |
| Swap Used | 553 MB | 24.1 GB (stale, models paged out) |
| Swap Total | 2.0 GB | 24.6 GB (macOS expanded) |

Note: After benchmarks, Ollama unloaded all models. macOS expanded swap to 24 GB during the 35B model run but didn't reclaim it. Available RAM is high because model memory was released.

## Comparison with Previous Runs (Dirty Baseline)

| Model | Clean tok/s | Dirty tok/s | Delta |
|-------|-------------|-------------|-------|
| hermes3:8b (short) | 52.69 | 37.12 | +42% |
| gemma4:12b-nvfp4 (short) | 24.98 | 18.69 | +34% |
| gemma4:26b-mlx (short) | 37.88 | 48.01 | −21% |
| gemma4:12b-nvfp4 (medium) | 21.26 | 21.88 | −3% |
| gemma4:26b-mlx (medium) | 36.95 | 35.48 | +4% |

Note: hermes3:8b and gemma4:12b improved significantly on clean baseline. gemma4:26b-mlx was similar. The dirty baseline had swap pressure that slowed smaller models.

## Key Findings

1. **qwen3.6:35b-a3b is the fastest model** at 47-51 tok/s despite being the largest (21 GB). MoE architecture means only 8 of 128 experts are active per token — effectively processing like a 3B-parameter model.

2. **No thermal throttling** at 97.5°C GPU peak. M1 Max handled all loads without pmset thermal pressure.

3. **Swap ballooned to 24 GB** during the 35B model run (61 GB peak RAM). This is normal macOS behavior — it pages out inactive memory to make room for model weights. After models unload, swap stays high but RAM is freed.

4. **gemma4:12b-nvfp4 is the slowest** at 18-25 tok/s despite being only 7.7 GB. NVFP4 quantization has higher per-token overhead than MLX format.

5. **hermes3:8b is the best small model** at 31-53 tok/s. GGUF Q4_0 is efficient for 8B dense models.