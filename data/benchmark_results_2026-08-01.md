# Asimon Benchmark Results — 2026-08-01

Hardware: MacBook Pro M1 Max, 64GB RAM, macOS Sequoia
Stack: asimon v0.1.0 + Prometheus + Grafana (Homebrew)
Method: 3 prompts × 3 models, via asimon proxy (port 11435), stream=false

## Inference Performance

| Model | Prompt | Eval Tokens | tok/s | Load Time (s) | Duration (s) |
|-------|--------|-------------|-------|---------------|---------------|
| dolphin3-abliterated:8b | Short | 84 | 37.12 | 4.09 | 2.26 |
| dolphin3-abliterated:8b | Medium | 320 | 29.62 | 2.06 | 10.81 |
| dolphin3-abliterated:8b | Long | 2,089 | 23.56 | 2.05 | 88.66 |
| gemma4:12b-nvfp4 | Short | 389 | 18.69 | 2.17 | 20.81 |
| gemma4:12b-nvfp4 | Medium | 1,904 | 21.88 | 1.38 | 87.01 |
| gemma4:12b-nvfp4 | Long | 2,211 | 16.87 | 2.09 | 131.02 |
| gemma4:26b-mlx | Short | 390 | 48.01 | 3.60 | 8.12 |
| gemma4:26b-mlx | Medium | 1,759 | 35.48 | 1.79 | 49.57 |
| gemma4:26b-mlx | Long | 2,318 | 35.51 | 1.79 | 65.27 |

## Hardware During Benchmarks (Prometheus 15s intervals)

| Metric | Idle | Peak | Avg |
|--------|------|------|-----|
| GPU Temp | 51.9°C | 92.8°C | 83.0°C |
| CPU Temp | 53.3°C | 88.3°C | 79.1°C |
| GPU Power | 2.2W | 25.3W | 15.5W |
| Sys Power | 16.7W | 75.9W | 59.4W |
| RAM Usage | 27.3GB | 64.2GB | 39.2GB |
| GPU Freq | 409 MHz | 1,194 MHz | 775 MHz |
| Thermal Throttling | 0 | 0 | 0 |
| Thermal Warning | 0 | 0 | 0 |

## Key Findings

1. **gemma4:26b-mlx is fastest** — 35-48 tok/s, beating the smaller 12B nvfp4 model.
   MLX format is dramatically more efficient than nvfp4 on Apple Silicon.

2. **No thermal throttling** — GPU hit 92.8°C peak but macOS never reported thermal pressure.
   The 500MHz threshold in the old code was a false positive.

3. **GPU scales from idle 409 MHz to active 1,194 MHz** during inference.

4. **RAM peaks at 64.2GB** during large model loads (26B MLX uses ~17GB).

5. **dolphin3-abliterated:8b** is the best small uncensored model at 23-37 tok/s.

## SQLite Raw Data (inference_runs)

| id | timestamp | model | prompt_eval | eval | tok/s | load_ns | total_ns |
|----|-----------|-------|-------------|------|-------|---------|----------|
| 2 | 17:47:47 | dolphin3-abliterated:8b | 19 | 84 | 37.12 | 4.09s | 6.46s |
| 3 | 17:48:08 | dolphin3-abliterated:8b | 40 | 320 | 29.62 | 2.06s | 13.04s |
| 4 | 17:49:49 | dolphin3-abliterated:8b | 71 | 2,089 | 23.56 | 2.05s | 90.94s |
| 5 | 17:50:24 | qwen3-abliterated:14b | 19 | 303 | 13.54 | 5.03s | 27.60s |
| 7 | 18:18:09 | gemma4:12b-nvfp4 | 24 | 389 | 18.69 | 2.17s | 23.30s |
| 8 | 18:19:45 | gemma4:12b-nvfp4 | 45 | 1,904 | 21.88 | 1.38s | 88.88s |
| 9 | 18:22:07 | gemma4:12b-nvfp4 | 78 | 2,211 | 16.87 | 2.09s | 133.78s |
| 10 | 18:22:23 | gemma4:26b-mlx | 24 | 390 | 48.01 | 3.60s | 12.07s |
| 11 | 18:23:22 | gemma4:26b-mlx | 45 | 1,759 | 35.48 | 1.79s | 51.85s |
| 12 | 18:24:38 | gemma4:26b-mlx | 78 | 2,318 | 35.51 | 1.79s | 67.59s |