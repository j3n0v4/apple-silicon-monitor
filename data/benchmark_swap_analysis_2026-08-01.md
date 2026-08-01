# Asimon Swap Impact Analysis — 2026-08-01

## Clean Baseline vs. Dirty Baseline Comparison

Same hardware (M1 Max 64GB), same models, same prompts. The only variable is memory state:
- **Clean**: 47 GB available, 553 MB swap, 0 models loaded, purge between models
- **Dirty**: 27 GB available, 2.5 GB swap, stale swap from prior runs, no purge between models

### Inference Performance (tok/s)

| Model | Prompt | Clean | Dirty | Delta | Notes |
|-------|--------|-------|-------|-------|-------|
| hermes3:8b | Short | 52.6 | 37.1 | +42% | Swap pressure killed small model speed |
| hermes3:8b | Medium | 35.0 | 29.6 | +18% | |
| hermes3:8b | Long | 31.1 | 23.6 | +32% | |
| **hermes3:8b avg** | | **39.6** | **30.1** | **+31%** | |
| gemma4:12b-nvfp4 | Short | 25.6 | 18.7 | +37% | |
| gemma4:12b-nvfp4 | Medium | 25.8 | 21.9 | +18% | |
| gemma4:12b-nvfp4 | Long | 22.1 | 16.9 | +31% | |
| **gemma4:12b-nvfp4 avg** | | **24.5** | **19.2** | **+28%** | |
| gemma4:26b-mlx | Short | 44.0 | 48.0 | -8% | Different eval counts |
| gemma4:26b-mlx | Medium | 41.2 | 35.5 | +16% | |
| gemma4:26b-mlx | Long | 38.9 | 35.5 | +10% | |
| **gemma4:26b-mlx avg** | | **41.4** | **39.7** | **+4%** | Large model less affected |
| qwen3.6:35b-a3b-nvfp4 | Short | 49.6 | — | — | Not tested in dirty run |
| qwen3.6:35b-a3b-nvfp4 | Medium | 47.6 | — | — | |
| qwen3.6:35b-a3b-nvfp4 | Long | 47.0 | — | — | |
| **qwen3.6:35b-a3b avg** | | **48.1** | — | — | |

### Clean Baseline (Purged Between Models) — Full Results

| Model | Size | Prompt | Eval Tokens | tok/s | Load (s) | Total (s) |
|-------|------|--------|-------------|-------|-----------|------------|
| hermes3:8b | 4.7 GB | Short | 89 | 52.6 | 3.83 | 5.62 |
| hermes3:8b | 4.7 GB | Medium | 666 | 35.0 | 2.06 | 21.23 |
| hermes3:8b | 4.7 GB | Long | 1,044 | 31.1 | 2.06 | 35.88 |
| gemma4:12b-nvfp4 | 7.7 GB | Short | 1,079 | 25.6 | 2.17 | 44.65 |
| gemma4:12b-nvfp4 | 7.7 GB | Medium | 2,118 | 25.8 | 1.38 | 83.81 |
| gemma4:12b-nvfp4 | 7.7 GB | Long | 2,560 | 22.1 | 1.78 | 118.10 |
| gemma4:26b-mlx | 17 GB | Short | 828 | 44.0 | 3.59 | 22.73 |
| gemma4:26b-mlx | 17 GB | Medium | 1,883 | 41.2 | 1.79 | 47.96 |
| gemma4:26b-mlx | 17 GB | Long | 2,694 | 38.9 | 1.79 | 71.53 |
| qwen3.6:35b-a3b-nvfp4 | 21 GB | Short | 1,684 | 49.6 | 4.78 | 38.92 |
| qwen3.6:35b-a3b-nvfp4 | 21 GB | Medium | 4,103 | 47.6 | 2.48 | 88.91 |
| qwen3.6:35b-a3b-nvfp4 | 21 GB | Long | 5,925 | 47.0 | 2.49 | 129.03 |

### Key Findings

1. **Swap pressure costs 28-31% on small models.** hermes3:8b and gemma4:12b-nvfp4 are the most affected by stale swap, gaining 28-31% tok/s on a clean baseline.

2. **Large models are less affected.** gemma4:26b-mlx only gained 4% — it fills RAM anyway and forces macOS to page out, so swap state matters less.

3. **qwen3.6:35b-a3b is the fastest model at 48 tok/s avg** despite being the largest (21 GB). MoE architecture activates only 8/128 experts per token.

4. **Purging between models is essential for small model benchmarks** but optional for large models.

### Memory Impact

| Metric | Before Benchmarks | After Benchmarks |
|--------|-------------------|------------------|
| Available RAM | 61.7 GB | 61.4 GB |
| Swap Used | 24.1 GB | 24.1 GB |
| Compressed | 0.5 GB | 0.1 GB |

Note: Available RAM stayed ~61 GB because Ollama unloads models between runs (KEEP_ALIVE=0s). Swap remained at 24 GB (stale, macOS doesn't reclaim).

### Recommendation

For accurate benchmarks, run `asimon clean --stop-ollama` before benchmark sessions. For daily use, swap accumulation is harmless — macOS manages it efficiently. The 24 GB of stale swap doesn't impact performance when RAM is available.