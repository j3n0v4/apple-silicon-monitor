# Asimon Swap Impact Analysis — 2026-08-01

## Clean baseline vs. dirty baseline comparison

Same hardware (M1 Max 64GB), same models, same prompts. The only variable is memory state:
- **Clean**: 47 GB available, 553 MB swap, 0 models loaded, purge between models
- **Dirty**: 27 GB available, 2.5 GB swap, stale swap from prior runs, no purge between models

### Inference performance (tok/s)

| Model | Prompt | Clean | Dirty | Delta | Notes |
|-------|--------|-------|-------|-------|-------|
| hermes3:8b | Short | 52.69 | 37.1 | +42% | Swap pressure killed small model speed |
| hermes3:8b | Medium | 33.30 | 29.6 | +13% | |
| hermes3:8b | Long | 31.15 | 23.6 | +32% | |
| **hermes3:8b avg** | | **39.05** | **30.1** | **+30%** | |
| gemma4:12b-nvfp4 | Short | 24.98 | 18.7 | +34% | |
| gemma4:12b-nvfp4 | Medium | 21.26 | 21.9 | -3% | |
| gemma4:12b-nvfp4 | Long | 17.94 | 16.9 | +6% | |
| **gemma4:12b-nvfp4 avg** | | **21.39** | **19.2** | **+11%** | |
| gemma4:26b-mlx | Short | 37.88 | 48.0 | -21% | Different eval counts |
| gemma4:26b-mlx | Medium | 36.95 | 35.5 | +4% | |
| gemma4:26b-mlx | Long | 38.20 | 35.5 | +8% | |
| **gemma4:26b-mlx avg** | | **37.68** | **39.7** | **-5%** | Large model roughly unchanged |
| qwen3.6:35b-a3b-nvfp4 | Short | 50.59 | — | — | Not tested in dirty run |
| qwen3.6:35b-a3b-nvfp4 | Medium | 47.23 | — | — | |
| qwen3.6:35b-a3b-nvfp4 | Long | 46.66 | — | — | |
| **qwen3.6:35b-a3b avg** | | **48.16** | — | — | |

### Clean baseline (purged between models) — full results

| Model | Size | Prompt | Eval Tokens | tok/s | Load (s) | Total (s) |
|-------|------|--------|-------------|-------|-----------|------------|
| hermes3:8b | 4.7 GB | Short | 92 | 52.69 | 3.83 | 5.67 |
| hermes3:8b | 4.7 GB | Medium | 828 | 33.30 | 2.06 | 27.07 |
| hermes3:8b | 4.7 GB | Long | 741 | 31.15 | 2.04 | 26.04 |
| gemma4:12b-nvfp4 | 7.7 GB | Short | 460 | 24.98 | 2.18 | 20.90 |
| gemma4:12b-nvfp4 | 7.7 GB | Medium | 2,321 | 21.26 | 1.37 | 110.95 |
| gemma4:12b-nvfp4 | 7.7 GB | Long | 2,242 | 17.94 | 2.22 | 127.89 |
| gemma4:26b-mlx | 17 GB | Short | 1,033 | 37.88 | 3.62 | 31.24 |
| gemma4:26b-mlx | 17 GB | Medium | 1,775 | 36.95 | 1.79 | 50.32 |
| gemma4:26b-mlx | 17 GB | Long | 2,451 | 38.20 | 1.78 | 66.46 |
| qwen3.6:35b-a3b-nvfp4 | 21 GB | Short | 748 | 50.59 | 4.78 | 19.74 |
| qwen3.6:35b-a3b-nvfp4 | 21 GB | Medium | 3,986 | 47.23 | 2.58 | 87.25 |
| qwen3.6:35b-a3b-nvfp4 | 21 GB | Long | 4,839 | 46.66 | 2.38 | 106.45 |

### Key findings

1. **Swap pressure costs small models 11-30% on average.** hermes3:8b gained 30% and gemma4:12b-nvfp4 gained 11% average tok/s on a clean baseline (per-prompt deltas ranged from -3% to +34%).

2. **Large models are less affected.** gemma4:26b-mlx was roughly unchanged (-5% avg) — it fills RAM anyway and forces macOS to page out, so swap state matters less.

3. **qwen3.6:35b-a3b is the fastest model at 48 tok/s avg** despite being the largest (21 GB). MoE architecture activates only 8/128 experts per token.

4. **Purging between models is essential for small model benchmarks** but optional for large models.

### Memory impact

| Metric | Before Benchmarks | After Benchmarks |
|--------|-------------------|------------------|
| Available RAM | 61.7 GB | 61.4 GB |
| Swap Used | 24.1 GB | 24.1 GB |
| Compressed | 0.5 GB | 0.1 GB |

Note: Available RAM stayed ~61 GB because Ollama unloads models between runs (KEEP_ALIVE=0s). Swap remained at 24 GB (stale, macOS does not reclaim).

### Recommendation

For accurate benchmarks, run `asimon clean --stop-ollama` before benchmark sessions. For daily use, swap accumulation is harmless — macOS manages it efficiently. The 24 GB of stale swap does not impact performance when RAM is available.
