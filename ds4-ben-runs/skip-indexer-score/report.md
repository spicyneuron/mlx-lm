# DeepSeek V4 MLX ref benchmark

- base: `mlx-lm@git+https://github.com/spicyneuron/mlx-lm@_ds4`
- ours: `mlx-lm@git+https://github.com/spicyneuron/mlx-lm@_ds4_perf`
- model: `284-deepseek-4`

| case | prompt toks | gen toks | base gen tok/s | ours gen tok/s | delta | base prompt tok/s | ours prompt tok/s | delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| decode_128 | 128 | 128 | 32.068 | 31.844 | -0.70% | 293.125 | 290.968 | -0.74% |
| csa_2k_decode | 2304 | 128 | 29.527 | 29.056 | -1.59% | 452.971 | 453.420 | +0.10% |
| long_8k_decode | 8192 | 128 | 28.312 | 28.393 | +0.28% | 434.572 | 434.783 | +0.05% |

Raw JSON:
- `base.json`
- `ours.json`

Rule of thumb: treat <3% as noise unless it repeats across runs; prioritize >5-10% deltas.
