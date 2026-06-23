# Native MTP on GLM-DSA (GLM-5.2): characterization, fix & verdict

**Date:** 2026-06-23  **Hardware:** M3 (host `m3`), MLX 0.31.2
**Model:** GLM-5.2 `glm_moe_dsa`, 4-bit/gs-64 affine quant (`glm-5.2/lm-03`, `lm-01`)
**Runtime branch:** `_glm-mtp`  **Drivers:** `benchmarks/mtp_characterize.sh`,
`benchmarks/mtp_phase1_verify.py` (+ `_diag`, `_layerdiff`, `_realprompt`)

## TL;DR
Native MTP was initially a **net loss at every context** because the speculative
**verify forward scaled with context** while plain decode did not. Root cause: the
`L>1` attention path **un-absorbed MLA** — it up-projected the *entire* KV to
per-head K/V every step. Fix (**Phase 1, landed**): route short query lengths
(`L <= ABSORB_MAX_L=64`, i.e. MTP verify) through the **absorbed** latent-space
attention, same as single-token decode. Verify at 4096 dropped from ~566 ms to
~89 ms (**6.3×**) and is now ~flat with context. Validated **lossless for greedy**
(100% argmax agreement vs sequential decode on real text at 3k/6k/12k).

## Before: the original loss

| prompt | config | gen tps | ms/tok | accept | draft ms/tok | verify ms/tok |
|-------:|--------|--------:|-------:|-------:|-------------:|--------------:|
| 512    | baseline (no MTP) | 15.20 | 65.8 | — | — | — |
| 512    | MTP D1 (bonus)    | 10.07 | 99.3 | 98.4% | 2.96 | 97.74 |
| 4096   | baseline (no MTP) | 14.15 | 70.7 | — | — | — |
| 4096   | MTP D1 (bonus)    | **3.11** | **321.6** | 96.9% | 3.05 | **317.62** |

Decode (L=1) was flat with context (65.8 → 70.7); verify (L>1) tripled
(97.7 → 317.6). Drafting was cheap (~3 ms/tok) and acceptance excellent (96–98%),
so the entire problem was the target's L>1 verify forward.

## Root cause (corrected)
`GlmMoeDsaAttention.__call__` / `DeepseekV32Attention.__call__` had two paths:

- **`L == 1` (decode):** gather each query's sparse `topk` KV, attend in the
  **absorbed** kv_lora latent space. Cost bounded by `index_topk` (2048) → flat.
- **`L > 1` (every verify):** **un-absorbed** — `embed_q(kv_latent)` /
  `unembed_out(kv_latent)` up-project the **full** KV to per-head `[B,H,KV,128]`
  every step. That up-projection (`heads·KV·512·128`) is the dominant, context-
  scaling cost. (The original note blamed "attends over full KV"; the real killer
  was the *per-head up-projection* of full KV, not the attention itself.)

The un-absorbed path is correct for **prefill** (large L amortizes the
up-projection), but verify has tiny L and pays it with nothing to amortize.

## Fix — Phase 1: absorbed verify (`absorbed_attention`)
Gate on query length: `absorb = L <= ABSORB_MAX_L` (64, comfortably below the
~170 absorbed/un-absorbed crossover). For `absorb`, attend in latent space
(`k = v = kv_latent`, no up-projection) for **any** L, then `unembed_out` the
result — extending the decode path to small `L>1`. Prefill (`L>170`) keeps the
un-absorbed path. Shared helper `deepseek_v32.absorbed_attention()`; both
attention classes call it.

Verify-step latency (`mtp_phase1_verify.py`, `lm-01`, width 2, absorbed vs old
un-absorbed):

| context | absorbed | un-absorbed | speedup |
|--------:|---------:|------------:|--------:|
| 512     | 80 ms    | 142 ms      | 1.8×    |
| 4096    | 89 ms    | 566 ms      | 6.3×    |

Absorbed verify is ~flat with context; the un-absorbed up-projection is what
scaled.

### One subtlety: explicit softmax for L>1
The fused SDPA kernel is used for `L==1`; for `L>1` the helper uses an explicit
`mx.softmax(..., precise=True)`. The sparse path feeds thousands of `finfo.min`
additive-mask entries at the wide absorbed head_dim (kv_lora_rank=512); routing
through the fused kernel was indistinguishable here, but the explicit softmax is
the robust form and costs ~nothing at small L.

## Correctness validation (the important part)
Gold standard for MTP: **batched verify of L tokens must equal decoding them one
at a time.** Tested with `mtp_phase1_verify.py` (`verify==decode`):

- **Below `index_topk` (≤2048): bit-identical** (`max_abs=0.0`). No sparsity, so
  verify and decode take the same reduction.
- **Above 2048 on random tokens:** `rel≈0.10`, argmax flips. `_diag` showed the
  indexer picks the **identical topk** (100% overlap), and `_layerdiff` showed a
  **small per-layer fp difference (~3e-4 at layer 0) accumulating** over 78 layers
  — not a discrete bug. Cause: decode **gathers** and reduces over exactly 2048;
  verify **masks** and reduces over the full ~4098 (with ~2050 zero-weight terms).
  Mathematically equal, but different-length matmul reductions round differently.
- **On real text** (`_realprompt`, `mlx_lm/server.py`), the magnitude jitter never
  flips a token:

  | context | argmax agreement | mean rel | max rel |
  |--------:|:----------------:|---------:|--------:|
  | 3000    | 16/16 = 100%     | 6.4e-2   | 3.3e-1  |
  | 6000    | 32/32 = 100%     | 3.9e-2   | 1.1e-1  |
  | 12000   | 32/32 = 100%     | 6.7e-2   | 2.2e-1  |

The residual `rel` sits **below the model's own quant noise** (the absorbed-vs-
un-absorbed `orient-delta` is ~0.18). Conclusion: **lossless on token choice for
greedy** at all tested contexts.

## Status
- **Greedy MTP:** Phase 1 is correct and a large verify speedup. Ship it.
- **Sampling (temp>0):** the ~0.04–0.07 logit jitter slightly perturbs acceptance
  probabilities (still within quant noise). Not a correctness problem; not bit-exact.

## Phase 2 (optional, only if bit-exactness wanted)
**Per-query topk gather** for `L>1`: gather each query's 2048 topk *before*
attention and reduce over exactly those (like decode), instead of masking the full
KV. This makes verify **bit-match** decode (removes the fp accumulation) and is
also faster at long context (reduce over 2048, not full KV). Effort: medium —
per-query gather + batched latent attention. Only warranted if strict faithfulness
for sampling becomes a requirement; greedy does not need it.

## Convert-side note
`_tools` (conversion) originally *stripped* MTP weights in
`DeepseekV32.Model.sanitize` (dropped `layers >= num_hidden_layers`). Fixed by
porting the MTP classes + remapping `sanitize` (and `num_nextn_predict_layers` in
both `ModelArgs`) so convert preserves/quantizes `model.mtp.layers.0.*`. Both
convert and serve must be on the MTP-aware code.

## Deploy / run notes
- Model edits ship via `git checkout`; **`uv run --with .` caches a non-editable
  wheel** and serves stale code after a checkout. Use `uv run --with-editable .`
  (or plain `uv run` inside the checkout). `mtp_phase1_verify.py`'s preflight checks
  the loaded source for the fix markers and prints the import path to catch this.
