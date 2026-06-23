# Native MTP on GLM-DSA (GLM-5.2): characterization & verdict

**Date:** 2026-06-23  **Hardware:** M3 (host `m3`), MLX 0.31.2
**Model:** GLM-5.2 `glm_moe_dsa`, 4-bit/gs-64 affine quant (`glm-5.2/lm-03`)
**Runtime branch:** `_glm-mtp` (`mtp_generate_step`)  **Method:** `benchmarks/mtp_characterize.sh`, temp=0, gen=256, 3 trials

## TL;DR
Native MTP is a **net loss at every context length tested**, despite excellent
(96–98%) draft acceptance and cheap (~3 ms/tok) drafting. The cost is entirely
the **target verification forward**, which scales with context length while plain
decode does not. This is **architectural**, not an MTP bug, and not fixable by
history policy, draft depth, or the bonus-token variant. Shelved.

## Numbers

| prompt | config | gen tps | ms/tok | accept | draft ms/tok | verify ms/tok | cycle ms |
|-------:|--------|--------:|-------:|-------:|-------------:|--------------:|---------:|
| 512    | baseline (no MTP) | 15.20 | 65.8 | — | — | — | — |
| 512    | MTP D1 (bonus)    | 10.07 | 99.3 | 98.4% | 2.96 | 97.74 | 198.5 |
| 512    | MTP D2            | 13.11 | 76.3 | 93.8% | 2.78 | 72.01 | 221.6 |
| 512    | MTP D1 no-bonus   | 10.76 | 92.9 | 98.4% | 2.73 | 90.84 | 93.6 |
| 4096   | baseline (no MTP) | 14.15 | 70.7 | — | — | — | — |
| 4096   | MTP D1 (bonus)    | **3.11** | **321.6** | 96.9% | 3.05 | **317.62** | 638.3 |

(16384 row not run — diagnosis was already conclusive.)

## What the data isolates
- **Decode (L=1) is flat with context:** baseline 65.8 → 70.7 ms/tok (512 → 4096).
- **Verify (L>1) triples with context:** 97.7 → 317.6 ms/tok (512 → 4096).
- Draft forward is negligible (~3 ms/tok) and acceptance is excellent, so neither
  draft cost nor acceptance is the problem.
- D1/D2 at 512 split verify cost into ≈153 ms fixed + ≈21 ms/token. D2 beats D1
  only by amortizing the fixed slice; at 4096 the **scaling** slice dominates and
  swamps any amortization.

## Root cause
In `GlmMoeDsaAttention.__call__` / `DeepseekV32Attention.__call__` there are two paths:

- **`L == 1` (normal decode):** gathers each query's sparse `topk` KV *first*
  (`take_along_axis`), then applies `embed_q`/`unembed_out` and attends over only
  `topk` positions. Cost is bounded by `index_topk` (2048) → **flat** with context.
- **`L > 1` (every speculative verify):** applies `embed_q`/`unembed_out` and
  attends over the **entire** `kv_latent` (full KV), masking out non-topk positions
  instead of gathering. Cost grows with **full context length**.

Speculative decoding *always* verifies with `L > 1` (confirmed token + drafts), so
it can never use the sparse single-token path. DSA's sparsity — the very thing that
makes single-token decode cheap — does not apply to batched verification, so
speculation fights the architecture.

## Why this is not MTP-specific
The cost lives in the **target's** L>1 forward, not in the draft. A separate draft
*model* would hit the identical penalty. No draft quality or MTP history policy can
overcome it.

## Ruled out (do not revisit)
- MTP history policy (cycle / window / committed) — irrelevant; draft is already cheap.
- Draft depth tuning — D2 only amortizes the fixed slice; loses at long context.
- `--draft-mtp-no-bonus` — same verify cost, still a loss.
- Greedy/logprobs/host-sync post-processing — that is the ~150 ms *fixed floor* at
  512, but at 4096 verify is 318 ms/tok and attention dominates. Optimizing the
  floor does not touch the scaling term.

## The only real fix
Make the **L>1 attention path sparse** like the L==1 path: gather each query
position's `topk` KV *before* the `embed_q`/`unembed_out` projections and attention,
so verify cost is ≈ `L × topk` instead of full-KV. This removes the context-scaling
term and is the sole path to making *any* speculative scheme viable on GLM-DSA.

- **Effort:** medium-to-large — a custom block-sparse multi-query attention in MLX
  (per-query topk gather, indexer-output reshaping, validity masking).
- **Risk:** moderate — touches the hottest kernel.
- **Decision:** only worth it if long-context decode throughput is a priority worth
  real kernel investment. Otherwise conclude speculative decoding isn't a fit for
  this model.

## Convert-side note
`_tools` (the conversion branch) originally *stripped* MTP weights in
`DeepseekV32.Model.sanitize` (dropped `layers >= num_hidden_layers`), so quantized
checkpoints had no MTP layer and the runtime failed to load. Fixed by porting the
MTP module classes + the remapping `sanitize` (and `num_nextn_predict_layers` in
both `ModelArgs`) so convert preserves and quantizes `model.mtp.layers.0.*`. To use
MTP, both convert and serve must be on the MTP-aware code.
