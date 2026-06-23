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

**Verdict (greedy):** with Phase 1, the right drafter precision, and the right
depth, native MTP is a **net win up to ~4k context** — D2 + bf16-MTP-head gives
**+16% at 512** and **+7.8% at 4096**. It **crosses back to a loss by 16k**
(acceptance collapses 91%→50%), so the win is **bounded and needs a context-length
gate** (~4–8k). Operating point: **`num_draft_tokens=2`, MTP head kept in bf16,
gated by context**. Open risks: (1) margins are greedy — **sampling (temp>0) is
unmeasured and may erode the 4096 win**; (2) exact gate crossover (8k) not yet
pinned.

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

## End-to-end throughput: the bf16-head unlock
Phase 1 made verify flat, but the end-to-end win then became **acceptance-bound**
at long context. Characterize (`mtp_characterize.sh`, temp=0, gen=256), 4-bit head
(`lm-01`), baselines 18.20 tps (512) / 16.96 tps (4096):

| context | D1 | D2 |
|--------:|----|----|
| 512  | 17.24 (96.9%) | **21.10 (92.2%)  +16%** |
| 4096 | 13.75 (66.7%) | 14.97 (60.6%)  **loss** |

Verify was no longer the problem (flat ~50–60 ms/tok); **MTP-head acceptance
collapsed at long context** (92% → 61% at D2). Root cause: the MTP head is a
**single layer quantized to 4-bit** — almost no redundancy to absorb quant error,
and at long context it does its hardest job (sparse topk retrieval + long-range
extrapolation in one layer). Requantizing with the **MTP head kept in bf16**
(`--q-override 'model\.mtp\.=bfloat16'`, ckpt `lm-01-mtp`) recovers it. 4096:

| depth | 4-bit head | bf16 head |
|------:|-----------|-----------|
| D1 | 66.7%, 13.75 | 74.7%, 13.86 |
| **D2** | 60.6%, 14.97 | **91.2%, 18.28  +7.8%** |
| D3 | — | 62.2%, 15.44 |

Keeping one layer in bf16 swung D2 acceptance **+30 points** and flipped 4096 from
loss to win. Cost: peak memory +3% (446 vs 431), draft +~1 ms/tok. Depth story:
**D1** is verify-cost-capped (can't beat baseline even at 100% accept); **D2** is
the sweet spot; **D3** overshoots — the 3rd autoregressive draft compounds error,
acceptance crashes to 62%, the longer cycle (188 ms) sinks it.

(Curiosity, non-blocking: bf16 D2 accepts *more* per token than D1, 91% vs 75% —
backwards from the usual depth penalty, likely a block-alignment artifact over the
same greedy sequence; worth a glance that the D1 path isn't leaving tokens on the
table.)

### The win is bounded: it crosses back at long context
D2 + bf16, greedy, by context:

| context | D2 acceptance | gen tps | vs baseline |
|--------:|--------------:|--------:|:-----------:|
| 512   | 92.2% (4-bit) | 21.10 | +16% |
| 4096  | 91.2% | 18.28 | +7.8% |
| 16384 | **50.2%** | 11.56 | **loss** (baseline pending, but well below the 4096 base) |

Acceptance holds ~91% through 4k then **falls off a cliff by 16k**. The cliff is
well above the 2048 sparsity threshold, so it's the 1-layer head's long-range
capacity giving out in the 4k–16k band — bf16 raised the ceiling (4-bit crashed at
4k; bf16 holds to 4k) but didn't remove it. **Operating envelope: gate MTP on below
~4–8k, fall back to plain decode above.** Crossover not yet pinned (8k pending).

## Status
- **Greedy MTP:** correct (lossless on token choice) and a **net throughput win at
  512 and 4096** with D2 + bf16 head. Ship it at that operating point.
- **Sampling (temp>0): UNRESOLVED and the key risk.** Two effects compound: the
  ~0.04–0.07 absorbed-verify logit jitter (within quant noise), and — more
  importantly — stochastic acceptance is lower than greedy exact-match, so the
  **+7.8% at 4096 could shrink to break-even or a loss**. Must measure with
  `--temp 1.0 --top-p 0.95` before claiming a sampling win. 512's +16% has more
  headroom and is the safer bet.
- **Upper context bound:** win is **bounded** — D2+bf16 wins ≤4k, loses by 16k
  (acceptance 50%). Needs a **context-length gate** (~4–8k); exact crossover pending
  an 8k point + a 16k baseline.

## Phase 2 (optional, only if sampling bit-exactness wanted)
**Per-query topk gather** for `L>1`: gather each query's 2048 topk *before*
attention and reduce over exactly those (like decode), instead of masking the full
KV. Makes verify **bit-match** decode and is faster at very long context (reduce
over 2048, not full KV).

**It does NOT help acceptance/throughput.** `mtp_phase1_accept.py` measured
`vg==vm = 100%` — the mask path picks the identical verify token as gather at every
position, so it costs zero acceptance (an earlier hypothesis that the ~6pt
bonus-vs-no-bonus gap was a mask penalty was wrong; that gap was trajectory noise).
So Phase 2 is purely a **sampling-faithfulness** lever — only warranted if the
temp>0 measurement shows the jitter matters. Greedy does not need it.

## Convert-side note
`_tools` (conversion) originally *stripped* MTP weights in
`DeepseekV32.Model.sanitize` (dropped `layers >= num_hidden_layers`). Fixed by
porting the MTP classes + remapping `sanitize` (and `num_nextn_predict_layers` in
both `ModelArgs`) so convert preserves/quantizes `model.mtp.layers.0.*`. Both
convert and serve must be on the MTP-aware code.

**Requant recipe (the operating point):** quantize the body at 4-bit but keep the
MTP head in bf16 — it's one extra layer (~3% memory) and worth +30 acceptance
points at long context:
```
--q-override 'model\.mtp\.=bfloat16'
```
`model.mtp.` matches every MTP submodule (`…mtp_block.*`, `…eh_proj`,
`…shared_head.norm`) and nothing in the backbone.

## Deploy / run notes
- Model edits ship via `git checkout`; **`uv run --with .` caches a non-editable
  wheel** and serves stale code after a checkout. Use `uv run --with-editable .`
  (or plain `uv run` inside the checkout). `mtp_phase1_verify.py`'s preflight checks
  the loaded source for the fix markers and prints the import path to catch this.
