# Native MTP on GLM-DSA (GLM-5.2): characterization, fix & final verdict

**Date:** 2026-06-24  **Hardware:** M3 (host `m3`), MLX 0.31.2
**Model:** GLM-5.2 `glm_moe_dsa`, 4-bit/gs-64 affine quant. MTP-head variants:
`lm-03`/`lm-01` (4-bit head) and `lm-01-mtp` (bf16 head,
`--q-override 'model\.mtp\.=bfloat16'`).
**Runtime branch:** `_glm-mtp`  **Drivers:** `mtp_characterize.sh`,
`mtp_phase1_verify.py`, `mtp_dense_proxy.py`, `mtp_realprompt_accept.py`
(+ `_diag`, `_layerdiff`, `_realprompt`, `mtp_phase1_accept.py`, `mtp_indexer_probe.py`)

> This is a preservation-oriented record: all measurements are kept. Numbers taken
> on `mlx_lm.benchmark`'s **random-token prompt** are labeled `[random]` — they are
> real but, as Section 4 shows, do **not** represent real-text behavior. The final
> verdict rests on the `[real-text]` numbers.

## TL;DR / final verdict
1. **Phase 1 (absorbed verify) is a legitimate, general fix** — MTP verify forward
   no longer scales with context (4096 verify ~566 ms → ~89 ms, **6.3×**). Greedy-
   lossless. Worth upstreaming regardless of MTP economics.
2. **The MTP head must be bf16, not 4-bit** — a single quantized layer has no error
   budget; bf16 (one extra layer, ~3% memory) is the correct recipe.
3. **`mlx_lm.benchmark`'s random prompts misrepresent acceptance** — they drove a
   wrong conclusion twice. Real text is flat ~81–84% acceptance with no long-context
   cliff, but the throughput win is only **~break-even**.

**`[real-text]` D2 + bf16 head — the numbers that decide it:**

| context | baseline tps | MTP tps | speedup | acceptance |
|--------:|-------------:|--------:|:-------:|-----------:|
| 4096  | 16.87 | 17.41 | **1.03×** | 81.4% |
| 16384 | 16.62 | 15.38 | **0.93×** | 84.2% |

**Decision: do not maintain a fork for the MTP path** (gain below maintenance
threshold). **Upstream Phase 1** as a standalone verify fix.

---

## 1. Phase 1: absorbed verify (the keeper)

### 1a. Pre-Phase-1 loss `[random]` (ckpt `lm-03`, before the fix)
| prompt | config | gen tps | ms/tok | accept | draft ms/tok | verify ms/tok |
|-------:|--------|--------:|-------:|-------:|-------------:|--------------:|
| 512  | baseline | 15.20 | 65.8 | — | — | — |
| 512  | MTP D1   | 10.07 | 99.3 | 98.4% | 2.96 | 97.74 |
| 4096 | baseline | 14.15 | 70.7 | — | — | — |
| 4096 | MTP D1   | **3.11** | **321.6** | 96.9% | 3.05 | **317.62** |

Decode (L=1) was flat with context (65.8→70.7); verify (L>1) tripled (97.7→317.6).
Drafting was cheap and acceptance excellent, so the whole problem was the L>1 verify.

### 1b. Root cause
In `GlmMoeDsaAttention` / `DeepseekV32Attention`:
- **`L == 1` (decode):** gather each query's sparse `topk` KV, attend in the
  **absorbed** kv_lora latent space. Cost bounded by `index_topk` (2048) → flat.
- **`L > 1` (verify):** **un-absorbed** — `embed_q`/`unembed_out` up-project the
  **full** KV to per-head `[B,H,KV,128]` every step. Up-projection (`heads·KV·512·128`)
  is the dominant, context-scaling cost. Correct for prefill (large L amortizes it),
  ruinous for verify (tiny L, nothing to amortize).

### 1c. The fix
`absorb = L <= ABSORB_MAX_L` (64, below the ~170 crossover). For `absorb`, attend in
latent space (`k = v = kv_latent`, no up-projection) for any L, then `unembed_out`.
Prefill keeps the un-absorbed path. Shared helper `deepseek_v32.absorbed_attention()`.
For `L>1` it uses explicit `mx.softmax(..., precise=True)` (the sparse mask injects
thousands of `finfo.min` entries at head_dim 512; explicit softmax is the robust
form, ~free at small L).

Verify-step latency (`mtp_phase1_verify.py`, `lm-01`, width 2):

| context | absorbed | un-absorbed | speedup |
|--------:|---------:|------------:|--------:|
| 512  | 80 ms | 142 ms | 1.8× |
| 4096 | 89 ms | 566 ms | 6.3× |

### 1d. Correctness — greedy lossless
Gold standard: batched verify of L tokens == decoding them one at a time.
- ≤ `index_topk` (2048): **bit-identical** (`max_abs=0.0`) — no sparsity, same reduction.
- Above 2048: small per-layer fp difference accumulates (decode *gathers*+reduces over
  2048; verify *masks*+reduces over full KV — equal math, different reduction length).
  `_diag`: identical topk (100% overlap). `_layerdiff`: accumulation (~3e-4 at layer 0
  → grows over 78 layers), not a discrete bug.
- Real text (`_realprompt`, `mlx_lm/server.py`): argmax agreement **100%**, jitter
  never flips a token:

  | context | argmax agreement | mean rel | max rel |
  |--------:|:----------------:|---------:|--------:|
  | 3000  | 16/16 = 100% | 6.4e-2 | 3.3e-1 |
  | 6000  | 32/32 = 100% | 3.9e-2 | 1.1e-1 |
  | 12000 | 32/32 = 100% | 6.7e-2 | 2.2e-1 |

  Residual `rel` sits **below the model's own quant noise** (absorbed-vs-un-absorbed
  `orient-delta` ≈ 0.18). **Lossless on token choice for greedy.** (temp>0 never
  measured end-to-end; jitter within quant noise. Moot given the shelve decision.)

---

## 2. End-to-end throughput & the bf16-head unlock

### 2a. 4-bit head characterize `[random]` (`mtp_characterize.sh`, temp=0, gen=256, ckpt `lm-01`)
Baselines: 512 = 18.20 tps, 4096 = 16.96 tps.

| context | config | gen tps | accept | draft ms/tok | verify ms/tok |
|--------:|--------|--------:|-------:|-------------:|--------------:|
| 512  | D1          | 17.24 | 96.9% | 2.90 | 56.06 |
| 512  | **D2**      | **21.10** | 92.2% | 2.64 | 43.65 |
| 512  | D1 no-bonus | 12.50 | 98.0% | 2.75 | 77.82 |
| 4096 | D1          | 13.75 | 66.7% | 3.01 | 59.50 |
| 4096 | D2          | 14.97 | 60.6% | 2.79 | 47.85 |
| 4096 | D1 no-bonus | 11.72 | 73.3% | 2.92 | 83.00 |

Verify is now flat (~44–83 ms/tok, not scaling). With 4-bit head, `[random]`
acceptance collapses 92%→61% (D2) from 512→4096 — *appeared* acceptance-bound.

### 2b. bf16 head `[random]` (ckpt `lm-01-mtp`), 4096:
| depth | 4-bit head | bf16 head |
|------:|-----------|-----------|
| D1 | 66.7%, 13.75 | 74.7%, 13.86 |
| **D2** | 60.6%, 14.97 | **91.2%, 18.28** |
| D3 | — | 62.2%, 15.44 |

Keeping one layer in bf16 swung D2 acceptance **+30 pts** (60.6→91.2). Cost ~3% peak
mem (446 vs 431 GB-units), ~1 ms/tok draft. Depth: **D1** verify-cost-capped (can't
beat baseline even at 100% accept); **D2** sweet spot; **D3** overshoots (3rd
autoregressive draft compounds error, acceptance crashes, cycle 188 ms sinks it).

### 2c. Norm-recycle fix (#2) `[random]`, 4096/16384
`mtp_forward` now recycles the post-`shared_head` hidden (vLLM parity, see §5).
- 4096 D2: 91.2% → **92.2%**, 18.28 → 19.0 tps (small bump, no regression).
- 16384 D2: 50.2% → **51.0%** (essentially unchanged — not the cliff's cause).

### 2d. The apparent cliff `[random]`, D2 + bf16, by context:
| context | D2 acceptance | gen tps |
|--------:|--------------:|--------:|
| 512   | 92.2% | 21.10 |
| 4096  | 91.2% | 18.28 |
| 16384 | **50.2% / 51.0%** | 11.56 |
| 16384 (D1, token-1 proxy) | 67.3% | 11.84 |

D1@16k = 67.3% (token-1 holds) but D2@16k = 51% (token-2 collapses): the D2<D1 flip
vs D2>D1 at 4k said **token-2** was the victim. This looked like a hard cliff — **but
it is a random-prompt artifact** (Section 4).

---

## 3. Did the absorbed L>1 (mask) verify hurt acceptance? No.
`mtp_phase1_accept.py` measured `vg==vm = 100%` — the L>1 mask verify picks the
**identical** token as the L==1 gather decode at every position. So the absorbed
verify is acceptance-neutral. (An earlier hypothesis that the ~6 pt bonus-vs-no-bonus
gap was a "mask penalty" was wrong — that gap is trajectory noise.) **Phase 2**
(per-query gather, bit-exact verify) is therefore purely a sampling-faithfulness
lever, **not** an acceptance/throughput lever. Not needed for greedy.

---

## 4. The benchmark trap & real-text results (the decisive correction)
`mlx_lm.benchmark` feeds a **seed-0 random-token prompt**. We confirmed random-prompt
acceptance swings **51%↔94%** just from a different random seed. It misrepresents in
both directions — degenerates into *easy repetition* at 4k (inflated 92%) and
*hard* output at 16k (deflated 51%).

`[real-text]` (`mtp_realprompt_accept.py`, `cat mlx_lm/*.py` = 77,854 tokens, D2,
bf16 head):

| context | random D2 accept | **real-text D2 accept** | base tps | MTP tps | speedup |
|--------:|-----------------:|------------------------:|---------:|--------:|--------:|
| 4096  | 92.2% | **81.4%** | 16.87 | 17.41 | **1.03×** |
| 16384 | 50.2% | **84.2%** | 16.62 | 15.38 | **0.93×** |

**No cliff on real text** — acceptance is flat ~81–84% across 4k→16k (the 16k
collapse was entirely the random prompt). But the real-text win is **marginal**:
1.03× at 4k, 0.93× at 16k. At 16k, verify cost (not acceptance) is the binding
constraint. (Caveat: concatenated source is somewhat repetitive, which can flatter
the absolute acceptance; the *no-cliff comparison* across contexts is robust because
the repetition is present at both lengths. The 512 win was never re-tested on real
text and remains `[random]`-only.)

---

## 5. Ruled-out / vetted implementation differences vs vLLM
- **Verify cost** — fixed by Phase 1 (§1).
- **Missing MTP weights at load** — `_tools` `sanitize` stripped `layers >=
  num_hidden_layers`; fixed by porting MTP classes + remapping sanitize +
  `num_nextn_predict_layers` in both `ModelArgs`.
- **MTP-head precision** — real; bf16 fixes it (§2b).
- **#2 norm recycling** (confirmed vs vLLM `deepseek_mtp.py:122-127`): we recycled
  the raw pre-`shared_head` hidden; vLLM recycles the post-norm hidden. **Fixed**
  (recycle `shared_head(mtp_hidden)`). +1% @4k, ~0 @16k. Correctness fix, kept.
- **#1 index sharing** (`index_share_for_mtp_iteration=True` in config, dropped by
  our loader; confirmed vs vLLM `set_skip_topk` + proposer `llm_base_proposer.py:508-534`):
  we recompute MTP topk every draft step instead of reusing step-0's. **Not
  implemented; proven non-causal.** `mtp_dense_proxy` upper-bound test (force MTP
  attention fully dense = superset of any topk strategy), `[random]` 16384 D2:
  **sparse 51.0% (proposed 253) vs dense 49.4% (proposed 257)** → dense ≈ sparse, so
  no topk-selection change can help. Single-step `mtp_indexer_probe` `[real-text]`
  4096: sparse 88.9% vs dense 90.5% (+1.6%) — also innocent. Caveat: the D2 dense
  proxy ran on the *random* prompt, but #1 was only motivated by the (artifactual)
  cliff, and verify cost — not acceptance — binds at 16k on real text. Not worth the
  plumbing.
- **Per-query gather (Phase 2)** — acceptance-neutral (§3); sampling-bit-exactness
  only. Not needed.
- **RoPE interleave** (`indexer_rope_interleave=True`, also dropped) — reviewed and
  **aligned**, not a bug: MLX `traditional=True` is the interleaved layout, which is
  exactly what vLLM selects via `is_neox_style = not indexer_rope_interleave`
  (`deepseek_v2.py:1034`); the main MLA rope is also `is_neox_style=False`
  (`:505`). Latent risk only if a future checkpoint sets the flag False — add the
  field + an assert if upstreamed.
- **Full-accept MTP cache drift** (`mtp_catchup_ab.py`) — **real but benign; not the
  ceiling.** On a full-accept block the bonus path jumps to `target_y[n_draft]` and
  never feeds the last accepted draft token through the MTP, so `mtp_cache` drops one
  position per full-accept block; its offset drifts −1 each time (`max_cache_skew`
  climbs to 74 over 256 D2 tokens) while `model_cache` stays correct. Standard spec
  decode handles this ("include the last draft token", `generate.py` `_draft_generate`);
  the MTP path did not. A catch-up step (`mtp_full_accept_catchup`) closes it exactly
  (skew → 1), but the A/B is a **deterministic net regression**: aligning the cache
  *lowered* D2 acceptance (4096 81.4→80.1%, 16384 84.2→80.6%) and tps. So the drift
  was tolerable-to-helpful and the ~82% ceiling is genuine head capacity, not this
  bug — the one untested confound on the shelve decision, now ruled out. (Likely
  cause of the regression: the catch-up writes K/V from the verify hidden, not the
  MTP recurrent hidden the rest of the cache uses.) Toggle left OFF by default.

Relevant config: `num_hidden_layers=78`, `num_nextn_predict_layers=1`,
`index_topk=2048`, `index_topk_freq=4`, `index_skip_topk_offset=3`,
`index_share_for_mtp_iteration=True`, `indexer_rope_interleave=True`; `indexer_types`
length 78 (last "full" layer = 74), so the MTP layer (idx 78) is off the end of the
array — handled by building the MTP block as a plain full-indexer `DeepseekV32`
attention.

---

## 6. If revisited / operating notes
- **Operating point** (if ever shipped): D2 (`num_draft_tokens=2`), **bf16 MTP head**.
  No context gate needed — real-text throughput is flat ~break-even, not a cliff.
- **The only lever that could change the verdict** is a *better drafter* (deeper /
  higher-capacity / better-trained MTP head). Acceptance (~82%) and verify cost — not
  the attention path — now bound throughput. That's retraining; out of scope.
- **Requant recipe:** `--q-override 'model\.mtp\.=bfloat16'` (matches every MTP
  submodule — `…mtp_block.*`, `…eh_proj`, `…shared_head.norm` — and nothing in the
  backbone). Convert and serve must both be on MTP-aware code.
- **Deploy gotcha:** `uv run --with .` caches a non-editable wheel and serves stale
  code after a `git checkout`. Use `uv run --with-editable .` (or plain `uv run`
  inside the checkout). `mtp_phase1_verify.py`'s preflight checks the loaded source
  for fix markers and prints the import path.

## 7. Recommendation
**Upstream Phase 1** (`absorbed_attention`) + the **#2 norm-recycle fix** as
standalone correctness/perf fixes — both are independently correct and helpful for
any L>1 forward on this architecture. **Shelve the MTP speculative path**: real-text
gain is ~break-even, below the cost of maintaining a fork. Revisit only if a stronger
MTP head becomes available.
