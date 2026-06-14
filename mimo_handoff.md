# MiMo-V2.5 MLX-LM Loop Investigation — Handoff for Independent Review

You are reviewing an in-progress debugging effort for a confirmed generation
bug. The investigator has narrowed the suspect surface and reached a tentative
conclusion. **Your job is adversarial: read the code, run the references, and
either corroborate or invalidate the working hypothesis. Do not anchor on it.**
The investigator wants the strongest, most truthful explanation — not a
rubber-stamp.

---

## 1. Model and environment

- Model: **Xiaomi MiMo-V2.5** (HF repo `XiaomiMiMo/MiMo-V2.5`, model_type
  `mimo_v2`). It's a hybrid SWA/full-attention MoE with attention sinks on SWA
  layers, partial RoPE (factor 0.334), and `attention_value_scale: 0.707`.
- MLX-LM model file: `mlx_lm/models/mimo_v2.py` on a branch called `_mimo` in
  the local repo, equivalent (per loaded module path in tracebacks) to what
  PyPI ships as `mlx_lm.models.mimo_v2`. There is *also* a sibling file
  `mlx_lm/models/mimo_v2_flash.py` registered as model_type `mimo_v2_flash`
  that handles a related variant without `attention_value_scale`. Both are
  worth diffing.
- Checkpoint on disk: `/Volumes/Models/mlx/311-mimo-2.5`. It was obtained by
  dequantizing an upstream fp8 release; the dequant logic is inlined in
  `Model.sanitize()` in `mimo_v2.py:354-488`. The checkpoint is then loaded
  *quantized* by the standard `mlx_lm.load(...)` path (uint32-packed
  weights), which matters for any patch that tries to multiply
  `v_proj.weight` directly.
- Generation harness: `mlx_lm` server in production, plus our diagnostic
  script (described below) for A/B tests. Loops reproduce across both, with
  and without the official chat template, with both `enable_thinking=true`
  and `false`, with and without prompt-cache reuse.

## 2. Symptom

Catastrophic single-token repetition during chat generation. In the
diagnostic run below it manifests as ~8 consecutive emissions of token id 18
(suspected newline-ish) after ~640 generated tokens on a prompt asking for 25
Python unit tests. Standard MCQ evals (ARC-Challenge, HellaSwag) do *not*
show obvious degradation, which suggests the per-step logit *ranking* stays
roughly correct even when the residual stream is drifting toward an
attractor.

## 3. Hypotheses considered and current status

Each was tested either statically (code read) or empirically (A/B patch in
the script, described in §4):

| # | Hypothesis | Status |
|---|---|---|
| H1 | `mx.fast.scaled_dot_product_attention` mishandles `head_dim=192` / `v_head_dim=128` (diff KV head dim). MLX docstring describes a single `D` across q/k/v; vLLM uses a dedicated `FlashAttentionDiffKVBackend` exactly for this case. | **Ruled out.** Phase A numerical probe (no model load) shows the fast kernel matches a manual fp32 GQA softmax to bf16 tolerance (rel err 4.2e-3). Phase C `manual_sdpa` variant (manual fp32 softmax that *definitely* handles diff KV dims and sinks) still loops identically. |
| H2 | MoE router GEMM precision. vLLM (`../vllm/vllm/model_executor/models/mimo_v2.py:155-199`) and HF (`temp-mimo/modeling_mimo_v2.py:155`) explicitly run the gate matmul with fp32 input and weight. MLX-LM runs it in bf16 with a post-GEMM `astype(float32)` for sigmoid only. | **Ruled out** for this failure. Phase C `router_fp32` variant (cast x to fp32 and use a cached fp32 copy of `weight`) still loops identically. Worth fixing for parity, but not the cause of the loop. |
| H3 | Chat-template / prompt-cache prefix-duplication via `mlx_lm/chat.py:120-156` (cache reused across turns while template re-emits system header). | **Ruled out by user observation.** Loops occur on turn 1 with a fresh cache. |
| H4 | `RotatingKVCache` (`mlx_lm/models/cache.py:410-578`) corruption during long SWA prefill (`sliding_window_size: 128`). | **Not directly tested**, but loops reproduce on prompts whose prefill is well under 128 tokens, weakening this. |
| H5 | Attention sink bias dtype / `mx.fast.scaled_dot_product_attention` interaction. | **Diagnostic only.** Phase C `no_sinks` patch *also* breaks output (loops earlier on a different token, id 1773), showing sinks are required for coherent generation but doesn't prove they're correct. |
| H6 | `attention_value_scale = 0.707` is being applied somewhere it shouldn't be (this is the current working hypothesis). | **Strongly suggested by empirics.** Three mathematically-equivalent applications of 0.707 all reproduce the loop. Only fully disabling the scale yields coherent output. See §5. |

## 4. Diagnostic methodology

We wrote `mimo_diag.py` (in the repo root) as a single autonomous diagnostic.
The script is self-contained, uses PEP 723 inline deps so `uv run` installs
mlx and mlx-lm automatically, and patches `mlx_lm.models.mimo_v2` at runtime
via monkey-patching of `Attention.__call__`, `MoEGate.__call__`, and the
module-level `scaled_dot_product_attention` import. No model conversion or
config edit is required to run it.

Pipeline:
1. **Phase A** — numerical probe of `mx.fast.scaled_dot_product_attention` at
   `D_qk=192`, `D_v=128` against a manual fp32 GQA softmax reference. No
   model load. Tests both fp32 and bf16.
2. **Phase B** — generate a battery of prompts under greedy (temp=0) *and* a
   second pass under `temp=0.7` with fixed seed. Fresh prompt cache per
   generation, max_tokens=1500. Streaming loop detector (single-token
   run-length ≥8 OR n-gram-of-length-3-to-12 repeating ≥4 times at the tail
   over the last 120 tokens) early-stops each generation.
3. **Phase C** — once a deterministic looper is identified, the same prompt
   is run under each of the following monkey-patched variants, fresh cache,
   same seed:
   - `baseline` — unpatched mimo_v2
   - `router_fp32` — cast `x` to fp32 and use cached fp32 copy of
     `MoEGate.weight` before the gate GEMM
   - `manual_sdpa` — replace the module-level `scaled_dot_product_attention`
     with a manual fp32 softmax that handles diff-KV dims and sinks
     correctly (GQA expansion via `mx.repeat`, sink appended as extra column,
     softmax in fp32, sliced back)
   - `no_sinks` — set `self.attention_sink_bias = None` for the call
   - `no_v_scale` — set `self.v_scale = None` for the call
   - `v_scale_fp32` — cast values to fp32, multiply by `v_scale`, cast back
     (mathematically equivalent to baseline)
   - `v_scale_via_o_proj` — skip V scaling, multiply attention *output* by
     `v_scale` before residual add (mathematically equivalent)
   - `v_scale_baked_v_proj` — attempt to fold `v_scale` into
     `v_proj.weight` at first call (failed because weights are quantized
     uint32; this row is informationless and should be ignored)

Loop detection is intentionally aggressive but has a 32-token warmup, so
genuinely long coherent outputs aren't false-flagged.

## 5. Empirical results

### Phase A

```
mlx.core.float32   max|fast-ref|=1.788e-06  rel=9.343e-07  OK
mlx.core.bfloat16  max|fast-ref|=7.812e-03  rel=4.184e-03  OK
--> PASS
```

`mx.fast.scaled_dot_product_attention` accepts and correctly handles
`v_head_dim != head_dim` to bf16 tolerance.

### Phase B

15 prompts × 2 passes (greedy + temp=0.7) = 30 (pass, prompt) generations.
**1/30 looped**: greedy on
> "Write a Python script with 25 unit tests for a simple Calculator class.
> Each test should be in its own def and use a unique numerical case."

That single prompt looped at 640 tokens with `single-token x8 (id=18)`. All
other 29 generations either ran to EOS or hit the 1500-token cap cleanly.
This is unusual: the user reports loops across many circumstances in
production, but greedy + fresh cache + diverse prompts only catches one.
That asymmetry is itself a clue — see §7.

### Phase C on the looping prompt

| Variant | Result | Notes |
|---|---|---|
| `baseline` | LOOP @ 640 tok, id=18 | — |
| `router_fp32` | LOOP @ 720 tok, id=18 | Router precision is not it. |
| `manual_sdpa` | LOOP @ 656 tok, id=18 | Fast SDPA / diff-KV-dim is not it. |
| `no_sinks` | LOOP @ 32 tok, id=1773 | Sinks are required for coherence; not informative about correctness. |
| `no_v_scale` | **OK**, 992 tok to EOS, coherent | Only patch that produces coherent output. |
| `v_scale_fp32` | LOOP @ 640 tok, id=18 | Mathematically equivalent to baseline; still loops. |
| `v_scale_via_o_proj` | LOOP @ 640 tok, id=18 | Mathematically equivalent to baseline (V · 0.707 = output · 0.707 since `o_proj` is linear); still loops identically. |
| `v_scale_baked_v_proj` | crashed (uint32 weight, quantized) | Ignore. |

The investigator's interpretation: any application of 0.707 in the V/output
path reproduces the failure exactly. The only way to recover is to skip the
scale entirely. Most natural explanation: the *effective* scale is already
present in the checkpoint (in `v_proj.weight`, `o_proj.weight`, or somewhere
upstream), and runtime application doubles it to ~0.5, compounding to
near-zero attention output magnitude across 64 layers until the residual
stream collapses.

## 6. Current working hypothesis (treat with suspicion)

> The MLX checkpoint at `/Volumes/Models/mlx/311-mimo-2.5` already has the
> 0.707 factor folded into the attention output path (most plausibly
> `v_proj.weight` or `o_proj.weight`), so `mimo_v2.py:128-129`'s runtime
> `values = values * self.v_scale` is a redundant second application. The
> upstream fp8 release likely baked it in during quantization; HF inference
> would exhibit the same bug on the same fp8 release; the bf16 release
> probably does not have it baked and HF works there. The sibling
> `mimo_v2_flash.py` (model_type `mimo_v2_flash`) has no `v_scale` handling
> at all, which is consistent with that file having been written for a
> checkpoint where the scale was already baked in.

Proposed mechanical fix: set `attention_value_scale: null` in the
checkpoint's `config.json`, or unconditionally remove the v_scale plumbing
from `mimo_v2.py` if all real-world MiMo-V2.5 MLX checkpoints have it baked.

## 7. Reasons this hypothesis might still be wrong

Please pressure-test these specifically:

1. **`v_scale_via_o_proj` should be numerically distinguishable from
   `v_scale_fp32`** (different rounding paths) yet both loop at exactly the
   same token count and id. That suggests either (a) the failure is on a
   knife-edge that both perturbations happen to fall the same side of, or
   (b) the model is genuinely so close to attractor collapse that any
   non-zero 0.707 application destabilizes it. (b) is more consistent with
   the data but doesn't fully explain why the failure is so reproducibly at
   token 640.
2. The HF reference (`temp-mimo/modeling_mimo_v2.py:302-303`) and vLLM
   (`../vllm/vllm/model_executor/models/mimo_v2.py:327-329`) both apply
   `v = v * v_scale` at runtime. If the HF safetensors don't have v_scale
   baked, the official HF inference works correctly. We're inferring that
   the *fp8 release* might be different — but that's an inference, not a
   measurement.
3. The dequant code in `mimo_v2.py:395-405` (`dequant_block`) and
   `split_qkv` (`mimo_v2.py:407-439`) doesn't reference `v_scale`, so
   whatever bake-in happened, it happened upstream of MLX's `sanitize()`. If
   it didn't happen upstream either, the bake-in story collapses and we need
   a different explanation.
4. `no_v_scale` produces output that looks coherent to the loop detector,
   but we have not validated it against any ground-truth reference. It could
   be coherently *wrong* (e.g., subtly off-distribution but locally
   plausible). MCQ evals being insensitive doesn't disprove this.
5. The investigator did not test that the loop reproduces under
   `mimo_v2_flash` (which is registered as a separate model_type and would
   require the user's config to say `model_type: mimo_v2_flash`). If
   mimo_v2_flash on this same checkpoint also loops, the bake-in story is
   harder to maintain.

## 8. Where to read

In this repo (`/Users/main/Sites/SN/mlx-lm`, branch `_mimo`):

- `mlx_lm/models/mimo_v2.py` — the model under test. Pay specific attention
  to:
  - `ModelArgs` (lines 18-62): config field declarations, including
    `attention_value_scale: Optional[float] = None`.
  - `Attention.__init__` (lines 64-106): stores `self.v_scale =
    args.attention_value_scale`.
  - `Attention.__call__` (lines 108-148): applies `values = values *
    self.v_scale` *before* RoPE on Q/K and *before* `cache.update_and_fetch`.
    Then calls `scaled_dot_product_attention(..., sinks=...)`.
  - `Model.sanitize` (lines 354-488): handles fp8 dequant (`dequant_block`,
    `from_fp8`, block-`scale_inv` multiplication), TP-aware fused-QKV split
    via `detect_tp()` / `split_qkv()`, expert stacking. Look for any
    weight-touching code that could implicitly scale `v_proj` or `o_proj`.
  - `Model.cast_predicate` (lines 537-542): only excludes
    `e_score_correction_bias` from dtype casting. Worth noting that
    `attention_sink_bias` is cast like everything else.
- `mlx_lm/models/mimo_v2_flash.py` — sibling implementation, *no*
  `attention_value_scale` handling. If this is the "correct" path for the
  same family of checkpoints, that's diagnostic.
- `mlx_lm/models/base.py:24-55` — `create_causal_mask`,
  `create_attention_mask`, and the wrapper
  `scaled_dot_product_attention` that mimo_v2 imports.
- `mlx_lm/models/cache.py:410-578` — `RotatingKVCache`. Read
  `_update_concat`, `_update_in_place`, `make_mask` carefully if you want to
  pressure-test H4.
- `tests/test_models.py:2927-2959` — the existing smoke test for `mimo_v2`.
  Uses `head_dim=96`, `v_head_dim=64`, `attention_value_scale=0.707`. It's a
  shape/forward-pass test only, no numerical comparison against HF — i.e. it
  would not catch this kind of bug.
- `temp-mimo/config.json` and `temp-mimo/modeling_mimo_v2.py` — local copy
  of Xiaomi's HF release. Most directly comparable reference; line numbers
  cited in §6 are stable here.
- `temp-mimo/chat_template.jinja` — official chat template. Re-emits the
  system header every render; relevant if you want to rule on H3 yourself.
- `mimo_diag.py` — the diagnostic script. Self-documenting.

Outside this repo, on the same machine:

- `../vllm/vllm/model_executor/models/mimo_v2.py` — vLLM reference. Pay
  attention to `MiMoV2MoE.forward` (gate fp32), `MiMoV2Attention.forward`
  (v_scale + diff-KV backend), and `MiMoV2FlashDecoderLayer.__init__` (sink
  bias wiring).
- `../llama.cpp/conversion/mimo.py` and `../llama.cpp/src/models/mimo2.cpp`
  — the llama.cpp port. Note `_tp_aware_qkv_dequant()` in the conversion
  (TP-rank-major QKV layout) and `value_scale`/`f_attn_value_scale`
  application in `mimo2.cpp:14-16, 91, 165-166`. llama.cpp also applies
  v_scale at runtime via `ggml_scale`.

## 9. Concrete asks of you

1. **Read `mimo_v2.py` end-to-end** before forming any opinion. Don't trust
   summary in §3 if you find something the investigator missed.
2. **Diff `Attention.__call__` in `mimo_v2.py` against
   `temp-mimo/modeling_mimo_v2.py:290-356` (the HF reference)** with eyes
   open for *any* mismatch: ordering of v_scale vs RoPE vs cache update,
   shape of `attention_sink_bias`, how `scaled_dot_product_attention`'s
   `sinks` kwarg differs from HF's "sink as extra softmax column"
   convention, RoPE half-rotation vs interleaved.
3. **Diff against `mlx_lm/models/mimo_v2_flash.py`** to see what the
   `mimo_v2_flash` path does differently. The most informative diff is
   whatever changed when `mimo_v2.py` was forked from it (if it was).
4. Look at `Model.sanitize` and the fp8 dequant for anything that could
   implicitly bake a factor into `v_proj` or `o_proj`. If you don't find
   anything, that *weakens* the bake-in hypothesis and you should propose
   an alternative.
5. **Propose at least one alternative hypothesis** that's consistent with
   *all* of:
   - Phase A: fast SDPA numerically fine for diff-KV-dim.
   - Phase C: three mathematically-equivalent applications of 0.707 all
     loop identically.
   - Phase C: skipping 0.707 entirely produces coherent output.
   - Single-token attractor at id 18 (newline-ish), not random noise.
   - MCQ evals undegraded.
   - User reports loops reproduce broadly in production but our greedy
     scan only caught 1/15 prompts. Sampling at temp=0.7 caught 0/15.
6. Decide whether the working hypothesis is the most parsimonious
   explanation of the evidence, or whether something else fits better.
   Write up your conclusion with the same level of specificity (file:line
   references).

## 10. Practical considerations

- You can re-run the diagnostic with
  `uv run mimo_diag.py /Volumes/Models/mlx/311-mimo-2.5 \
    --looper-prompt "<the prompt above>"`
  which skips Phase B and re-runs Phase C in ~3 minutes.
- You can extend the variant list in `mimo_diag.py`'s `main()` (search for
  `variants = []`) with any further patch you want to test.
- The user does not have a vLLM environment, so token-level cross-checks
  against vLLM output need to be done via static code reading or
  hypothetical analysis, not measurement.
- The user has separately mentioned they will rerun ARC-C / HellaSwag with
  `attention_value_scale: null` while you work. If those eval scores
  *improve*, that supports the working hypothesis; if they *drop*,
  `no_v_scale` is masking the symptom rather than fixing it and you should
  argue for that.
