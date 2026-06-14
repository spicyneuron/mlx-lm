#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "mlx",
#   "mlx-lm",
# ]
# ///
"""MiMo-V2.5 MLX-LM loop diagnostic — autonomous.

Usage (PEP 723 inline deps; uv installs everything on first run):
    uv run mimo_diag.py /path/to/mlx_mimo_model
    uv run mimo_diag.py /path/to/mlx_mimo_model --extra-prompts prompts.txt

If you need to run against a LOCAL fork of mlx-lm (e.g. a branch that
contains mlx_lm/models/mimo_v2.py that hasn't been released on PyPI):
    uv run --with /abs/path/to/your/mlx-lm-fork mimo_diag.py /path/to/model
or activate the venv that already has your fork and:
    python mimo_diag.py /path/to/mlx_mimo_model

What it does, with no further input:
    1. Numerical probe of mx.fast.scaled_dot_product_attention with
       head_dim=192, v_head_dim=128 (no model load required).
    2. Greedy-generates a battery of built-in prompts (+ any in
       --extra-prompts) under a FRESH prompt cache and looks for loops.
       Early-stops each generation as soon as a loop is detected.
    3. Picks the first deterministic looper and runs the same prompt
       under six monkey-patched variants:
         baseline, router_fp32, manual_sdpa, no_v_scale, no_sinks,
         manual+router.
    4. Prints a single VERDICT block at the end. Copy-paste that back.
"""
import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx


# ---------------------------------------------------------------------------
# Built-in prompt battery (diverse styles; any of these should trigger if
# the model is broadly broken in single-turn).
# ---------------------------------------------------------------------------
DEFAULT_PROMPTS = [
    # Long-form writing (sustained generation past sliding-window=128)
    "Write a 1500-word short story about an astronaut who discovers an "
    "abandoned alien library on Europa. Include vivid descriptions of the "
    "library's architecture, the artifacts found inside, and the astronaut's "
    "internal monologue throughout.",
    "Write a detailed 1000-word essay on the causes and consequences of the "
    "fall of the Western Roman Empire, with at least five distinct sections.",
    "Compose a long narrative poem of at least 40 stanzas about the cycle of "
    "the seasons, using consistent meter and an ABAB rhyme scheme.",

    # Enumeration / structured (repetition-priming)
    "Count from 1 to 100, one number per line, and give a brief one-sentence "
    "fun fact about each number.",
    "List 50 unique English nouns starting with the letter 'S', each on its "
    "own line with a short definition.",
    "Generate 30 distinct example sentences using the word 'serendipity'.",
    "Write a numbered list of 40 healthy dinner recipes with one sentence of "
    "description for each.",

    # Code with patterns (often induces repetitive token sequences)
    "Write a Python module that demonstrates 15 classic design patterns. For "
    "each pattern include a class, a docstring, and a short usage example.",
    "Write a Python script with 25 unit tests for a simple Calculator class. "
    "Each test should be in its own def and use a unique numerical case.",
    "Generate a TypeScript file that defines 20 React components, each with "
    "its own props interface and a short JSDoc comment.",

    # Translation / list-style (natural repetition structure)
    "Translate the sentence 'The quick brown fox jumps over the lazy dog' "
    "into 20 different languages, one per line, with the language name first.",

    # Math sequences (loop-prone)
    "List the first 50 prime numbers, then for each one give the sum of its "
    "digits on the same line.",

    # Long-prefill (forces SWA cache rotation during prefill itself)
    "Please carefully read the following passage and then write a 600-word "
    "critical analysis. Passage: " + ("In the beginning, philosophy and "
    "science were inseparable disciplines, both grounded in the conviction "
    "that careful observation and rigorous argument could uncover the hidden "
    "structure of reality. " * 20),

    # Short controls (kept in case any of these still reproduces)
    "Hello! How are you today?",
    "Write a short poem about the ocean.",
]


# ---------------------------------------------------------------------------
# Loop detection
# ---------------------------------------------------------------------------
def detect_loop(token_ids, min_ngram=3, max_ngram=12, min_repeats=4, tail=120):
    """Return (looped: bool, why: str)."""
    ids = token_ids[-tail:]
    n = len(ids)
    if n < min_ngram * min_repeats:
        return False, "too short"
    run = 1
    for i in range(1, n):
        if ids[i] == ids[i - 1]:
            run += 1
            if run >= 8:
                return True, f"single-token x{run} (id={ids[i]})"
        else:
            run = 1
    for ng in range(min_ngram, max_ngram + 1):
        if n < ng * min_repeats:
            continue
        tail_gram = tuple(ids[-ng:])
        reps = 1
        for k in range(2, min_repeats + 4):
            start = n - k * ng
            if start < 0:
                break
            if tuple(ids[start : start + ng]) == tail_gram:
                reps = k
            else:
                break
        if reps >= min_repeats:
            return True, f"{ng}-gram x{reps} at tail"
    return False, "no loop"


# ---------------------------------------------------------------------------
# Phase A: SDPA numerical probe, no model
# ---------------------------------------------------------------------------
def phase_a():
    print("\n========== PHASE A: mx.fast.scaled_dot_product_attention "
          "(D_qk=192, D_v=128) ==========")
    mx.random.seed(0)
    B, Nq, Nkv, Tq, Tkv = 1, 16, 4, 8, 32
    Dqk, Dv = 192, 128
    results = []
    for dtype in (mx.float32, mx.bfloat16):
        q = mx.random.normal((B, Nq, Tq, Dqk)).astype(dtype)
        k = mx.random.normal((B, Nkv, Tkv, Dqk)).astype(dtype)
        v = mx.random.normal((B, Nkv, Tkv, Dv)).astype(dtype)
        scale = Dqk ** -0.5

        groups = Nq // Nkv
        k32 = mx.repeat(k.astype(mx.float32), groups, axis=1)
        v32 = mx.repeat(v.astype(mx.float32), groups, axis=1)
        scores = (q.astype(mx.float32) @ k32.swapaxes(-1, -2)) * scale
        ref = (mx.softmax(scores, axis=-1) @ v32).astype(dtype)
        mx.eval(ref)

        try:
            fast = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
            mx.eval(fast)
        except Exception as e:
            verdict = f"ERROR {type(e).__name__}: {e} >>> REJECTS diff-KV dim"
            print(f"  {dtype}: {verdict}")
            results.append((str(dtype), "error", verdict))
            continue
        if fast.shape != ref.shape:
            verdict = f"SHAPE MISMATCH fast={fast.shape} ref={ref.shape}"
            print(f"  {dtype}: {verdict}")
            results.append((str(dtype), "shape", verdict))
            continue
        diff = mx.abs(fast.astype(mx.float32) - ref.astype(mx.float32)).max().item()
        scale_ref = mx.abs(ref.astype(mx.float32)).max().item() + 1e-9
        rel = diff / scale_ref
        tol = 5e-2 if dtype == mx.bfloat16 else 1e-4
        ok = rel < tol
        verdict = (f"max|fast-ref|={diff:.3e} rel={rel:.3e} "
                   f"{'OK' if ok else 'LARGE DIFF — fast kernel likely wrong'}")
        print(f"  {dtype}: {verdict}")
        results.append((str(dtype), "ok" if ok else "diff", verdict))
    failed = any(r[1] != "ok" for r in results)
    print(f"\nPhase A status: {'FAIL' if failed else 'PASS'}")
    return results, failed


# ---------------------------------------------------------------------------
# Generation (greedy, fresh cache, streaming early-stop on loop)
# ---------------------------------------------------------------------------
def render_prompt(tokenizer, user_text, enable_thinking=False, system_prompt=None):
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": user_text})
    try:
        return tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False
        )


def run_generate(model, tokenizer, user_text, max_tokens, enable_thinking=False,
                 temp=0.0, top_p=0.0, min_p=0.0, seed=0, check_every=16,
                 system_prompt=None, repetition_penalty=0.0,
                 repetition_context_size=20):
    """Run a generation. temp=0.0 → greedy; temp>0 → sampled with fixed seed."""
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler, make_logits_processors
    from mlx_lm.models.cache import make_prompt_cache
    mx.random.seed(seed)
    prompt = render_prompt(tokenizer, user_text, enable_thinking, system_prompt)
    sampler = make_sampler(temp=temp, top_p=top_p, min_p=min_p)
    logits_processors = None
    if repetition_penalty and repetition_penalty != 1.0:
        logits_processors = make_logits_processors(
            None, repetition_penalty, repetition_context_size
        )
    fresh_cache = make_prompt_cache(model)
    text, ids = "", []
    t0 = time.time()
    early_looped = False
    kwargs = {
        "max_tokens": max_tokens, "sampler": sampler,
        "prompt_cache": fresh_cache,
    }
    if logits_processors:
        kwargs["logits_processors"] = logits_processors
    for resp in stream_generate(model, tokenizer, prompt, **kwargs):
        text += resp.text
        ids.append(resp.token)
        if len(ids) >= 32 and len(ids) % check_every == 0:
            looped, _ = detect_loop(ids)
            if looped:
                early_looped = True
                break
    return text, ids, time.time() - t0, early_looped


# ---------------------------------------------------------------------------
# Monkey-patches
# ---------------------------------------------------------------------------
def patch_router_fp32():
    from mlx_lm.models import mimo_v2
    orig = mimo_v2.MoEGate.__call__

    def patched(self, x):
        if not hasattr(self, "_w_f32"):
            self._w_f32 = self.weight.astype(mx.float32)
        gates = x.astype(mx.float32) @ self._w_f32.T
        return mimo_v2.group_expert_select(
            gates, self.e_score_correction_bias,
            self.top_k, self.n_group, self.topk_group,
            self.routed_scaling_factor, self.norm_topk_prob,
        )

    return (lambda: setattr(mimo_v2.MoEGate, "__call__", patched),
            lambda: setattr(mimo_v2.MoEGate, "__call__", orig))


def patch_manual_sdpa():
    from mlx_lm.models import mimo_v2
    orig = mimo_v2.scaled_dot_product_attention

    def manual(queries, keys, values, cache, scale, mask, sinks=None):
        B, Nq, Tq, Dqk = queries.shape
        _, Nkv, Tkv, Dv = values.shape
        if Nq != Nkv:
            g = Nq // Nkv
            keys = mx.repeat(keys, g, axis=1)
            values = mx.repeat(values, g, axis=1)
        out_dtype = queries.dtype
        q = queries.astype(mx.float32)
        k = keys.astype(mx.float32)
        v = values.astype(mx.float32)
        scores = (q @ k.swapaxes(-1, -2)) * scale
        if mask is not None and not isinstance(mask, str):
            scores = scores + mask.astype(mx.float32)
        elif mask == "causal" and Tq > 1:
            causal = mx.triu(
                mx.full((Tq, Tkv), -float("inf")), k=Tkv - Tq + 1
            )
            scores = scores + causal
        if sinks is not None:
            sink = sinks.astype(mx.float32).reshape(1, Nq, 1, 1)
            sink = mx.broadcast_to(sink, (B, Nq, Tq, 1))
            ext = mx.concatenate([scores, sink], axis=-1)
            attn = mx.softmax(ext, axis=-1)[..., :Tkv]
        else:
            attn = mx.softmax(scores, axis=-1)
        return (attn @ v).astype(out_dtype)

    return (lambda: setattr(mimo_v2, "scaled_dot_product_attention", manual),
            lambda: setattr(mimo_v2, "scaled_dot_product_attention", orig))


def patch_no_v_scale():
    from mlx_lm.models import mimo_v2
    saved = {}

    def on():
        if "fn" in saved:
            return
        orig_call = mimo_v2.Attention.__call__
        saved["fn"] = orig_call

        def patched(self, x, mask=None, cache=None):
            real = self.v_scale
            self.v_scale = None
            try:
                return orig_call(self, x, mask=mask, cache=cache)
            finally:
                self.v_scale = real

        mimo_v2.Attention.__call__ = patched

    def off():
        if "fn" in saved:
            mimo_v2.Attention.__call__ = saved.pop("fn")

    return on, off


def patch_no_sinks():
    from mlx_lm.models import mimo_v2
    saved = {}

    def on():
        if "fn" in saved:
            return
        orig_call = mimo_v2.Attention.__call__
        saved["fn"] = orig_call

        def patched(self, x, mask=None, cache=None):
            real = self.attention_sink_bias
            self.attention_sink_bias = None
            try:
                return orig_call(self, x, mask=mask, cache=cache)
            finally:
                self.attention_sink_bias = real

        mimo_v2.Attention.__call__ = patched

    def off():
        if "fn" in saved:
            mimo_v2.Attention.__call__ = saved.pop("fn")

    return on, off


def _reimpl_call_with_v_scale(mode):
    """Return a patched Attention.__call__ that applies v_scale via `mode`.
    Mathematically equivalent paths; only the numerical handling differs.
    """
    from mlx_lm.models import mimo_v2

    def patched(self, x, mask=None, cache=None):
        B, L, _ = x.shape
        queries = (
            self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim).swapaxes(1, 2)
        )
        keys = (
            self.k_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim).swapaxes(1, 2)
        )
        values = (
            self.v_proj(x)
            .reshape(B, L, self.n_kv_heads, self.v_head_dim)
            .swapaxes(1, 2)
        )

        v_scale = self.v_scale
        if v_scale is not None and mode == "fp32":
            orig_dtype = values.dtype
            values = (values.astype(mx.float32) * float(v_scale)).astype(orig_dtype)
        elif v_scale is not None and mode == "via_o_proj":
            pass  # apply after attention
        elif v_scale is not None and mode == "baked_v_proj":
            if not getattr(self.v_proj, "_v_baked", False):
                self.v_proj.weight = self.v_proj.weight * float(v_scale)
                self.v_proj._v_baked = True
            # values already has the scaling baked in via weight
        elif v_scale is not None:  # default same as original
            values = values * v_scale

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = mimo_v2.scaled_dot_product_attention(
            queries, keys, values,
            cache=cache, scale=self.scale, mask=mask,
            sinks=self.attention_sink_bias,
        )
        output = self.o_proj(output.swapaxes(1, 2).reshape(B, L, -1))
        if v_scale is not None and mode == "via_o_proj":
            output = output * float(v_scale)
        return output

    return patched


def patch_kvcache_for_swa():
    """Replace RotatingKVCache (SWA layers) with unlimited KVCache. The
    windowed mask still gates attention to the last sliding_window_size
    keys; only the storage strategy differs. Tests whether the ring-buffer
    semantics are the bug."""
    from mlx_lm.models import mimo_v2
    from mlx_lm.models.cache import KVCache

    saved = {}

    def on():
        if "fn" in saved:
            return
        saved["fn"] = mimo_v2.Model.make_cache

        def make_cache(self):
            return [KVCache() for _ in self.layers]

        mimo_v2.Model.make_cache = make_cache

    def off():
        if "fn" in saved:
            mimo_v2.Model.make_cache = saved.pop("fn")

    return on, off


def patch_fp32_sinks():
    """Cast attention_sink_bias to fp32 before SDPA. Tests whether sink
    bias precision matters in the softmax denominator."""
    from mlx_lm.models import mimo_v2

    saved = {}

    def on():
        if "fn" in saved:
            return
        orig_call = mimo_v2.Attention.__call__
        saved["fn"] = orig_call

        def patched(self, x, mask=None, cache=None):
            if self.attention_sink_bias is not None and not getattr(
                self, "_sink_f32", False
            ):
                self.attention_sink_bias = self.attention_sink_bias.astype(mx.float32)
                self._sink_f32 = True
            return orig_call(self, x, mask=mask, cache=cache)

        mimo_v2.Attention.__call__ = patched

    def off():
        if "fn" in saved:
            mimo_v2.Attention.__call__ = saved.pop("fn")

    return on, off


def patch_v_scale_mode(mode):
    """mode in {'fp32', 'via_o_proj', 'baked_v_proj'}."""
    from mlx_lm.models import mimo_v2
    saved = {}

    def on():
        if "fn" in saved:
            return
        saved["fn"] = mimo_v2.Attention.__call__
        mimo_v2.Attention.__call__ = _reimpl_call_with_v_scale(mode)

    def off():
        if "fn" in saved:
            mimo_v2.Attention.__call__ = saved.pop("fn")
        # Unbake v_proj weights if mode did weight surgery
        if mode == "baked_v_proj":
            # Re-divide by v_scale on every Attention instance that got baked
            try:
                from mlx_lm.models.mimo_v2 import Attention as _A
            except Exception:
                return
            # Walk attribute graph isn't trivial without the model handle.
            # We rely on a sentinel: if _v_baked is True, divide once.
            # Caller passes `model` via on()? Simpler: we leave the bake in
            # place and rely on running this variant LAST. See main().
            pass

    return on, off


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="Path to MLX MiMo model dir")
    ap.add_argument("--extra-prompts", help="Optional file with extra prompts, one per line")
    ap.add_argument("--max-tokens", type=int, default=1500,
                    help="Hard cap per generation; early-stops on detected loop")
    ap.add_argument("--enable-thinking", action="store_true")
    ap.add_argument("--temp", type=float, default=0.7,
                    help="Sampling temperature for the second (stochastic) pass")
    ap.add_argument("--seed", type=int, default=0,
                    help="Random seed for sampled pass (reproducibility)")
    ap.add_argument("--skip-sampled", action="store_true",
                    help="Skip the stochastic pass (greedy only)")
    ap.add_argument("--looper-prompt",
                    help="Skip Phase B entirely and run Phase C with this prompt")
    ap.add_argument("--looper-temp", type=float, default=0.0,
                    help="Temperature for --looper-prompt (default greedy)")
    ap.add_argument("--looper-top-p", type=float, default=0.0,
                    help="top_p for --looper-prompt")
    ap.add_argument("--looper-min-p", type=float, default=0.0,
                    help="min_p for --looper-prompt")
    ap.add_argument("--num-seeds", type=int, default=10,
                    help="Per-variant seed count in Phase C (only matters when looper-temp>0)")
    ap.add_argument("--system-prompt", default=None,
                    help="Optional system prompt to inject (default: chat template's built-in)")
    args = ap.parse_args()

    overall_start = time.time()
    sdpa_results, sdpa_failed = phase_a()

    # ------------------------ Load model ------------------------
    print("\n========== LOADING MODEL ==========")
    t0 = time.time()
    from mlx_lm import load
    model, tokenizer = load(
        args.model,
        tokenizer_config={"trust_remote_code": True},
        trust_remote_code=True,
    )
    print(f"Loaded in {time.time()-t0:.1f}s")

    # ------------------------ Phase B: find a deterministic looper ------------------------
    scan_rows = []
    looper = None
    looper_text = ""
    looper_ids = []
    looper_pass = None

    if args.looper_prompt:
        # Phase B is fully skipped — Phase C's baseline IS the verification.
        looper = args.looper_prompt
        looper_text = ""
        looper_ids = []
        pass_name = (f"temp={args.looper_temp}/p={args.looper_top_p}"
                     f"/m={args.looper_min_p}")
        looper_pass = (pass_name, args.looper_temp)
    else:
        prompts = list(DEFAULT_PROMPTS)
        if args.extra_prompts:
            for line in Path(args.extra_prompts).read_text().splitlines():
                s = line.strip()
                if s and not s.startswith("#"):
                    prompts.append(s)

        passes = [("greedy", 0.0)]
        if not args.skip_sampled:
            passes.append((f"temp={args.temp}", args.temp))

        for pass_name, temp in passes:
            print(f"\n========== PHASE B[{pass_name}]: scanning prompts "
                  f"(max_tokens={args.max_tokens}) ==========")
            for i, p in enumerate(prompts):
                text, ids, dt, _ = run_generate(
                    model, tokenizer, p, args.max_tokens, args.enable_thinking,
                    temp=temp, seed=args.seed,
                )
                looped, why = detect_loop(ids)
                scan_rows.append((pass_name, p, looped, why, len(ids), dt))
                flag = "LOOP" if looped else "ok  "
                print(f"  [{i+1:2d}/{len(prompts)}] {flag}  {len(ids):4d} tok  "
                      f"{dt:5.1f}s  {why:30s}  {p[:55]!r}")
                if looped and looper is None:
                    looper, looper_text, looper_ids = p, text, ids
                    looper_pass = (pass_name, temp)

    # ------------------------ Phase C: A/B variants on looper ------------------------
    # ab_rows entries: (name, n_loop, n_seeds, sample_text, sample_ids, total_dt)
    ab_rows = []
    if looper is None:
        print("\n========== PHASE C: SKIPPED (no deterministic looper found) ==========")
    else:
        n_seeds = args.num_seeds if args.looper_prompt else 1
        # Sampling args used in Phase C must match the looper's pass.
        if args.looper_prompt:
            ab_temp = args.looper_temp
            ab_top_p = args.looper_top_p
            ab_min_p = args.looper_min_p
        else:
            ab_temp = looper_pass[1]
            ab_top_p = 0.0
            ab_min_p = 0.0

        print(f"\n========== PHASE C: A/B variants "
              f"(temp={ab_temp}, top_p={ab_top_p}, min_p={ab_min_p}, "
              f"seeds=0..{n_seeds-1}) ==========")

        # Final run: only the two candidate production fixes we want to
        # cross-check, with v_scale potentially restored in config.
        variants = [
            ("min_p=0.05",       [], [], {"min_p": 0.05}),
            ("rep_penalty=1.05", [], [], {"repetition_penalty": 1.05}),
        ]

        # Prepend baseline (no overrides) so it's just another row.
        variants = [("baseline", [], [], {})] + variants

        for entry in variants:
            name, ons, offs = entry[0], entry[1], entry[2]
            extra_kwargs = entry[3] if len(entry) > 3 else {}
            for fn in ons: fn()
            try:
                print(f"  {name:28s}: ", end="", flush=True)
                n_loop = 0
                loop_tails = []
                total_dt = 0.0
                for s in range(n_seeds):
                    gen_kwargs = dict(
                        temp=ab_temp, top_p=ab_top_p, min_p=ab_min_p,
                        system_prompt=args.system_prompt,
                    )
                    gen_kwargs.update(extra_kwargs)
                    text, ids, dt, _ = run_generate(
                        model, tokenizer, looper, args.max_tokens,
                        args.enable_thinking, seed=s, **gen_kwargs,
                    )
                    looped, why = detect_loop(ids)
                    total_dt += dt
                    print("L" if looped else "ok", end=" ", flush=True)
                    if looped:
                        n_loop += 1
                        loop_tails.append((s, why, text[-100:].replace("\n", "\\n")))
                print(f" -> {n_loop}/{n_seeds} loop  ({total_dt:.0f}s)")
                for s, why, tail in loop_tails:
                    print(f"      seed={s}  {why}: ...{tail!r}")
                ab_rows.append((name, n_loop, n_seeds, "", [], total_dt))
            finally:
                for fn in offs: fn()

    # ------------------------ Final verdict (copy-paste this) ------------------------
    print("\n" + "=" * 78)
    print("VERDICT — copy this whole block back")
    print("=" * 78)
    print(f"mlx version: {mx.__version__ if hasattr(mx, '__version__') else 'unknown'}")
    print(f"model path:  {args.model}")
    print(f"total wall:  {time.time()-overall_start:.1f}s")
    print()
    print("PHASE A (fast SDPA, D_qk=192 / D_v=128):")
    for dtype, status, msg in sdpa_results:
        print(f"  {dtype:20s} {status:6s} {msg}")
    print(f"  --> {'FAIL' if sdpa_failed else 'PASS'}")
    print()
    print("PHASE B (loop scan, fresh cache per gen):")
    n_loop = sum(1 for r in scan_rows if r[2])
    print(f"  {n_loop}/{len(scan_rows)} (pass, prompt) pairs looped.")
    for pass_name, p, looped, why, ntok, dt in scan_rows:
        print(f"  [{'LOOP' if looped else 'ok  '}] {pass_name:8s} {ntok:4d}tok "
              f"{dt:5.1f}s {why:24s} {p[:50]!r}")
    print()
    print("PHASE C (A/B variants on first looper):")
    if not ab_rows:
        print("  (skipped — no looping prompt found in Phase B)")
    else:
        print(f"  prompt: {looper!r}")
        print(f"  sampling: temp={ab_temp}, top_p={ab_top_p}, min_p={ab_min_p}")
        for name, n_loop, n_seeds, tail, _, dt in ab_rows:
            status = "LOOP" if n_loop == n_seeds else ("ok  " if n_loop == 0
                                                       else "MIX ")
            print(f"  [{status}] {name:24s} {n_loop}/{n_seeds} seeds loop  ({dt:.0f}s)")
        base_n = ab_rows[0][1]
        base_total = ab_rows[0][2]
        print()
        if base_n == base_total:
            # Baseline loops every seed; look for variants that strictly reduce
            full_fixes = [r[0] for r in ab_rows[1:] if r[1] == 0]
            partial = [(r[0], r[1], r[2]) for r in ab_rows[1:] if 0 < r[1] < r[2]]
            if full_fixes:
                print(f"  >>> Variant(s) that STOPPED the loop on every seed: "
                      f"{', '.join(full_fixes)}")
            if partial:
                print(f"  >>> Variant(s) with partial effect:")
                for n, l, t in partial:
                    print(f"      {n}: {l}/{t} still loop")
            if not full_fixes and not partial:
                print("  >>> No variant changed the loop rate.")
        elif base_n == 0:
            print("  >>> Baseline didn't loop in any seed; A/B is inconclusive.")
        else:
            print(f"  >>> Baseline loops {base_n}/{base_total} seeds (not reliable). "
                  "Compare variant rates to baseline.")
    print("=" * 78)


if __name__ == "__main__":
    main()
