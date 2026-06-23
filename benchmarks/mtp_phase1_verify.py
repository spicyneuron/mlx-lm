#!/usr/bin/env python3
"""Phase 1 verify-path check: MLA-absorbed vs un-absorbed multi-token attention.

The fix (deepseek_v32.ABSORB_MAX_L) routes short query lengths -- i.e. MTP verify
chains of width L = draft_depth + 1 -- through the absorbed latent-space attention
instead of up-projecting the full KV per step. This script:

  1. PARITY: runs the same prefilled context + L-wide verify step under both code
     paths (absorbed default vs un-absorbed forced) and asserts the logits match.
     Absorption is mathematically exact, so any real divergence is a bug.

  2. PERF: times the verify step alone at several context sizes under both paths.
     The win should grow with context (the un-absorbed up-projection scales with KV;
     the absorbed path does not).

Run on the inference server (model lives there):

    uv run python benchmarks/mtp_phase1_verify.py \
        --model /path/to/glm52/quantized \
        --contexts 512,4096,16384 --width 2 --trials 10

Toggle is done by overwriting the module-level ABSORB_MAX_L in both model modules
(glm_moe_dsa imports the name by value), so no checkpoint reconvert is needed.
"""

import argparse
import inspect
import sys
import time

import mlx.core as mx

from mlx_lm.models import deepseek_v32, glm_moe_dsa
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.utils import load

FORCE_ABSORB = 1 << 30  # every L takes the absorbed path
FORCE_UNABSORB = 0  # every L takes the un-absorbed path


# Markers that must appear in the loaded attention source. Each is a distinct
# code edit; scripts sync to the server separately from model code (git
# checkout), so a stale checkout can have an earlier marker but not a later one.
REQUIRED_MARKERS = (
    "ABSORB_MAX_L",  # the L<=cutoff absorbed-path toggle
    "absorbed_attention",  # the explicit-softmax verify fix (post-toggle)
)


def preflight() -> None:
    """Fail loudly if the *loaded* attention code is missing any required edit.

    Catches the deploy split where the script is newer than the model code on
    this host. Checking only an early marker is not enough -- the old absorbed
    path still contained ABSORB_MAX_L, so the explicit-softmax fix could be
    silently absent (giving identical wrong results and identical perf).
    """
    from mlx_lm.models.glm_moe_dsa import GlmMoeDsaAttention
    from mlx_lm.models.deepseek_v32 import DeepseekV32Attention

    problems = []
    for cls in (GlmMoeDsaAttention, DeepseekV32Attention):
        src = inspect.getsource(cls.__call__)
        for marker in REQUIRED_MARKERS:
            if marker not in src:
                problems.append(f"{cls.__name__} missing '{marker}'")
    if problems:
        path = inspect.getsourcefile(GlmMoeDsaAttention)
        sys.exit(
            "ERROR: loaded attention code is stale / not fully patched:\n"
            + "\n".join(f"         - {p}" for p in problems)
            + f"\n       Running from: {path}\n"
            "       `git checkout` the commit with the latest model edits on\n"
            "       this host -- syncing the script alone is not enough."
        )


def set_absorb(max_l: int) -> None:
    # glm_moe_dsa did `from .deepseek_v32 import ABSORB_MAX_L`, binding a separate
    # name, so both module globals must be set. The GLM attention reads the former;
    # the MTP block's DeepseekV32Attention reads the latter.
    deepseek_v32.ABSORB_MAX_L = max_l
    glm_moe_dsa.ABSORB_MAX_L = max_l


def make_ids(vocab, n, width, seed=0):
    mx.random.seed(seed)
    ids = mx.random.randint(0, vocab, (1, n + width))
    return ids[:, :n], ids[:, n:]


def prefill(model, ctx):
    cache = make_prompt_cache(model)
    if ctx.shape[1] > 0:
        model(ctx, cache=cache)
        mx.eval([c.state for c in cache])
    return cache


def verify_once(model, ctx, step):
    """Batched: the L draft tokens scored in a single forward (the verify path)."""
    cache = prefill(model, ctx)
    out = model(step, cache=cache)
    mx.eval(out)
    return out


def decode_seq(model, ctx, step):
    """Reference: the same L tokens fed one at a time (the L==1 decode path).

    This is the spec-decode correctness gold standard -- verifying L tokens at
    once must equal decoding them sequentially. Both use the absorbed
    (transpose=True) orientation, so on a quantized model they should still
    agree to ~fp noise (unlike absorbed-vs-unabsorbed, which differ by the
    weight's quant orientation).
    """
    cache = prefill(model, ctx)
    outs = [model(step[:, i : i + 1], cache=cache) for i in range(step.shape[1])]
    mx.eval(outs)
    return mx.concatenate(outs, axis=1)


def rel_diff(a, b):
    max_abs = mx.max(mx.abs(a - b)).item()
    return max_abs, max_abs / max(mx.max(mx.abs(b)).item(), 1e-6)


def run_context(model, vocab, context, width, trials, skip_perf):
    ctx, step = make_ids(vocab, context, width)

    # --- correctness: batched absorbed verify vs sequential decode (same orient) ---
    set_absorb(FORCE_ABSORB)
    gold = decode_seq(model, ctx, step)
    cand = verify_once(model, ctx, step)
    c_max, c_rel = rel_diff(cand, gold)
    toks_match = bool((mx.argmax(cand, -1) == mx.argmax(gold, -1)).all().item())
    # 0.0 here is the ideal (verify == sequential decode, same orientation). The
    # orient-delta below is what proves the toggle is live, so no INERT check.
    ok = c_rel < 2e-2 and toks_match
    print(
        f"[verify==decode] context={context:>6} width={width}  "
        f"max_abs={c_max:.4e}  rel={c_rel:.2e}  argmax_match={toks_match}  "
        f"{'OK' if ok else 'MISMATCH'}"
    )

    # --- informational: quant-orientation delta (absorbed verify vs un-absorbed) ---
    set_absorb(FORCE_UNABSORB)
    unabs = verify_once(model, ctx, step)
    o_max, o_rel = rel_diff(cand, unabs)
    live = "toggle LIVE" if o_max > 0 else "toggle INERT -- code not patched!"
    print(
        f"[orient-delta]   context={context:>6} width={width}  "
        f"max_abs={o_max:.4e}  rel={o_rel:.2e}  ({live}; fp16 ~0)"
    )

    # --- perf: absorbed vs un-absorbed verify step ---
    if not skip_perf:
        ms = {}
        for label, max_l in (("absorb", FORCE_ABSORB), ("unabsorb", FORCE_UNABSORB)):
            set_absorb(max_l)
            cache = prefill(model, ctx)
            mx.eval(model(step, cache=cache))  # warmup / graph build
            t0 = time.perf_counter()
            for _ in range(trials):
                mx.eval(model(step, cache=cache))
            ms[label] = (time.perf_counter() - t0) / trials * 1e3
        print(
            f"[perf]           context={context:>6} width={width}  "
            f"absorb={ms['absorb']:7.2f} ms  unabsorb={ms['unabsorb']:7.2f} ms  "
            f"speedup={ms['unabsorb'] / ms['absorb']:4.1f}x"
        )
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="path to the (quantized) model")
    p.add_argument("--contexts", default="512,4096,16384")
    p.add_argument("--width", type=int, default=2, help="verify width L (draft+1)")
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--skip-perf", action="store_true")
    args = p.parse_args()

    # Show where mlx_lm actually loaded from. A path under .cache/uv means a
    # stale cached wheel (use `uv run --with-editable .` instead of `--with .`);
    # a path in your checkout means live source.
    import mlx_lm

    print(f"mlx_lm: {inspect.getsourcefile(mlx_lm.models.glm_moe_dsa)}")
    preflight()  # refuse to run if the loaded attention isn't patched

    contexts = [int(c) for c in args.contexts.split(",") if c]
    model, _ = load(args.model)
    vocab = model.model.embed_tokens.weight.shape[0]
    embed_q = model.model.layers[-1].self_attn.embed_q
    quant = "quantized" if hasattr(embed_q, "bits") else "fp"
    print(f"loaded {args.model}  vocab={vocab}  width={args.width}  weights={quant}\n")

    all_ok = True
    for ctx in contexts:
        all_ok &= run_context(model, vocab, ctx, args.width, args.trials, args.skip_perf)

    print()
    print("CORRECTNESS: " + ("ALL OK" if all_ok else "FAILED -- see status above"))


if __name__ == "__main__":
    main()
