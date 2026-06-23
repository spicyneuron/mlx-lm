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


def preflight() -> None:
    """Fail loudly if the *loaded* attention code lacks the Phase 1 toggle.

    Scripts get synced to the server separately from the model code (git
    checkout). If the server's checkout predates the ABSORB_MAX_L edit, the
    toggle below is a no-op: every L>1 takes the un-absorbed path, parity is
    trivially bit-identical (max_abs=0.0) and perf is 1.0x. Catch that here
    instead of after a long sweep.
    """
    from mlx_lm.models.glm_moe_dsa import GlmMoeDsaAttention
    from mlx_lm.models.deepseek_v32 import DeepseekV32Attention

    missing = [
        cls.__name__
        for cls in (GlmMoeDsaAttention, DeepseekV32Attention)
        if "ABSORB_MAX_L" not in inspect.getsource(cls.__call__)
    ]
    if missing:
        src = inspect.getsourcefile(GlmMoeDsaAttention)
        sys.exit(
            "ERROR: loaded attention code is NOT patched for Phase 1 "
            f"({', '.join(missing)} has no ABSORB_MAX_L).\n"
            f"       Running from: {src}\n"
            "       The model code on this host is stale -- `git checkout` the\n"
            "       commit with the ABSORB_MAX_L edit, don't just sync the script."
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


def run_context(model, vocab, context, width, trials, skip_perf):
    """Prefill once per code path, then derive parity + timing from that cache.

    Prefill (esp. at long context) dominates wall time, so each setting is
    prefilled exactly once. The first verify step's logits feed the parity
    check; the remaining steps are timed.
    """
    ctx, step = make_ids(vocab, context, width)
    logits = {}
    ms = {}
    for label, max_l in (("absorb", FORCE_ABSORB), ("unabsorb", FORCE_UNABSORB)):
        set_absorb(max_l)
        cache = make_prompt_cache(model)
        if context > 0:
            model(ctx, cache=cache)
            mx.eval([c.state for c in cache])
        first = model(step, cache=cache)  # captured for parity
        mx.eval(first)
        logits[label] = first
        if not skip_perf:
            mx.eval(model(step, cache=cache))  # warmup / graph build
            t0 = time.perf_counter()
            for _ in range(trials):
                mx.eval(model(step, cache=cache))
            ms[label] = (time.perf_counter() - t0) / trials * 1e3

    la, lu = logits["absorb"], logits["unabsorb"]
    max_abs = mx.max(mx.abs(la - lu)).item()
    rel = max_abs / max(mx.max(mx.abs(lu)).item(), 1e-6)
    tok_a = mx.argmax(la[:, -1], axis=-1).item()
    tok_u = mx.argmax(lu[:, -1], axis=-1).item()
    # Exact 0.0 is NOT a pass: the two paths use different matmul orders, so a
    # working toggle yields small fp noise. Bit-identical means the same graph
    # ran both times -> the toggle never switched.
    if max_abs == 0.0:
        status, ok = "INERT (toggle did nothing)", False
    elif rel < 1e-2 and tok_a == tok_u:
        status, ok = "OK", True
    else:
        status, ok = "MISMATCH", False
    print(
        f"[parity] context={context:>6} width={width}  "
        f"max_abs={max_abs:.4e}  rel={rel:.2e}  "
        f"argmax={tok_a}/{tok_u}  {status}"
    )
    if not skip_perf:
        print(
            f"[perf]   context={context:>6} width={width}  "
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

    preflight()  # refuse to run if the loaded attention isn't patched

    contexts = [int(c) for c in args.contexts.split(",") if c]
    model, _ = load(args.model)
    vocab = model.model.embed_tokens.weight.shape[0]
    print(f"loaded {args.model}  vocab={vocab}  width={args.width}\n")

    all_ok = True
    for ctx in contexts:
        all_ok &= run_context(model, vocab, ctx, args.width, args.trials, args.skip_perf)

    print()
    print("PARITY: " + ("ALL OK" if all_ok else "FAILED -- see status above"))


if __name__ == "__main__":
    main()
