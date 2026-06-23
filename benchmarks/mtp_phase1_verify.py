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
import time

import mlx.core as mx

from mlx_lm.models import deepseek_v32, glm_moe_dsa
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.utils import load

FORCE_ABSORB = 1 << 30  # every L takes the absorbed path
FORCE_UNABSORB = 0  # every L takes the un-absorbed path


def set_absorb(max_l: int) -> None:
    # glm_moe_dsa did `from .deepseek_v32 import ABSORB_MAX_L`, binding a separate
    # name, so both module globals must be set. The GLM attention reads the former;
    # the MTP block's DeepseekV32Attention reads the latter.
    deepseek_v32.ABSORB_MAX_L = max_l
    glm_moe_dsa.ABSORB_MAX_L = max_l


def verify_step(model, ctx_ids, step_ids):
    """Prefill ctx_ids, then run the L-wide verify step. Returns (logits, cache)."""
    cache = make_prompt_cache(model)
    if ctx_ids.shape[1] > 0:
        model(ctx_ids, cache=cache)
        mx.eval([c.state for c in cache])
    logits = model(step_ids, cache=cache)
    mx.eval(logits)
    return logits, cache


def make_ids(vocab, n, width, seed=0):
    mx.random.seed(seed)
    ids = mx.random.randint(0, vocab, (1, n + width))
    return ids[:, :n], ids[:, n:]


def parity(model, vocab, context, width):
    ctx, step = make_ids(vocab, context, width)

    set_absorb(FORCE_ABSORB)
    la, _ = verify_step(model, ctx, step)
    set_absorb(FORCE_UNABSORB)
    lu, _ = verify_step(model, ctx, step)

    diff = mx.abs(la - lu)
    max_abs = mx.max(diff).item()
    scale = mx.max(mx.abs(lu)).item()
    rel = max_abs / max(scale, 1e-6)
    tok_a = mx.argmax(la[:, -1], axis=-1).item()
    tok_u = mx.argmax(lu[:, -1], axis=-1).item()
    ok = rel < 1e-2 and tok_a == tok_u
    print(
        f"[parity] context={context:>6} width={width}  "
        f"max_abs={max_abs:.4e}  rel={rel:.2e}  "
        f"argmax(absorb)={tok_a} argmax(unabsorb)={tok_u}  "
        f"{'OK' if ok else 'MISMATCH'}"
    )
    return ok


def time_step(model, vocab, context, width, trials):
    ctx, step = make_ids(vocab, context, width)
    results = {}
    for label, max_l in (("absorb", FORCE_ABSORB), ("unabsorb", FORCE_UNABSORB)):
        set_absorb(max_l)
        # Prefill once, then time repeated verify steps on the warmed cache.
        cache = make_prompt_cache(model)
        model(ctx, cache=cache)
        mx.eval([c.state for c in cache])
        for _ in range(2):  # warmup / graph build
            mx.eval(model(step, cache=cache))
        t0 = time.perf_counter()
        for _ in range(trials):
            mx.eval(model(step, cache=cache))
        ms = (time.perf_counter() - t0) / trials * 1e3
        results[label] = ms
    speedup = results["unabsorb"] / results["absorb"]
    print(
        f"[perf]   context={context:>6} width={width}  "
        f"absorb={results['absorb']:7.2f} ms  "
        f"unabsorb={results['unabsorb']:7.2f} ms  "
        f"speedup={speedup:4.1f}x"
    )
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="path to the (quantized) model")
    p.add_argument("--contexts", default="512,4096,16384")
    p.add_argument("--width", type=int, default=2, help="verify width L (draft+1)")
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--skip-perf", action="store_true")
    args = p.parse_args()

    contexts = [int(c) for c in args.contexts.split(",") if c]
    model, _ = load(args.model)
    model.eval()
    vocab = model.model.embed_tokens.weight.shape[0]
    print(f"loaded {args.model}  vocab={vocab}  width={args.width}\n")

    all_ok = True
    for ctx in contexts:
        all_ok &= parity(model, vocab, ctx, args.width)
    print()
    if not args.skip_perf:
        for ctx in contexts:
            time_step(model, vocab, ctx, args.width, args.trials)

    print()
    print("PARITY: " + ("ALL OK" if all_ok else "FAILED -- absorbed path diverges"))


if __name__ == "__main__":
    main()
