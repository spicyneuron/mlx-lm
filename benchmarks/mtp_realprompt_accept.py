#!/usr/bin/env python3
"""D2 MTP acceptance on REAL long text vs the benchmark's random prompt.

The 16k "cliff" (D2 acceptance ~51%) was measured on a seed-0 random-token prompt
-- the worst case for a drafter, and we saw acceptance swing 51%<->94% just from a
different random prompt. The dense-vs-sparse proxy ruled out the MTP indexer as the
cause, so the remaining question is whether the cliff is even real on coherent text
(which held ~91% at 4k). This measures D2 acceptance on a real document at several
context lengths.

    cat mlx_lm/*.py > $TMPDIR/big.txt
    uv run --with-editable . benchmarks/mtp_realprompt_accept.py \
        --model ../../glm-5.2/lm-01-mtp --prompt-file $TMPDIR/big.txt \
        --contexts 4096,16384 --num-draft-tokens 2
"""

import argparse

import mlx.core as mx

from mlx_lm.utils import load
from mlx_lm.generate import stream_generate


def run(model, tokenizer, prompt, mtp, n_draft, prefill_step_size):
    """Generate 256 tokens; return (acceptance, generation_tps). mtp=False = baseline."""
    stats = {}
    kw = dict(max_tokens=256, prefill_step_size=prefill_step_size, temp=0.0)
    if mtp:
        kw.update(
            mtp=True,
            num_draft_tokens=n_draft,
            mtp_stats_callback=lambda s: (stats.clear(), stats.update(s)),
        )
    last = None
    for last in stream_generate(model, tokenizer, prompt, **kw):
        pass
    acc, prop = stats.get("accepted", 0), stats.get("proposed", 0)
    rate = acc / prop if prop else 0.0
    return rate, last.generation_tps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--prompt-file", required=True)
    p.add_argument("--contexts", default="4096,16384")
    p.add_argument("--num-draft-tokens", type=int, default=2)
    p.add_argument("--prefill-step-size", type=int, default=2048)
    args = p.parse_args()

    model, tokenizer = load(args.model)
    tokenizer._eos_token_ids = {}  # force full 256-token generations
    ids_all = tokenizer.encode(open(args.prompt_file).read())
    contexts = [int(c) for c in args.contexts.split(",") if c]
    print(f"loaded {args.model}  doc tokens={len(ids_all)}  D{args.num_draft_tokens}\n")

    print(f"{'context':>7} {'base_tps':>9} {'mtp_tps':>8} {'speedup':>8} {'accept':>7}")
    for C in contexts:
        if len(ids_all) < C:
            print(f"{C:>7}  SKIP (need {C} tokens, have {len(ids_all)})")
            continue
        prompt = ids_all[:C]
        _, base_tps = run(model, tokenizer, prompt, False, args.num_draft_tokens, args.prefill_step_size)
        acc, mtp_tps = run(model, tokenizer, prompt, True, args.num_draft_tokens, args.prefill_step_size)
        speedup = mtp_tps / base_tps if base_tps else 0.0
        print(f"{C:>7} {base_tps:>9.2f} {mtp_tps:>8.2f} {speedup:>7.2f}x {acc:>7.1%}")

    print(
        "\nspeedup > 1 at 16k on real text => the random-prompt 'loss' was an artifact;\n"
        "MTP wins across the range and the context gate may be unnecessary.\n"
        "speedup < 1 => real-text envelope is bounded; gate at the crossover."
    )


if __name__ == "__main__":
    main()
