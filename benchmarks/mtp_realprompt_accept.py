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


def measure(model, tokenizer, prompt, n_draft, prefill_step_size):
    stats = {}
    for _ in stream_generate(
        model,
        tokenizer,
        prompt,
        max_tokens=256,
        prefill_step_size=prefill_step_size,
        mtp=True,
        num_draft_tokens=n_draft,
        mtp_stats_callback=lambda s: (stats.clear(), stats.update(s)),
        temp=0.0,
    ):
        pass
    acc = stats.get("accepted", 0)
    prop = stats.get("proposed", 0)
    return (acc / prop if prop else 0.0), prop


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

    print(f"{'context':>7} {'real_acc':>9} {'proposed':>9}")
    for C in contexts:
        if len(ids_all) < C:
            print(f"{C:>7}  SKIP (need {C} tokens, have {len(ids_all)})")
            continue
        prompt = ids_all[:C]
        acc, prop = measure(model, tokenizer, prompt, args.num_draft_tokens, args.prefill_step_size)
        print(f"{C:>7} {acc:>9.1%} {prop:>9}")

    print(
        "\nIf real_acc holds high at 16k (~vs the 91% at 4k), the random-prompt 51%\n"
        "was an artifact and the win likely extends. If real_acc also collapses, the\n"
        "16k limit is real on coherent text -> gate MTP at the measured crossover."
    )


if __name__ == "__main__":
    main()
