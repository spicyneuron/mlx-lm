#!/usr/bin/env python3
"""Does the verify-vs-decode divergence actually flip tokens on real text?

The layer-by-layer diff shows the absorbed verify (full-KV mask) accumulates a
small per-layer fp difference vs sequential decode (topk gather), because the
two reduce over different-sized sets once sparsity is active (>index_topk). On
random tokens this reaches rel~0.10 and flips the argmax -- but random tokens
are the worst case for argmax stability, and the gap is below the model's own
quant noise (the absorbed-vs-unabsorbed orient-delta ~0.18).

This measures the metric that actually matters for greedy MTP: over in-distribution
tokens at long context, how often does verify's argmax disagree with decode's?
~0% disagreement => the mask path is fine, skip the per-query-gather refactor.

    uv run --with-editable . benchmarks/mtp_phase1_realprompt.py \
        --model ../../glm-5.2/lm-01 --prompt-file long_doc.txt --width 16

Provide a text file with comfortably more than index_topk (2048) tokens so the
sparse path is exercised. Without --prompt-file it falls back to random tokens
(and says so) -- useful only as the pessimistic bound.
"""

import argparse

import mlx.core as mx

from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.utils import load


def prefill(model, ctx):
    cache = make_prompt_cache(model)
    model(ctx, cache=cache)
    mx.eval([c.state for c in cache])
    return cache


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--prompt-file", default=None)
    p.add_argument("--min-context", type=int, default=3000)
    p.add_argument("--width", type=int, default=16, help="verify positions to compare")
    args = p.parse_args()

    model, tok = load(args.model)
    vocab = model.model.embed_tokens.weight.shape[0]

    if args.prompt_file:
        with open(args.prompt_file) as f:
            ids_list = tok.encode(f.read())
        source = f"{args.prompt_file} ({len(ids_list)} tokens)"
        need = args.min_context + args.width
        if len(ids_list) < need:
            raise SystemExit(f"need >= {need} tokens, file has {len(ids_list)}")
        ids = mx.array(ids_list[:need])[None]
    else:
        source = "RANDOM TOKENS (pessimistic bound -- pass --prompt-file for real text)"
        mx.random.seed(0)
        ids = mx.random.randint(0, vocab, (1, args.min_context + args.width))

    ctx, step = ids[:, : args.min_context], ids[:, args.min_context :]
    print(f"loaded {args.model}  source: {source}")
    print(f"context={args.min_context}  compared positions={args.width}\n")

    # verify: one batched forward over the real next `width` tokens
    cache_v = prefill(model, ctx)
    cand = model(step, cache=cache_v)
    mx.eval(cand)

    # decode: same tokens fed one at a time
    cache_s = prefill(model, ctx)
    gold = mx.concatenate(
        [model(step[:, i : i + 1], cache=cache_s) for i in range(step.shape[1])], axis=1
    )
    mx.eval(gold)

    cand_tok = mx.argmax(cand, axis=-1)[0]
    gold_tok = mx.argmax(gold, axis=-1)[0]
    agree = (cand_tok == gold_tok)
    n_agree = int(agree.sum().item())
    n = step.shape[1]

    # per-position relative logit diff
    diff = mx.abs(cand - gold)[0]
    scale = mx.maximum(mx.max(mx.abs(gold)[0], axis=-1), 1e-6)
    rel = (mx.max(diff, axis=-1) / scale)
    mx.eval(rel)

    print(f"argmax agreement: {n_agree}/{n} = {n_agree / n:6.2%}")
    print(f"per-position rel logit diff: mean={rel.mean().item():.2e}  max={rel.max().item():.2e}")
    if n_agree < n:
        flips = [i for i in range(n) if not bool(agree[i].item())]
        print(f"disagreeing positions: {flips}")
    print()
    print(
        "100% agreement on real text => mask path is fine for greedy MTP.\n"
        "Frequent disagreement => per-query topk gather needed for faithfulness."
    )


if __name__ == "__main__":
    main()
