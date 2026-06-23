#!/usr/bin/env python3
"""Proxy test: would fixing the MTP step-1+ topk selection recover D2 acceptance?

token-1 (D1) holds at long context but D2 collapses (D2<D1 at 16k), so the second
draft token is the victim -- the suspect is index_share_for_mtp_iteration (steps
1+ recompute topk from a speculative hidden instead of reusing step 0's). Properly
implementing index-share is real surgery, so first check the necessary condition:
force the MTP block to attend DENSE (no topk drop) for ALL draft steps. Dense is a
superset of "reuse step-0's 2048", so:

  dense >> sparse  => the step-1+ topk selection is the problem; index-share worth
                      implementing.
  dense ~= sparse  => topk selection is not it; index-share won't help; stop.

Uses the same seed-0 random prompt as mlx_lm.benchmark, so the sparse number here
should reproduce the benchmark's D2 acceptance (~51% at 16k).

    uv run --with-editable . benchmarks/mtp_dense_proxy.py \
        --model ../../glm-5.2/lm-01-mtp --prompt-tokens 16384 --num-draft-tokens 2
"""

import argparse

import mlx.core as mx

from mlx_lm.utils import load
from mlx_lm.generate import stream_generate


def mtp_indexers(model):
    out = []
    for layer in model.model.mtp.layers:
        ix = getattr(layer.mtp_block.self_attn, "indexer", None)
        if ix is not None:
            out.append(ix)
    return out


def measure(model, tokenizer, prompt, n_draft, prefill_step_size):
    stats = {}
    for r in stream_generate(
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
    return acc / prop if prop else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--prompt-tokens", type=int, default=16384)
    p.add_argument("--num-draft-tokens", type=int, default=2)
    p.add_argument("--prefill-step-size", type=int, default=2048)
    args = p.parse_args()

    model, tokenizer = load(args.model)
    idxs = mtp_indexers(model)
    if not idxs:
        raise SystemExit("MTP block has no indexer to toggle")
    real_topk = idxs[0].index_topk

    # same synthetic prompt as mlx_lm.benchmark
    mx.random.seed(0)
    vocab = model.model.embed_tokens.weight.shape[0]
    prompt = mx.random.randint(0, vocab, (1, args.prompt_tokens)).tolist()[0]
    print(f"loaded {args.model}  prompt_tokens={args.prompt_tokens} "
          f"D{args.num_draft_tokens}  MTP index_topk={real_topk}\n")

    for ix in idxs:
        ix.index_topk = real_topk
    sparse = measure(model, tokenizer, prompt, args.num_draft_tokens, args.prefill_step_size)
    print(f"[sparse] MTP acceptance: {sparse:6.1%}")

    for ix in idxs:
        ix.index_topk = 1 << 30  # force dense: indexer returns None
    dense = measure(model, tokenizer, prompt, args.num_draft_tokens, args.prefill_step_size)
    for ix in idxs:
        ix.index_topk = real_topk
    print(f"[dense]  MTP acceptance: {dense:6.1%}")

    print(f"\ndelta = {dense - sparse:+.1%}")
    print(
        "dense >> sparse => step-1+ topk selection is the problem; implement index-share.\n"
        "dense ~= sparse => topk selection innocent; index-share won't help; stop."
    )


if __name__ == "__main__":
    main()
