#!/usr/bin/env python3
"""Localize the >2048-context verify mismatch: indexer selection vs attention.

verify==decode is exact below index_topk (no sparsity) but diverges above it,
so the bug lives in the topk-sparse path. Two candidates:

  (a) INDEXER: batched L>1 picks a different topk *set* than sequential L==1.
  (b) ATTENTION: same topk set, but the L>1 mask path != the L==1 gather path.

This captures the topk indices emitted by the first full-indexer layer during a
batched verify (L tokens at once) vs sequential decode (one token at a time) and
reports the per-position set overlap. ~100% overlap => bug is (b) attention;
materially <100% => bug is (a) indexer selection.

    uv run --with . benchmarks/mtp_phase1_diag.py --model ../../glm-5.2/lm-01 \
        --context 4096 --width 2
"""

import argparse

import mlx.core as mx

from mlx_lm.models.deepseek_v32 import Indexer
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.utils import load


def first_full_indexer(model):
    for i, layer in enumerate(model.model.layers):
        if getattr(layer.self_attn, "indexer", None) is not None:
            return i, layer.self_attn.indexer
    raise SystemExit("no full-indexer layer found")


def capture(target, setup, measured):
    """Run setup() un-recorded (prefill), then measured() with target recorded.

    Prefill at context > index_topk also invokes the indexer, so recording must
    start only after prefill -- otherwise the prefill's topk is captured too.
    """
    setup()
    orig = Indexer.__call__
    records = []

    def patched(self, *a, **k):
        out = orig(self, *a, **k)
        if self is target and out is not None:
            records.append(out)
        return out

    Indexer.__call__ = patched
    try:
        measured()
    finally:
        Indexer.__call__ = orig
    mx.eval(records)
    return records


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--context", type=int, default=4096)
    p.add_argument("--width", type=int, default=2)
    args = p.parse_args()

    model, _ = load(args.model)
    vocab = model.model.embed_tokens.weight.shape[0]
    lf, target = first_full_indexer(model)
    print(f"loaded {args.model}  vocab={vocab}  first full-indexer layer={lf}")
    print(f"context={args.context}  width={args.width}\n")

    mx.random.seed(0)
    ids = mx.random.randint(0, vocab, (1, args.context + args.width))
    ctx, step = ids[:, : args.context], ids[:, args.context :]

    def prefill(cache):
        model(ctx, cache=cache)
        mx.eval([c.state for c in cache])

    cache_b = make_prompt_cache(model)
    rb = capture(
        target,
        lambda: prefill(cache_b),
        lambda: mx.eval(model(step, cache=cache_b)),
    )

    cache_s = make_prompt_cache(model)

    def seq_steps():
        for i in range(step.shape[1]):
            mx.eval(model(step[:, i : i + 1], cache=cache_s))

    rs = capture(target, lambda: prefill(cache_s), seq_steps)

    # batched: one record [1,1,L,topk]; sequential: L records [1,1,1,topk]
    assert len(rb) == 1, f"expected 1 batched indexer call, got {len(rb)}"
    assert len(rs) == args.width, f"expected {args.width} seq calls, got {len(rs)}"
    batched_idx = rb[0]
    topk = batched_idx.shape[-1]
    print(f"index_topk={topk}\n")

    for i in range(args.width):
        b_set = set(batched_idx[0, 0, i, :].tolist())
        s_set = set(rs[i][0, 0, 0, :].tolist())
        inter = len(b_set & s_set)
        print(
            f"position {i}: overlap {inter}/{topk} = {inter / topk:6.2%}  "
            f"(batched_only={len(b_set - s_set)}, seq_only={len(s_set - b_set)})"
        )

    print(
        "\n=> ~100% overlap: bug is in ATTENTION (gather vs mask).\n"
        "   materially <100%: bug is in INDEXER selection (batched vs sequential)."
    )


if __name__ == "__main__":
    main()
