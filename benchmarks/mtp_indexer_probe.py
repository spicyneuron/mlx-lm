#!/usr/bin/env python3
"""Is the long-context acceptance cliff a broken MTP indexer, or head capacity?

The MTP block runs its OWN full DSA indexer (built with layer_idx=78, off the end
of the length-78 indexer_types array) over its own KV cache -- an unverified path
(vLLM GLM-MTP is dense; transformers has no DSA MTP). Acceptance is ~92% below the
2048 sparsity threshold and falls in lockstep with the topk/context retention
ratio above it (91% @4k=50% kept, 50% @16k=12.5% kept), which is what a poor topk
selection in the head would look like.

This A/Bs the MTP head's draft quality with its sparse indexer ON vs forced DENSE
(no topk drop), at each context. Draft quality = P(MTP-head argmax == target greedy
argmax) over real tokens, mirroring _draft_block's pairing (prev hidden + current
token -> predict next).

  - DENSE >> SPARSE at 16k  => the MTP indexer is dropping context it shouldn't:
                               a real, fixable bug; the envelope may extend.
  - DENSE ~= SPARSE         => the head is capacity-limited; sparsity is innocent;
                               stopping is the right call.

Forcing dense is non-invasive: set the MTP block indexer's index_topk huge, so
`k.shape[2] <= index_topk` always holds and the indexer returns None (dense attn
over the full MTP KV). No weights touched.

    uv run --with-editable . benchmarks/mtp_indexer_probe.py \
        --model ../../glm-5.2/lm-01-mtp --prompt-file mlx_lm/server.py \
        --contexts 4096,16384 --width 64
"""

import argparse

import mlx.core as mx

from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.utils import load


def mtp_indexers(model):
    out = []
    for layer in model.model.mtp.layers:
        idx = getattr(layer.mtp_block.self_attn, "indexer", None)
        if idx is not None:
            out.append(idx)
    return out


def draft_vs_target(model, ids, C, W):
    """Return (acc, n): P(MTP-head argmax == target greedy argmax) over W positions."""
    pre = ids[: C - 1][None]
    walk = ids[C - 1 : C - 1 + W]

    model_cache = make_prompt_cache(model)
    mtp_cache = model.make_mtp_cache()
    _, h = model(pre, cache=model_cache, return_hidden=True)
    model.mtp_forward(h, ids[1:C][None], mtp_cache)
    mx.eval([c.state for c in model_cache + mtp_cache])

    agree = 0
    n = 0
    prev_h = None
    for i in range(W):
        tok = walk[i : i + 1]
        logits, h = model(tok[None], cache=model_cache, return_hidden=True)
        target = mx.argmax(logits[0, -1])
        if prev_h is not None:
            d_logits, _ = model.mtp_forward(prev_h, tok[None], mtp_cache, return_hidden=True)
            draft = mx.argmax(d_logits[0, -1])
            agree += int((draft == target).item())
            n += 1
        prev_h = h
    return agree, n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--prompt-file", required=True)
    p.add_argument("--contexts", default="4096,16384")
    p.add_argument("--width", type=int, default=64)
    args = p.parse_args()

    model, tok = load(args.model)
    if not getattr(model, "has_mtp", False):
        raise SystemExit("model has no MTP layer")
    idxs = mtp_indexers(model)
    if not idxs:
        raise SystemExit("MTP block has no indexer (already shared/dense?) -- nothing to probe")
    real_topk = idxs[0].index_topk
    ids_all = tok.encode(open(args.prompt_file).read())
    contexts = [int(c) for c in args.contexts.split(",") if c]
    W = args.width
    print(f"loaded {args.model}  doc tokens={len(ids_all)}  width={W}")
    print(f"MTP indexers: {len(idxs)}  index_topk={real_topk}\n")

    print(f"{'context':>7} {'sparse_acc':>11} {'dense_acc':>10} {'delta':>8}")
    for C in contexts:
        need = (C - 1) + W + 1
        if len(ids_all) < need:
            print(f"{C:>7}  SKIP (need {need} tokens, have {len(ids_all)})")
            continue
        ids = mx.array(ids_all[:need])

        # sparse: real indexer
        for ix in idxs:
            ix.index_topk = real_topk
        a_s, n = draft_vs_target(model, ids, C, W)

        # dense: indexer returns None (no topk drop) -> head sees full MTP KV
        for ix in idxs:
            ix.index_topk = 1 << 30
        a_d, _ = draft_vs_target(model, ids, C, W)
        for ix in idxs:
            ix.index_topk = real_topk  # restore

        acc_s, acc_d = a_s / n, a_d / n
        print(f"{C:>7} {acc_s:>11.1%} {acc_d:>10.1%} {acc_d - acc_s:>+8.1%}")

    print(
        "\ndense >> sparse at long context => MTP indexer is dropping needed context\n"
        "                                   (fixable bug; envelope may extend).\n"
        "dense ~= sparse                 => head capacity limit; sparsity innocent; stop."
    )


if __name__ == "__main__":
    main()
