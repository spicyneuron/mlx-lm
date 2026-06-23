#!/usr/bin/env python3
"""Separate inherent MTP-head degradation from the mask-path acceptance penalty.

At each of W in-distribution positions (same trajectory), compute three argmaxes
for the *same* next token:
  - draft        : the MTP head's proposal (model.mtp_forward), as in real drafting
  - gather verify: the target's argmax via the L==1 gather path (== normal decode)
  - mask verify  : the target's argmax via the L>1 absorbed mask path (Phase 1)

Reports, per context:
  - acc_gather = P(draft == gather verify)  -> inherent head quality / acceptance
                 ceiling with a faithful verify
  - acc_mask   = P(draft == mask verify)    -> what Phase-1 verify actually accepts
  - penalty    = acc_gather - acc_mask      -> the mask-path acceptance tax
  - vg==vm     = P(gather == mask)          -> verify-path argmax agreement at the
                 positions that actually matter for acceptance

Efficiency: one prefill per context. The gather walk advances the cache; we trim
it back by W and reuse it for the batched mask verify (no second prefill).

    uv run --with-editable . benchmarks/mtp_phase1_accept.py \
        --model ../../glm-5.2/lm-01 --prompt-file mlx_lm/server.py --contexts 512,4096
"""

import argparse

import mlx.core as mx

from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache
from mlx_lm.utils import load


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--prompt-file", required=True)
    p.add_argument("--contexts", default="512,4096")
    p.add_argument("--width", type=int, default=32, help="positions to evaluate")
    args = p.parse_args()

    model, tok = load(args.model)
    if not getattr(model, "has_mtp", False):
        raise SystemExit("model has no MTP layer; cannot measure draft acceptance")
    ids_all = tok.encode(open(args.prompt_file).read())
    contexts = [int(c) for c in args.contexts.split(",") if c]
    W = args.width
    print(f"loaded {args.model}  doc tokens={len(ids_all)}  width={W}")
    print(f"index_topk threshold for sparsity: {model.args.index_topk}\n")

    print(f"{'context':>7} {'acc_gather':>11} {'acc_mask':>9} {'penalty':>8} {'vg==vm':>7}")
    for C in contexts:
        need = (C - 1) + W + 1
        if len(ids_all) < need:
            print(f"{C:>7}  SKIP (need {need} tokens, have {len(ids_all)})")
            continue
        ids = mx.array(ids_all[:need])
        pre = ids[: C - 1][None]  # prime positions 0..C-2
        walk = ids[C - 1 : C - 1 + W]  # W tokens at positions C-1 .. C-2+W

        model_cache = make_prompt_cache(model)
        mtp_cache = model.make_mtp_cache()

        # prime model_cache + mtp_cache over the context (mirrors generate._prefill)
        _, h = model(pre, cache=model_cache, return_hidden=True)
        model.mtp_forward(h, ids[1:C][None], mtp_cache)
        mx.eval([c.state for c in model_cache + mtp_cache])

        # gather walk: one token at a time -> gather verify argmax + draft argmax
        gather_pred, draft_pred = [], []
        for i in range(W):
            tok = walk[i : i + 1]
            logits, h = model(tok[None], cache=model_cache, return_hidden=True)
            gather_pred.append(mx.argmax(logits[0, -1]))
            d_logits, _ = model.mtp_forward(h, tok[None], mtp_cache, return_hidden=True)
            draft_pred.append(mx.argmax(d_logits[0, -1]))
        gather_pred = mx.stack(gather_pred)
        draft_pred = mx.stack(draft_pred)

        # reuse the cache: trim the W walk steps back off, then batched mask verify
        trim_prompt_cache(model_cache, W)
        mask_logits = model(walk[None], cache=model_cache)
        mask_pred = mx.argmax(mask_logits[0], axis=-1)
        mx.eval(gather_pred, draft_pred, mask_pred)

        acc_g = (draft_pred == gather_pred).mean().item()
        acc_m = (draft_pred == mask_pred).mean().item()
        vgvm = (gather_pred == mask_pred).mean().item()
        print(f"{C:>7} {acc_g:>11.1%} {acc_m:>9.1%} {acc_g - acc_m:>+8.1%} {vgvm:>7.1%}")

    print(
        "\nacc_gather falling 512->long  => inherent MTP-head degradation (Phase 2 can't fix).\n"
        "penalty large                 => mask path costs acceptance (Phase 2 / gather verify helps).\n"
        "vg==vm < 100% at long context => the gather/mask jitter does flip verify argmax\n"
        "                                 at draft-relevant positions (the penalty's mechanism)."
    )


if __name__ == "__main__":
    main()
