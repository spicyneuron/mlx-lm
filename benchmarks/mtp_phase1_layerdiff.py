#!/usr/bin/env python3
"""Find the first layer where batched verify diverges from sequential decode.

verify==decode is exact below the sparsity threshold but diverges above it, and
the divergence survives swapping the fused SDPA for an explicit softmax -- so it
is not the attention kernel. This captures, per decoder layer, the attention
output and the layer output for a batched L-wide verify vs the same tokens
decoded one at a time, and reports the first layer (and sublayer) that differs.

    uv run --with-editable . benchmarks/mtp_phase1_layerdiff.py \
        --model ../../glm-5.2/lm-01 --context 4096 --width 2
"""

import argparse

import mlx.core as mx

from mlx_lm.models.glm_moe_dsa import GlmMoeDsaAttention, GlmMoeDsaDecoderLayer
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.utils import load


def capture_during(measured):
    """Run measured() recording every decoder-layer and attention output."""
    o_layer = GlmMoeDsaDecoderLayer.__call__
    o_attn = GlmMoeDsaAttention.__call__
    layer_out, attn_out = [], []

    def p_layer(self, *a, **k):
        out = o_layer(self, *a, **k)
        layer_out.append(out[0])
        return out

    def p_attn(self, *a, **k):
        out = o_attn(self, *a, **k)
        attn_out.append((getattr(self, "skip_topk", False), out[0]))
        return out

    GlmMoeDsaDecoderLayer.__call__ = p_layer
    GlmMoeDsaAttention.__call__ = p_attn
    try:
        measured()
    finally:
        GlmMoeDsaDecoderLayer.__call__ = o_layer
        GlmMoeDsaAttention.__call__ = o_attn
    mx.eval(layer_out, [a for _, a in attn_out])
    return layer_out, attn_out


def rel(a, b):
    m = mx.max(mx.abs(a - b)).item()
    return m, m / max(mx.max(mx.abs(b)).item(), 1e-6)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--context", type=int, default=4096)
    p.add_argument("--width", type=int, default=2)
    args = p.parse_args()

    model, _ = load(args.model)
    vocab = model.model.embed_tokens.weight.shape[0]
    print(f"loaded {args.model}  vocab={vocab}  context={args.context} width={args.width}\n")

    mx.random.seed(0)
    ids = mx.random.randint(0, vocab, (1, args.context + args.width))
    ctx, step = ids[:, : args.context], ids[:, args.context :]

    def prefill(cache):
        model(ctx, cache=cache)
        mx.eval([c.state for c in cache])

    # batched verify: one forward of the L tokens
    cache_v = make_prompt_cache(model)
    prefill(cache_v)
    v_layers, v_attn = capture_during(lambda: mx.eval(model(step, cache=cache_v)))

    # sequential decode: one forward per token, on a shared cache
    cache_s = make_prompt_cache(model)
    prefill(cache_s)
    seq_layers, seq_attn = [], []
    for i in range(args.width):
        sl, sa = capture_during(lambda: mx.eval(model(step[:, i : i + 1], cache=cache_s)))
        seq_layers.append(sl)
        seq_attn.append(sa)

    n = len(v_layers)
    print(f"{n} decoder layers; comparing per position\n")
    print(f"{'layer':>5} {'shared':>6} {'attn_rel':>10} {'layer_rel':>10}")
    first = None
    for l in range(n):
        # stack the per-step outputs back into [b, width, ...] to match verify
        dec_layer = mx.concatenate([seq_layers[i][l] for i in range(args.width)], axis=1)
        dec_attn = mx.concatenate([seq_attn[i][l][1] for i in range(args.width)], axis=1)
        shared = v_attn[l][0]
        _, a_rel = rel(v_attn[l][1], dec_attn)
        _, h_rel = rel(v_layers[l], dec_layer)
        flag = ""
        if first is None and (a_rel > 1e-2 or h_rel > 1e-2):
            first = l
            flag = "  <-- first divergence"
        print(f"{l:>5} {str(shared):>6} {a_rel:>10.2e} {h_rel:>10.2e}{flag}")

    print()
    if first is None:
        print("no layer diverged above 1e-2 (mismatch must be in lm_head/norm)")
    else:
        sublayer = "ATTENTION" if rel(v_attn[first][1], mx.concatenate(
            [seq_attn[i][first][1] for i in range(args.width)], axis=1))[1] > 1e-2 else "MLP"
        print(f"first divergence at layer {first} (shared={v_attn[first][0]}), sublayer={sublayer}")


if __name__ == "__main__":
    main()
