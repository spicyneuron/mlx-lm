# Copyright © 2026 Apple Inc.

"""Shared MoE primitives for sigmoid-gated grouped routing (DSV3 / GLM-MoE).

`group_expert_select` is the common `topk_method="noaux_tc"` selection used by
deepseek_v3, deepseek_v32, glm4_moe, and glm4_moe_lite. Consolidating it here
lets MLX share the `@mx.compile` cache across files and lets us fold the gate
matmul inside the compiled region, where it can fuse with the sigmoid /
bias-add / selection epilogue.
"""

import mlx.core as mx


@mx.compile
def group_expert_select(
    x,
    gate_weight,
    e_score_correction_bias,
    top_k,
    n_group,
    topk_group,
    routed_scaling_factor,
    norm_topk_prob,
    denom_eps,
):
    gates = x @ gate_weight.T
    scores = mx.sigmoid(gates.astype(mx.float32))
    orig_scores = scores
    scores = scores + e_score_correction_bias
    if n_group > 1:
        scores = mx.unflatten(scores, axis=-1, shape=(n_group, -1))
        group_scores = mx.topk(scores, 2, axis=-1).sum(axis=-1, keepdims=True)
        k = n_group - topk_group
        group_idx = mx.argpartition(group_scores, kth=k - 1, axis=-2)[..., :k, :]
        scores = mx.put_along_axis(
            scores, mx.stop_gradient(group_idx), mx.array(0.0), axis=-2
        )
        scores = mx.flatten(scores, -2, -1)

    k = top_k
    # Partition such that the largest k land at the tail, avoiding the
    # full negated-copy that `argpartition(-scores, k-1)[..., :k]` requires.
    inds = mx.argpartition(scores, kth=-k, axis=-1)[..., -k:]
    scores = mx.take_along_axis(orig_scores, inds, axis=-1)
    if top_k > 1 and norm_topk_prob:
        denominator = scores.sum(axis=-1, keepdims=True)
        scores = scores / (denominator + denom_eps)
    scores = scores * routed_scaling_factor

    return inds, scores
