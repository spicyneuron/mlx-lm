import json

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_lm.utils import (
    adapt_weight_structure,
    canonicalize_weight_keys,
    ensure_matching_weights,
    format_weight_compatibility_error,
    inspect_weight_compatibility,
    load_model,
)


def test_canonicalize_weight_keys_remaps_common_aliases():
    weights = {
        "model.language_model.visual.blocks.0.weight": mx.array([1]),
        "language_model.model.visual.blocks.1.weight": mx.array([2]),
        "model.visual.blocks.2.weight": mx.array([3]),
        "model.language_model.layers.0.weight": mx.array([4]),
        "lm_head.weight": mx.array([5]),
    }
    expected_keys = {
        "vision_tower.blocks.0.weight",
        "vision_tower.blocks.1.weight",
        "vision_tower.blocks.2.weight",
        "language_model.model.layers.0.weight",
        "language_model.lm_head.weight",
    }

    canonicalized = canonicalize_weight_keys(weights, expected_keys)

    assert set(canonicalized) == expected_keys


def test_canonicalize_weight_keys_keeps_existing_targets_and_unknown_aliases():
    weights = {
        "lm_head.weight": mx.array([1]),
        "language_model.lm_head.weight": mx.array([2]),
        "model.language_model.visual.blocks.0.weight": mx.array([3]),
    }
    expected_keys = {"language_model.lm_head.weight"}

    canonicalized = canonicalize_weight_keys(weights, expected_keys)

    assert "language_model.lm_head.weight" in canonicalized
    assert "lm_head.weight" in canonicalized
    assert "model.language_model.visual.blocks.0.weight" in canonicalized


def test_adapt_weight_structure_splits_in_proj_weights_and_biases():
    weights = {
        "resampler.attn.in_proj_weight": mx.zeros((12, 4)),
        "resampler.attn.in_proj_bias": mx.zeros((12,)),
    }
    expected_keys = {
        "resampler.attn.q_proj.weight",
        "resampler.attn.k_proj.weight",
        "resampler.attn.v_proj.weight",
        "resampler.attn.q_proj.bias",
        "resampler.attn.k_proj.bias",
        "resampler.attn.v_proj.bias",
    }

    adapted = adapt_weight_structure(weights, expected_keys)

    assert "resampler.attn.in_proj_weight" not in adapted
    assert "resampler.attn.in_proj_bias" not in adapted
    assert adapted["resampler.attn.q_proj.weight"].shape == (4, 4)
    assert adapted["resampler.attn.k_proj.weight"].shape == (4, 4)
    assert adapted["resampler.attn.v_proj.weight"].shape == (4, 4)
    assert adapted["resampler.attn.q_proj.bias"].shape == (4,)
    assert adapted["resampler.attn.k_proj.bias"].shape == (4,)
    assert adapted["resampler.attn.v_proj.bias"].shape == (4,)


def test_adapt_weight_structure_splits_switch_mlp_gate_up_and_down_proj():
    weights = {
        "language_model.model.layers.0.mlp.experts.gate_up_proj": mx.zeros((2, 8, 4)),
        "language_model.model.layers.0.mlp.experts.down_proj": mx.zeros((2, 4, 8)),
    }
    expected_keys = {
        "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight",
        "language_model.model.layers.0.mlp.switch_mlp.up_proj.weight",
        "language_model.model.layers.0.mlp.switch_mlp.down_proj.weight",
    }

    adapted = adapt_weight_structure(weights, expected_keys)

    assert "language_model.model.layers.0.mlp.experts.gate_up_proj" not in adapted
    assert "language_model.model.layers.0.mlp.experts.down_proj" not in adapted
    assert (
        adapted["language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight"].shape
        == (2, 4, 4)
    )
    assert (
        adapted["language_model.model.layers.0.mlp.switch_mlp.up_proj.weight"].shape
        == (2, 4, 4)
    )
    assert (
        adapted["language_model.model.layers.0.mlp.switch_mlp.down_proj.weight"].shape
        == (2, 4, 8)
    )


def test_adapt_weight_structure_keeps_existing_targets():
    weights = {
        "resampler.attn.in_proj_weight": mx.zeros((12, 4)),
        "resampler.attn.q_proj.weight": mx.ones((4, 4)),
    }
    expected_keys = {
        "resampler.attn.q_proj.weight",
        "resampler.attn.k_proj.weight",
        "resampler.attn.v_proj.weight",
    }

    adapted = adapt_weight_structure(weights, expected_keys)

    assert "resampler.attn.in_proj_weight" in adapted
    assert adapted["resampler.attn.q_proj.weight"].shape == (4, 4)


def test_inspect_weight_compatibility_finds_missing_unexpected_and_shape_mismatches():
    expected_weights = {
        "vision_tower.weight": mx.zeros((4, 4)),
        "language_model.lm_head.weight": mx.zeros((4, 4)),
        "language_model.model.layers.0.weight": mx.zeros((4, 4)),
    }
    weights = {
        "vision_tower.weight": mx.zeros((8, 4)),
        "lm_head.weight": mx.zeros((4, 4)),
        "extra.weight": mx.zeros((2, 2)),
    }

    issues = inspect_weight_compatibility(expected_weights, weights)

    assert issues["missing"] == [
        "language_model.lm_head.weight",
        "language_model.model.layers.0.weight",
    ]
    assert issues["unexpected"] == ["extra.weight", "lm_head.weight"]
    assert issues["shape_mismatches"] == [
        ("vision_tower.weight", (4, 4), (8, 4))
    ]
    assert issues["suggestions"] == [
        ("lm_head.weight", "language_model.lm_head.weight")
    ]


def test_format_weight_compatibility_error_includes_actionable_sections():
    issues = {
        "expected_count": 3,
        "provided_count": 2,
        "missing": ["language_model.lm_head.weight"],
        "unexpected": ["lm_head.weight"],
        "shape_mismatches": [("vision_tower.weight", (4, 4), (8, 4))],
        "suggestions": [("lm_head.weight", "language_model.lm_head.weight")],
    }

    message = format_weight_compatibility_error(issues)

    assert "Checkpoint weights do not match the instantiated model" in message
    assert "Missing keys:" in message
    assert "Unexpected keys:" in message
    assert "Shape mismatches:" in message
    assert "Possible remaps:" in message
    assert "lm_head.weight -> language_model.lm_head.weight" in message


def test_ensure_matching_weights_raises_helpful_error():
    expected_weights = {"language_model.lm_head.weight": mx.zeros((4, 4))}
    weights = {"lm_head.weight": mx.zeros((4, 4))}

    with pytest.raises(ValueError) as exc_info:
        ensure_matching_weights(expected_weights, weights)

    message = str(exc_info.value)
    assert "Missing: 1, unexpected: 1, shape mismatches: 0." in message
    assert "lm_head.weight -> language_model.lm_head.weight" in message


def test_load_model_adapts_common_exported_weights(tmp_path):
    class Proj(nn.Module):
        def __init__(self, bias=False):
            super().__init__()
            self.weight = mx.zeros((4, 4))
            if bias:
                self.bias = mx.zeros((4,))

    class Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = Proj(bias=True)
            self.k_proj = Proj(bias=True)
            self.v_proj = Proj(bias=True)

    class Resampler(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = Attn()

    class LanguageModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lm_head = Proj()

    class DummyModel(nn.Module):
        def __init__(self, args):
            super().__init__()
            self.args = args
            self.resampler = Resampler()
            self.language_model = LanguageModel()

    class DummyArgs:
        @classmethod
        def from_dict(cls, config):
            return cls()

    weights = {
        "resampler.attn.in_proj_weight": mx.arange(48, dtype=mx.float32).reshape(12, 4),
        "resampler.attn.in_proj_bias": mx.arange(12, dtype=mx.float32),
        "lm_head.weight": mx.arange(16, dtype=mx.float32).reshape(4, 4),
    }

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "dummy"}))
    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)

    model, _ = load_model(
        tmp_path,
        get_model_classes=lambda config: (DummyModel, DummyArgs),
    )

    assert mx.array_equal(model.resampler.attn.q_proj.weight, weights["resampler.attn.in_proj_weight"][:4])
    assert mx.array_equal(model.resampler.attn.k_proj.weight, weights["resampler.attn.in_proj_weight"][4:8])
    assert mx.array_equal(model.resampler.attn.v_proj.weight, weights["resampler.attn.in_proj_weight"][8:])
    assert mx.array_equal(model.resampler.attn.q_proj.bias, weights["resampler.attn.in_proj_bias"][:4])
    assert mx.array_equal(model.resampler.attn.k_proj.bias, weights["resampler.attn.in_proj_bias"][4:8])
    assert mx.array_equal(model.resampler.attn.v_proj.bias, weights["resampler.attn.in_proj_bias"][8:])
    assert mx.array_equal(model.language_model.lm_head.weight, weights["lm_head.weight"])
