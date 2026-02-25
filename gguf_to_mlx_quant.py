#!/usr/bin/env python3

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Pattern, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map_with_path

from mlx_lm.utils import _download, load, load_config, quantize_model, save


DEFAULT_RECIPE = {
    "tensor_types": [
        "output.weight=Q6_K",
        "token_embd.weight=Q4_K",
        r"blk\.(\d+)\.attn_k_b.weight=Q8_0",
        r"blk\.(\d+)\.attn_kv_a_mqa.weight=Q8_0",
        r"blk\.(\d+)\.attn_output.weight=Q3_K",
        r"blk\.(0|1|2|3|4|7|8|9|10|11|12|13|14|15|16|17|18|19|20|21|22|23|24|25|26|27|28|29|30|31|32|33|34|35|36|37|38|39|40|41|42|43|44|45|46|47|48|49|50|51|52|53|54|55|56|57|58)\.attn_q_a.weight=Q4_K",
        r"blk\.(5|6|59|60)\.attn_q_a.weight=Q5_K",
        r"blk\.(0|2|4)\.attn_q_b.weight=F16",
        r"blk\.(1|3|5|6|8|9|10|12|16)\.attn_q_b.weight=Q8_0",
        r"blk\.(7|11|13|14|15|21|26|32|57|60)\.attn_q_b.weight=Q6_K",
        r"blk\.(17|18|19|20|22|23|24|25|27|28|29|30|31|33|34|35|36|37|38|39|40|41|42|43|44|45|46|47|48|49|50|51|52|53|54|55|56|58)\.attn_q_b.weight=Q4_K",
        r"blk\.(59)\.attn_q_b.weight=Q5_K",
        r"blk\.(\d+)\.attn_v_b.weight=Q8_0",
        r"blk\.(\d+)\.ffn_down.weight=Q3_K",
        r"blk\.(\d+)\.ffn_gate.weight=Q4_K",
        r"blk\.(\d+)\.ffn_up.weight=Q4_K",
        r"blk\.(\d+)\.ffn_down_exps.weight=Q3_K",
        r"blk\.(\d+)\.ffn_down_shexp.weight=Q3_K",
        r"blk\.(\d+)\.ffn_gate_exps.weight=Q2_K",
        r"blk\.(\d+)\.ffn_gate_inp.weight=F32",
        r"blk\.(\d+)\.ffn_gate_shexp.weight=Q4_K",
        r"blk\.(\d+)\.ffn_up_exps.weight=Q2_K",
        r"blk\.(\d+)\.ffn_up_shexp.weight=Q4_K",
    ],
    "default_type": "Q4_K",
}

GGUF_TO_BITS = {
    "Q2_K": 2,
    "Q3_K": 3,
    "Q4_K": 4,
    "Q5_K": 5,
    "Q6_K": 6,
    "Q8_0": 8,
}


def normalize_path(path) -> str:
    if isinstance(path, str):
        return path
    return ".".join(str(p) for p in path)


def mlx_weight_to_gguf_name(weight_name: str) -> str:
    name = weight_name

    if name == "lm_head.weight":
        return "output.weight"
    if name in ("model.embed_tokens.weight", "embed_tokens.weight"):
        return "token_embd.weight"

    name = re.sub(r"^(?:model\.)?layers\.", "blk.", name)

    # Specific patterns first.
    replacements = [
        ("self_attn.kv_a_proj_with_mqa", "attn_kv_a_mqa"),
        ("self_attn.embed_q", "attn_k_b"),
        ("self_attn.unembed_out", "attn_v_b"),
        ("self_attn.q_a_proj", "attn_q_a"),
        ("self_attn.q_b_proj", "attn_q_b"),
        ("self_attn.q_proj", "attn_q"),
        ("self_attn.k_proj", "attn_k"),
        ("self_attn.v_proj", "attn_v"),
        ("self_attn.o_proj", "attn_output"),
        ("mlp.switch_mlp.down_proj", "ffn_down_exps"),
        ("mlp.switch_mlp.gate_proj", "ffn_gate_exps"),
        ("mlp.switch_mlp.up_proj", "ffn_up_exps"),
        ("mlp.shared_experts.down_proj", "ffn_down_shexp"),
        ("mlp.shared_experts.gate_proj", "ffn_gate_shexp"),
        ("mlp.shared_experts.up_proj", "ffn_up_shexp"),
        ("mlp.gate.weight", "ffn_gate_inp.weight"),
        ("mlp.down_proj", "ffn_down"),
        ("mlp.gate_proj", "ffn_gate"),
        ("mlp.up_proj", "ffn_up"),
        ("block_sparse_moe.gate", "ffn_gate_inp"),
        ("model.embed_tokens", "token_embd"),
        ("embed_tokens", "token_embd"),
        ("lm_head", "output"),
    ]
    for src, dst in replacements:
        name = name.replace(src, dst)

    return name


def gguf_type_to_bits(gguf_type: str) -> Optional[int]:
    gguf_type = gguf_type.upper()
    if gguf_type in ("F16", "F32"):
        return None
    if gguf_type in GGUF_TO_BITS:
        return GGUF_TO_BITS[gguf_type]

    m = re.match(r"^Q(\d+)", gguf_type)
    if m is None:
        raise ValueError(f"Unsupported GGUF type: {gguf_type}")
    bits = int(m.group(1))
    if bits < 2 or bits > 8:
        raise ValueError(f"Unsupported GGUF quant bits: {gguf_type}")
    return bits


def parse_rules(entries: List[str]) -> List[Tuple[Pattern[str], str, str]]:
    rules = []
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"Invalid tensor_types entry: {entry}")
        pattern, gguf_type = entry.split("=", 1)
        pattern = pattern.strip()
        gguf_type = gguf_type.strip().upper()
        try:
            compiled = re.compile(pattern)
        except re.error as err:
            raise ValueError(f"Invalid regex in tensor_types entry '{entry}': {err}")
        rules.append((compiled, gguf_type, pattern))
    return rules


def load_recipe_arg(recipe_arg: Optional[str]) -> Dict:
    if recipe_arg is None:
        return DEFAULT_RECIPE

    if recipe_arg == "-":
        return json.load(sys.stdin)

    recipe_path = Path(recipe_arg)
    if recipe_path.exists():
        with recipe_path.open("r") as f:
            return json.load(f)

    return json.loads(recipe_arg)


class Matcher:
    def __init__(self, recipe: Dict):
        if "tensor_types" not in recipe or "default_type" not in recipe:
            raise ValueError("Recipe must have 'tensor_types' and 'default_type'.")
        self.rules = parse_rules(recipe["tensor_types"])
        self.default_type = recipe["default_type"].upper()
        _ = gguf_type_to_bits(self.default_type)

    def match(self, gguf_name: str) -> Tuple[str, Optional[str]]:
        for pattern, gguf_type, raw in self.rules:
            if pattern.fullmatch(gguf_name):
                return gguf_type, raw
        for pattern, gguf_type, raw in self.rules:
            if pattern.search(gguf_name):
                return gguf_type, raw
        return self.default_type, None


def ensure_remote_code_allowed(
    model_ref: str,
    revision: Optional[str],
    trust_remote_code: bool,
) -> None:
    model_path = _download(
        model_ref,
        revision=revision,
        allow_patterns=["config.json", "generation_config.json"],
    )
    config = load_config(model_path)
    model_file = config.get("model_file")
    if model_file is not None and not trust_remote_code:
        raise ValueError(
            f"Model requires custom code via config.json:model_file={model_file!r}. "
            "Re-run with --trust-remote-code to allow loading it."
        )


def apply_dtype_cast(model: nn.Module, dtype_name: str) -> None:
    dtype = getattr(mx, dtype_name)
    cast_predicate = getattr(model, "cast_predicate", lambda _: True)

    def set_dtype(path, value):
        path = normalize_path(path)
        if cast_predicate(path) and mx.issubdtype(value.dtype, mx.floating):
            return value.astype(dtype)
        return value

    model.update(tree_map_with_path(set_dtype, model.parameters()))


def apply_float_overrides(model: nn.Module, matcher: Matcher) -> Counter:
    cast_counts = Counter()

    def maybe_cast(path, value):
        path = normalize_path(path)
        if not path.endswith(".weight"):
            return value
        if not mx.issubdtype(value.dtype, mx.floating):
            return value
        gguf_name = mlx_weight_to_gguf_name(path)
        gguf_type, _ = matcher.match(gguf_name)
        if gguf_type == "F32" and value.dtype != mx.float32:
            cast_counts["F32"] += 1
            return value.astype(mx.float32)
        if gguf_type == "F16" and value.dtype != mx.float16:
            cast_counts["F16"] += 1
            return value.astype(mx.float16)
        return value

    model.update(tree_map_with_path(maybe_cast, model.parameters()))
    return cast_counts


def main():
    parser = argparse.ArgumentParser(
        description="Approximate GGUF tensor-type quantization rules in MLX."
    )
    parser.add_argument(
        "--model",
        "--hf-path",
        required=True,
        help="Hugging Face model id or local model path.",
    )
    parser.add_argument(
        "--mlx-path",
        default="mlx_model",
        help="Where to save the quantized MLX model.",
    )
    parser.add_argument(
        "--recipe",
        default=None,
        help=(
            "Recipe source. Omit to use built-in default; pass path to JSON file; "
            "pass '-' to read JSON from stdin; or pass inline JSON."
        ),
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=64,
        help="Group size for affine quantization.",
    )
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        default=None,
        help="Optional cast for all floating model weights before quantization.",
    )
    parser.add_argument("--revision", default=None)
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow loading custom model/tokenizer code from the model repo.",
    )
    args = parser.parse_args()

    mlx_path = Path(args.mlx_path)
    if mlx_path.exists():
        raise ValueError(
            f"Cannot save to {mlx_path} because it already exists. "
            "Pick a new output path."
        )

    recipe = load_recipe_arg(args.recipe)
    matcher = Matcher(recipe)
    print(f"[INFO] Loaded recipe with {len(matcher.rules)} rules.")

    ensure_remote_code_allowed(args.model, args.revision, args.trust_remote_code)

    print("[INFO] Loading model")
    model, tokenizer, config = load(
        args.model,
        revision=args.revision,
        return_config=True,
        tokenizer_config={"trust_remote_code": args.trust_remote_code},
        lazy=True,
    )

    if args.dtype is not None:
        print(f"[INFO] Casting model weights to {args.dtype}")
        apply_dtype_cast(model, args.dtype)

    float_casts = apply_float_overrides(model, matcher)
    if float_casts:
        for dtype_name, n in sorted(float_casts.items()):
            print(f"[INFO] Applied {dtype_name} overrides to {n} tensor(s).")

    assignment_counts = Counter()
    matched_rule_counts = Counter()

    def quant_predicate(path: str, _module: nn.Module):
        weight_path = f"{path}.weight"
        gguf_name = mlx_weight_to_gguf_name(weight_path)
        gguf_type, matched_rule = matcher.match(gguf_name)

        assignment_counts[gguf_type] += 1
        if matched_rule is not None:
            matched_rule_counts[matched_rule] += 1

        bits = gguf_type_to_bits(gguf_type)
        if bits is None:
            return False
        return {"group_size": args.group_size, "bits": bits, "mode": "affine"}

    default_bits = gguf_type_to_bits(matcher.default_type) or 4
    print("[INFO] Quantizing")
    model, config = quantize_model(
        model,
        config,
        group_size=args.group_size,
        bits=default_bits,
        mode="affine",
        quant_predicate=quant_predicate,
    )

    print("[INFO] Quantization assignment summary:")
    for gguf_type, n in sorted(assignment_counts.items()):
        print(f"  {gguf_type}: {n}")
    if matched_rule_counts:
        print("[INFO] Top matched rules:")
        for rule, n in matched_rule_counts.most_common(10):
            print(f"  {rule} -> {n}")

    save(args.mlx_path, args.model, model, tokenizer, config)
    print(f"[INFO] Saved quantized model to {args.mlx_path}")


if __name__ == "__main__":
    main()
