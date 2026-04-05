import argparse
from collections import defaultdict
import glob
import importlib.util
import inspect
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, MutableMapping, Tuple, Type

import mlx.core as mx
import mlx.nn as nn

from .utils import _download, _get_classes, load_config

WEIGHT_SUFFIXES = (".weight", ".bias", ".scales", ".biases")


def add_alias(
    aliases: MutableMapping[str, List[str]] | None,
    source: str,
    *targets: str,
) -> None:
    if aliases is None or not targets:
        return
    bucket = aliases.setdefault(source, [])
    for target in targets:
        if target not in bucket:
            bucket.append(target)


def infer_hf_weight_aliases(
    original_weights: Dict[str, Any],
    sanitized_weights: Dict[str, Any],
    explicit_aliases: MutableMapping[str, List[str]] | None = None,
) -> Dict[str, List[str]]:
    aliases = {}

    for key in original_weights.keys() & sanitized_weights.keys():
        add_alias(aliases, key, key)

    sanitized_by_id = defaultdict(list)
    for key, value in sanitized_weights.items():
        sanitized_by_id[id(value)].append(key)

    for key, value in original_weights.items():
        add_alias(aliases, key, *sanitized_by_id.get(id(value), ()))

    for source, targets in (explicit_aliases or {}).items():
        add_alias(aliases, source, *targets)

    return aliases


def module_path(path: str) -> str:
    for suffix in WEIGHT_SUFFIXES:
        if path.endswith(suffix):
            return path[: -len(suffix)]
    return path


def _shape(value: Any) -> List[int]:
    return [int(dim) for dim in getattr(value, "shape", ())]


def _dtype(value: Any) -> str:
    return str(getattr(value, "dtype", "unknown"))


def build_name_report(
    original_weights: Dict[str, Any],
    sanitized_weights: Dict[str, Any],
    explicit_aliases: MutableMapping[str, List[str]] | None = None,
) -> Dict[str, Any]:
    aliases = infer_hf_weight_aliases(
        original_weights,
        sanitized_weights,
        explicit_aliases=explicit_aliases,
    )

    used_targets = set()
    source_weights = []
    for source_key in sorted(original_weights):
        target_keys = aliases.get(source_key, [])
        used_targets.update(target_keys)
        if not target_keys:
            kind = "unmapped"
        elif len(target_keys) == 1 and target_keys[0] == source_key:
            kind = "unchanged"
        elif len(target_keys) == 1:
            kind = "renamed"
        else:
            kind = "aliased"

        source_weights.append(
            {
                "kind": kind,
                "source_dtype": _dtype(original_weights[source_key]),
                "source_key": source_key,
                "source_module": module_path(source_key),
                "source_shape": _shape(original_weights[source_key]),
                "target_keys": target_keys,
                "target_modules": [module_path(key) for key in target_keys],
            }
        )

    created_targets = []
    for target_key in sorted(sanitized_weights):
        if target_key in used_targets:
            continue
        created_targets.append(
            {
                "target_dtype": _dtype(sanitized_weights[target_key]),
                "target_key": target_key,
                "target_module": module_path(target_key),
                "target_shape": _shape(sanitized_weights[target_key]),
            }
        )

    return {
        "summary": {
            "created_targets": len(created_targets),
            "mapped_sources": sum(1 for row in source_weights if row["target_keys"]),
            "source_weights": len(source_weights),
            "target_weights": len(sanitized_weights),
            "unmapped_sources": sum(
                1 for row in source_weights if not row["target_keys"]
            ),
        },
        "created_targets": created_targets,
        "source_weights": source_weights,
    }


def _load_model_classes(model_path: Path, config: dict) -> Tuple[Type[nn.Module], Type]:
    if (model_file := config.get("model_file")) is not None:
        spec = importlib.util.spec_from_file_location(
            "custom_model",
            model_path / model_file,
        )
        arch = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(arch)
        return arch.Model, arch.ModelArgs
    return _get_classes(config=config)


def _sanitize_metadata(model: nn.Module) -> Dict[str, Any]:
    if not hasattr(model, "sanitize"):
        return {"defined": False}

    method = model.sanitize
    try:
        file_path = inspect.getsourcefile(method)
        _, line = inspect.getsourcelines(method)
    except (OSError, TypeError):
        file_path, line = None, None
    return {
        "defined": True,
        "file": file_path,
        "line": line,
    }


def build_repo_name_report(
    path_or_hf_repo: str,
    revision: str = None,
) -> Dict[str, Any]:
    model_path = _download(path_or_hf_repo, revision=revision)
    config = load_config(model_path)

    weights = {}
    for weight_file in glob.glob(str(model_path / "model*.safetensors")):
        weights.update(mx.load(weight_file))
    if not weights:
        raise FileNotFoundError(f"No safetensors found in {model_path}")
    raw_weights = weights.copy()

    if "quantization_config" not in config:
        text_config = config.get("text_config", {})
        if "quantization_config" in text_config:
            config["quantization_config"] = text_config["quantization_config"]

    model_class, model_args_class = _load_model_classes(model_path, config)
    model_args = model_args_class.from_dict(config)
    model = model_class(model_args)
    sanitize = _sanitize_metadata(model)
    if hasattr(model, "sanitize"):
        sanitized_weights = model.sanitize(weights)
        if sanitized_weights is None:
            sanitized_weights = weights
    else:
        sanitized_weights = weights

    report = build_name_report(raw_weights, sanitized_weights)
    report["model"] = {
        "class": f"{model.__class__.__module__}.{model.__class__.__name__}",
        "model_type": config["model_type"],
        "path": str(model_path),
        "repo": path_or_hf_repo,
        "revision": revision,
    }
    report["sanitize"] = sanitize
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Print a best-effort map from source checkpoint weight names to "
            "their MLX sanitized names."
        )
    )
    parser.add_argument("model", help="Local model path or Hugging Face repo ID.")
    parser.add_argument(
        "--revision",
        help="Optional Hugging Face revision to inspect.",
        default=None,
    )
    args = parser.parse_args()

    report = build_repo_name_report(args.model, revision=args.revision)
    json.dump(report, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
