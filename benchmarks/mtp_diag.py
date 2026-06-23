"""Diagnose weight-load mismatches for an MTP/GLM checkpoint.

Loads the model non-strict (so it won't raise), then diffs the parameters the
instantiated model expects against the keys actually present on disk. Keys are
collapsed by layer index (`.<n>.` -> `.N.`) so the output is a short summary
instead of thousands of per-expert lines.

Usage:
    uv run python benchmarks/mtp_diag.py /path/to/model
"""

import glob
import os
import re
import sys
from collections import Counter
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

from mlx_lm.utils import load_model

COLLAPSE = re.compile(r"\.\d+\.")


def summarize(keys):
    counts = Counter(COLLAPSE.sub(".N.", k) for k in keys)
    return counts.most_common()


def main():
    path = Path(sys.argv[1])
    model, config = load_model(path, lazy=True, strict=False)

    expected = {k for k, _ in tree_flatten(model.parameters())}

    ondisk = set()
    for f in glob.glob(str(path / "model*.safetensors")):
        ondisk |= set(mx.load(f).keys())

    missing = expected - ondisk
    extra = ondisk - expected

    print(f"model_type={config.get('model_type')} "
          f"num_hidden_layers={config.get('num_hidden_layers')} "
          f"num_nextn_predict_layers={config.get('num_nextn_predict_layers')} "
          f"quantization={config.get('quantization') is not None}")
    print(f"expected={len(expected)} ondisk={len(ondisk)} "
          f"missing={len(missing)} extra={len(extra)}")

    print(f"\n--- MISSING (model wants, checkpoint lacks): {len(missing)} ---")
    for pat, n in summarize(missing):
        print(f"  {n:5d}  {pat}")

    print(f"\n--- EXTRA (checkpoint has, model dropped): {len(extra)} ---")
    for pat, n in summarize(extra):
        print(f"  {n:5d}  {pat}")


if __name__ == "__main__":
    main()
