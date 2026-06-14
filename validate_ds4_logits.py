#!/usr/bin/env python3
"""
One-off DeepSeek V4 base-vs-experiment logits validator.

Runs each mlx-lm git ref in an isolated uvx environment, saves final logits for
fixed deterministic prompts, then compares exact equality and numeric deltas.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

BASE_SPEC = "mlx-lm@git+https://github.com/spicyneuron/mlx-lm@_ds4"
OURS_SPEC = "mlx-lm@git+https://github.com/spicyneuron/mlx-lm@_ds4_perf"
DEFAULT_MODEL = "mlx-community/DeepSeek-V4-Flash-8bit"
JSON_START = "===DS4_VALIDATE_JSON_START==="
JSON_END = "===DS4_VALIDATE_JSON_END==="

DRIVER = r'''
import argparse
import hashlib
import json
import sys
import time

import mlx.core as mx
import numpy as np
from mlx_lm.generate import stream_generate
from mlx_lm.models import cache as cache_lib
from mlx_lm.utils import load

JSON_START = "===DS4_VALIDATE_JSON_START==="
JSON_END = "===DS4_VALIDATE_JSON_END==="


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def tokenize(tokenizer, text):
    return tokenizer.encode(text, add_special_tokens=False)


def build_prompt(tokenizer, target_tokens):
    text = (
        "DeepSeek V4 logits validation prompt. "
        "This text is repeated to create deterministic tokens. "
    )
    base = tokenize(tokenizer, text) or [0]
    reps = (target_tokens + len(base) - 1) // len(base)
    return (base * reps)[:target_tokens]


def final_logits(model, tokens):
    x = mx.array(tokens, dtype=mx.int32)[None]
    logits = model(x)[:, -1, :].astype(mx.float32)
    mx.eval(logits)
    return np.array(logits[0])


def cached_final_logits(model, tokens, prefill_step_size):
    x = mx.array(tokens, dtype=mx.int32)
    prompt_cache = cache_lib.make_prompt_cache(model)
    pos = 0
    while len(tokens) - pos > 1:
        n = min(prefill_step_size, len(tokens) - pos - 1)
        model(x[pos : pos + n][None], cache=prompt_cache)
        mx.eval([c.state for c in prompt_cache])
        pos += n
        mx.clear_cache()
    logits = model(x[pos:][None], cache=prompt_cache)[:, -1, :].astype(mx.float32)
    mx.eval(logits)
    return np.array(logits[0])


def greedy_tokens(model, tokenizer, tokens, max_tokens, prefill_step_size):
    generated = []
    for resp in stream_generate(
        model,
        tokenizer,
        tokens,
        max_tokens=max_tokens,
        prefill_step_size=prefill_step_size,
    ):
        generated.append(int(resp.token))
    mx.eval()
    return generated


def greedy_trace(model, tokens, max_tokens, prefill_step_size):
    x = mx.array(tokens, dtype=mx.int32)
    prompt_cache = cache_lib.make_prompt_cache(model)
    pos = 0
    while len(tokens) - pos > 1:
        n = min(prefill_step_size, len(tokens) - pos - 1)
        model(x[pos : pos + n][None], cache=prompt_cache)
        mx.eval([c.state for c in prompt_cache])
        pos += n
        mx.clear_cache()

    current = x[pos:]
    rows = []
    for step in range(max_tokens):
        logits = model(current[None], cache=prompt_cache)[:, -1, :].astype(mx.float32)
        mx.eval(logits)
        logits_np = np.array(logits[0])
        top_ids = np.argsort(-logits_np)[:2].astype(np.int64).tolist()
        top_values = logits_np[top_ids].astype(np.float32).tolist()
        token_id = int(top_ids[0])
        rows.append(
            {
                "token": token_id,
                "top1": int(top_ids[0]),
                "top2": int(top_ids[1]),
                "top1_logit": float(top_values[0]),
                "top2_logit": float(top_values[1]),
                "margin": float(top_values[0] - top_values[1]),
            }
        )
        current = mx.array([token_id], dtype=mx.int32)
        mx.clear_cache()
    return rows


def array_meta(arr):
    raw = arr.tobytes()
    top = np.argsort(-arr)[:10]
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "top10_ids": top.tolist(),
        "top10_values": arr[top].tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-tokens", required=True)
    ap.add_argument("--prefill-step-size", type=int, required=True)
    ap.add_argument("--max-tokens", type=int, required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--trust-remote-code", action="store_true")
    args = ap.parse_args()

    log(f"Loading model: {args.model}")
    started = time.perf_counter()
    model, tokenizer = load(
        args.model,
        tokenizer_config={"trust_remote_code": True if args.trust_remote_code else None},
    )
    mx.eval(model.parameters())
    load_s = time.perf_counter() - started
    log(f"Loaded in {load_s:.2f}s")

    prompt_tokens = [int(x) for x in args.prompt_tokens.split(",") if x]
    arrays = {}
    meta = {}
    for n in prompt_tokens:
        tokens = build_prompt(tokenizer, n)
        log(f"Prompt tokens: {n}")
        direct = final_logits(model, tokens)
        cached = cached_final_logits(model, tokens, args.prefill_step_size)
        generated = greedy_tokens(
            model, tokenizer, tokens, args.max_tokens, args.prefill_step_size
        )
        arrays[f"direct_{n}"] = direct
        arrays[f"cached_{n}"] = cached
        meta[f"direct_{n}"] = array_meta(direct)
        meta[f"cached_{n}"] = array_meta(cached)
        meta[f"greedy_{n}"] = {
            "tokens": generated,
            "sha256": hashlib.sha256(
                np.array(generated, dtype=np.int32).tobytes()
            ).hexdigest(),
        }
        meta[f"trace_{n}"] = greedy_trace(
            model, tokens, args.max_tokens, args.prefill_step_size
        )
        mx.clear_cache()

    np.savez(args.output, **arrays)
    out = {
        "label": args.label,
        "model": args.model,
        "load_s": load_s,
        "prompt_tokens": prompt_tokens,
        "prefill_step_size": args.prefill_step_size,
        "max_tokens": args.max_tokens,
        "output": args.output,
        "arrays": meta,
    }
    print(JSON_START)
    print(json.dumps(out, sort_keys=True))
    print(JSON_END)


if __name__ == "__main__":
    main()
'''


def require_uvx() -> str:
    uvx = shutil.which("uvx")
    if not uvx:
        raise SystemExit("uvx not found. Install uv first: https://docs.astral.sh/uv/")
    return uvx


def run_ref(uvx: str, spec: str, label: str, args, out_path: Path) -> dict:
    print(f"\n== {label}: {spec} ==", flush=True)
    cmd = [uvx, "--isolated", "--with", spec]
    if args.refresh:
        cmd.append("--refresh")
    cmd += [
        "python",
        "-c",
        DRIVER,
        "--label",
        spec,
        "--model",
        args.model,
        "--prompt-tokens",
        args.prompt_tokens,
        "--prefill-step-size",
        str(args.prefill_step_size),
        "--max-tokens",
        str(args.max_tokens),
        "--output",
        str(out_path),
    ]
    if args.trust_remote_code:
        cmd.append("--trust-remote-code")
    print("+", " ".join(cmd[:6] + ["...", "--", "<validation args>"]), flush=True)
    proc = subprocess.run(cmd, check=True, text=True, stdout=subprocess.PIPE)
    try:
        blob = proc.stdout.split(JSON_START, 1)[1].split(JSON_END, 1)[0]
    except IndexError:
        print(proc.stdout)
        raise RuntimeError("Validation JSON markers not found in uvx stdout")
    return json.loads(blob)


def compare_arrays(base_path: Path, ours_path: Path) -> list[dict]:
    base = np.load(base_path)
    ours = np.load(ours_path)
    rows = []
    for key in sorted(base.files):
        b = base[key]
        o = ours[key]
        diff = np.abs(o - b)
        b_top = np.argsort(-b)[:10]
        o_top = np.argsort(-o)[:10]
        rows.append(
            {
                "name": key,
                "exact": bool(np.array_equal(b, o)),
                "max_abs": float(diff.max()),
                "mean_abs": float(diff.mean()),
                "top1_equal": bool(b_top[0] == o_top[0]),
                "top10_equal": bool(np.array_equal(b_top, o_top)),
                "base_top1": int(b_top[0]),
                "ours_top1": int(o_top[0]),
            }
        )
    return rows


def compare_sequences(base_meta: dict, ours_meta: dict) -> list[dict]:
    rows = []
    for key in sorted(k for k in base_meta["arrays"] if k.startswith("greedy_")):
        base_tokens = base_meta["arrays"][key]["tokens"]
        ours_tokens = ours_meta["arrays"][key]["tokens"]
        trace_key = key.replace("greedy_", "trace_")
        base_trace = base_meta["arrays"].get(trace_key, [])
        ours_trace = ours_meta["arrays"].get(trace_key, [])
        mismatch = next(
            (i for i, (b, o) in enumerate(zip(base_tokens, ours_tokens)) if b != o),
            None,
        )
        base_margin = "" if mismatch is None else base_trace[mismatch]["margin"]
        ours_margin = "" if mismatch is None else ours_trace[mismatch]["margin"]
        rows.append(
            {
                "name": key,
                "exact": base_tokens == ours_tokens,
                "tokens": min(len(base_tokens), len(ours_tokens)),
                "first_mismatch": "" if mismatch is None else mismatch,
                "base_token": "" if mismatch is None else base_tokens[mismatch],
                "ours_token": "" if mismatch is None else ours_tokens[mismatch],
                "base_margin": base_margin,
                "ours_margin": ours_margin,
            }
        )
    return rows


def print_summary(
    rows: list[dict],
    seq_rows: list[dict],
    base_meta: dict,
    ours_meta: dict,
    paths: list[Path],
):
    print("\n\n===== COPY/PASTE VALIDATION =====")
    print("# DeepSeek V4 logits validation")
    print("")
    print(f"- base: `{base_meta['label']}`")
    print(f"- ours: `{ours_meta['label']}`")
    print(f"- model: `{ours_meta['model']}`")
    print(f"- prompt tokens: `{','.join(map(str, ours_meta['prompt_tokens']))}`")
    print(f"- prefill step: `{ours_meta['prefill_step_size']}`")
    print(f"- greedy max tokens: `{ours_meta['max_tokens']}`")
    print("")
    print("## Greedy generation")
    print("")
    print(
        "| sequence | exact | compared tokens | first mismatch | base token | ours token | base margin | ours margin |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in seq_rows:
        print(
            "| {name} | {exact} | {tokens} | {first_mismatch} | {base_token} | {ours_token} | {base_margin} | {ours_margin} |".format(
                **row
            )
        )
    print("")
    print("## Final logits")
    print("")
    print(
        "| logits | exact | max abs diff | mean abs diff | top1 equal | top10 equal | base top1 | ours top1 |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            "| {name} | {exact} | {max_abs:.9g} | {mean_abs:.9g} | {top1_equal} | {top10_equal} | {base_top1} | {ours_top1} |".format(
                **row
            )
        )
    print("")
    print("## Cached files")
    for path in paths:
        print(f"- `{path}`")
    print("===== END VALIDATION =====")


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare final logits for two mlx-lm refs.")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--base", default=BASE_SPEC)
    ap.add_argument("--ours", default=OURS_SPEC)
    ap.add_argument("--prompt-tokens", default="1,128,2304")
    ap.add_argument("--prefill-step-size", type=int, default=512)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--cache-dir", default="ds4-ben-runs/validate")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    uvx = require_uvx()
    out_dir = Path(args.cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    base_path = out_dir / f"{stamp}-base.npz"
    ours_path = out_dir / f"{stamp}-ours.npz"
    meta_path = out_dir / f"{stamp}-meta.json"

    base_meta = run_ref(uvx, args.base, "base", args, base_path)
    ours_meta = run_ref(uvx, args.ours, "ours", args, ours_path)
    rows = compare_arrays(base_path, ours_path)
    seq_rows = compare_sequences(base_meta, ours_meta)
    meta_path.write_text(
        json.dumps(
            {
                "base": base_meta,
                "ours": ours_meta,
                "logits_comparison": rows,
                "sequence_comparison": seq_rows,
            },
            indent=2,
        )
        + "\n"
    )
    print_summary(rows, seq_rows, base_meta, ours_meta, [base_path, ours_path, meta_path])


if __name__ == "__main__":
    main()
