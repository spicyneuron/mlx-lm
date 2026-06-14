#!/usr/bin/env python3
"""
One-off DeepSeek V4 MLX benchmark harness.

Runs each git ref in an isolated uvx environment and prints compact markdown
results to stdout. Fresh run JSON is cached under ds4-ben-runs/cache by default.

Example:
  ./bench_ds4_refs.py --model 284-deepseek-4
  ./bench_ds4_refs.py --model 284-deepseek-4 --only ours --baseline-json base.json

Useful knobs:
  --cases decode_128,csa_2k_decode,long_8k_decode
  --prefill-step-sizes 16,64,128,256,512,2048
  --repeats 3 --warmups 1
  --max-tokens 128
  --refresh
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

BASE_SPEC = "mlx-lm@git+https://github.com/spicyneuron/mlx-lm@_ds4"
OURS_SPEC = "mlx-lm@git+https://github.com/spicyneuron/mlx-lm@_ds4_perf"
DEFAULT_MODEL = "mlx-community/DeepSeek-V4-Flash-8bit"
JSON_START = "===DS4_BENCH_JSON_START==="
JSON_END = "===DS4_BENCH_JSON_END==="

DRIVER = r'''
import argparse
import json
import platform
import sys
import time
import time
from statistics import median

import mlx.core as mx
from mlx_lm.generate import stream_generate
from mlx_lm.utils import load

JSON_START = "===DS4_BENCH_JSON_START==="
JSON_END = "===DS4_BENCH_JSON_END==="


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def tokenize(tokenizer, text):
    return tokenizer.encode(text, add_special_tokens=False)


def build_prompt(tokenizer, target_tokens, seed_text):
    base = tokenize(tokenizer, seed_text)
    if not base:
        base = [0]
    reps = (target_tokens + len(base) - 1) // len(base)
    return (base * reps)[:target_tokens]


def run_once(model, tokenizer, prompt, max_tokens, prefill_step_size, seed, case_name):
    mx.random.seed(seed)
    t0 = time.perf_counter()
    last = None
    for resp in stream_generate(
        model,
        tokenizer,
        prompt,
        max_tokens=max_tokens,
        prefill_step_size=prefill_step_size,
    ):
        last = resp
    mx.eval()
    total_s = time.perf_counter() - t0
    if last is None:
        raise RuntimeError(f"case {case_name!r} generated no tokens")
    return {
        "case": case_name,
        "prompt_tokens": int(last.prompt_tokens),
        "generation_tokens": int(last.generation_tokens),
        "prompt_tps": float(last.prompt_tps),
        "generation_tps": float(last.generation_tps),
        "total_s": total_s,
        "peak_memory_gb": float(last.peak_memory),
        "finish_reason": last.finish_reason,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--cases-json", required=True)
    ap.add_argument("--prefill-step-sizes-json", required=True)
    ap.add_argument("--repeats", type=int, required=True)
    ap.add_argument("--warmups", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
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

    cases = json.loads(args.cases_json)
    prefill_steps = json.loads(args.prefill_step_sizes_json)
    records = []
    seed_text = (
        "DeepSeek V4 performance benchmark prompt. "
        "This text is repeated to create a deterministic context. "
    )

    for case in cases:
        prompt = build_prompt(tokenizer, case["prompt_tokens"], seed_text)
        max_tokens = case["max_tokens"]
        name = case["name"]
        for prefill_step_size in prefill_steps:
            log(
                f"Case {name}: prompt={len(prompt)} max_tokens={max_tokens} "
                f"prefill_step_size={prefill_step_size}"
            )

            for i in range(args.warmups):
                log(f"  warmup {i + 1}/{args.warmups}")
                run_once(
                    model,
                    tokenizer,
                    prompt,
                    max_tokens,
                    prefill_step_size,
                    args.seed + i,
                    name,
                )
                mx.clear_cache()

            for i in range(args.repeats):
                log(f"  repeat {i + 1}/{args.repeats}")
                rec = run_once(
                    model,
                    tokenizer,
                    prompt,
                    max_tokens,
                    prefill_step_size,
                    args.seed + 1000 + i,
                    name,
                )
                rec["repeat"] = i
                rec["prefill_step_size"] = prefill_step_size
                records.append(rec)
                log(
                    "    prompt_tps={:.3f} gen_tps={:.3f} total_s={:.3f}".format(
                        rec["prompt_tps"], rec["generation_tps"], rec["total_s"]
                    )
                )
                mx.clear_cache()

    by_case = {}
    for rec in records:
        key = f'{rec["case"]}@{rec["prefill_step_size"]}'
        by_case.setdefault(key, []).append(rec)

    summary = {}
    for key, rows in by_case.items():
        summary[key] = {
            "case": rows[0]["case"],
            "prefill_step_size": rows[0]["prefill_step_size"],
            "prompt_tokens": rows[0]["prompt_tokens"],
            "generation_tokens": rows[0]["generation_tokens"],
            "prompt_tps_median": median(r["prompt_tps"] for r in rows),
            "generation_tps_median": median(r["generation_tps"] for r in rows),
            "total_s_median": median(r["total_s"] for r in rows),
            "peak_memory_gb_max": max(r["peak_memory_gb"] for r in rows),
            "repeats": len(rows),
        }

    out = {
        "label": args.label,
        "model": args.model,
        "load_s": load_s,
        "python": sys.version,
        "platform": platform.platform(),
        "mlx_default_device": str(mx.default_device()),
        "records": records,
        "summary": summary,
    }
    print(JSON_START)
    print(json.dumps(out, sort_keys=True))
    print(JSON_END)


if __name__ == "__main__":
    main()
'''


@dataclass(frozen=True)
class Target:
    label: str
    spec: str


def require_uvx() -> str:
    uvx = shutil.which("uvx")
    if not uvx:
        raise SystemExit("uvx not found. Install uv first: https://docs.astral.sh/uv/")
    return uvx


def default_cases(max_tokens: int) -> list[dict[str, int | str]]:
    return [
        {"name": "decode_128", "prompt_tokens": 128, "max_tokens": max_tokens},
        {"name": "csa_2k_decode", "prompt_tokens": 2304, "max_tokens": max_tokens},
        {"name": "long_8k_decode", "prompt_tokens": 8192, "max_tokens": max_tokens},
    ]


def parse_cases(raw: str, max_tokens: int) -> list[dict[str, int | str]]:
    presets = {c["name"]: c for c in default_cases(max_tokens)}
    cases = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if item in presets:
            cases.append(dict(presets[item]))
            continue
        parts = item.split(":")
        if len(parts) not in (2, 3):
            raise SystemExit(f"Bad case {item!r}. Use preset or name:prompt_tokens[:max_tokens]")
        cases.append(
            {
                "name": parts[0],
                "prompt_tokens": int(parts[1]),
                "max_tokens": int(parts[2]) if len(parts) == 3 else max_tokens,
            }
        )
    return cases


def parse_prefill_steps(raw: str) -> list[int]:
    steps = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not steps:
        raise SystemExit("--prefill-step-sizes must include at least one integer")
    if any(s <= 0 for s in steps):
        raise SystemExit("--prefill-step-sizes must be positive integers")
    return steps


def pct(new: float, old: float) -> float:
    return 100.0 * (new - old) / old if old else float("nan")


def load_baseline(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    return data.get("result", data)


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "result"


def save_result(result: dict, cache_dir: str, meta: dict) -> Path:
    out_dir = Path(cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    label = safe_name(result["label"].split("@")[-1])
    path = out_dir / f"{meta['created_at']}-{label}.json"
    latest = out_dir / f"latest-{label}.json"
    payload = {"meta": meta, "result": result}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    latest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def run_target(uvx: str, target: Target, args, cases) -> dict:
    print(f"\n== {target.label}: {target.spec} ==", flush=True)
    cmd = [uvx, "--isolated", "--with", target.spec]
    if args.refresh:
        cmd.append("--refresh")
    cmd += [
        "python",
        "-c",
        DRIVER,
        "--label", target.spec,
        "--model", args.model,
        "--cases-json", json.dumps(cases),
        "--prefill-step-sizes-json", json.dumps(args.prefill_steps),
        "--repeats", str(args.repeats),
        "--warmups", str(args.warmups),
        "--seed", str(args.seed),
    ]
    if args.trust_remote_code:
        cmd.append("--trust-remote-code")

    shown = " ".join(cmd[:6] + ["...", "--", "<benchmark args>"])
    print("+", shown, flush=True)
    proc = subprocess.run(cmd, check=True, text=True, stdout=subprocess.PIPE)
    stdout = proc.stdout
    try:
        blob = stdout.split(JSON_START, 1)[1].split(JSON_END, 1)[0]
    except IndexError:
        print(stdout)
        raise RuntimeError("Benchmark JSON markers not found in uvx stdout")
    return json.loads(blob)


def sorted_summary(result: dict) -> list[dict]:
    return [
        row
        for _, row in sorted(
            result["summary"].items(),
            key=lambda kv: (
                kv[1].get("prompt_tokens", 0),
                kv[1].get("generation_tokens", 0),
                kv[1].get("prefill_step_size", 0),
                kv[1].get("case", kv[0]),
            ),
        )
    ]


def row_key(row: dict) -> tuple:
    return (
        row["case"],
        row["prefill_step_size"],
        row["prompt_tokens"],
        row["generation_tokens"],
    )


def result_table(result: dict, title: str) -> list[str]:
    lines = [
        f"## {title}",
        f"- label: `{result['label']}`",
        f"- load: `{result.get('load_s', 0):.2f}s`",
        "",
        "| case | step | prompt toks | gen toks | gen tok/s | prompt tok/s | total s | peak GB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted_summary(result):
        lines.append(
            "| {case_name} | {step} | {pt} | {gt} | {g:.3f} | {p:.3f} | {t:.3f} | {m:.3f} |".format(
                case_name=row["case"],
                step=row["prefill_step_size"],
                pt=row["prompt_tokens"],
                gt=row["generation_tokens"],
                g=row["generation_tps_median"],
                p=row["prompt_tps_median"],
                t=row["total_s_median"],
                m=row["peak_memory_gb_max"],
            )
        )
    return lines


def markdown_summary(base: dict | None, ours: dict | None, meta: dict) -> str:
    lines = [
        "# DeepSeek V4 MLX benchmark",
        "",
        f"- host: `{meta['host']}`",
        f"- model: `{meta['model']}`",
        f"- cases: `{meta['cases']}`",
        f"- prefill steps: `{meta['prefill_steps']}`",
        f"- repeats/warmups: `{meta['repeats']}/{meta['warmups']}`",
    ]
    if base:
        lines.append(f"- base: `{base['label']}`")
    if ours:
        lines.append(f"- ours: `{ours['label']}`")
    lines.append("")

    if base:
        lines += result_table(base, "Base results")
        lines.append("")
    if ours:
        lines += result_table(ours, "Ours results")
        lines.append("")

    if base and ours:
        ours_by_key = {row_key(row): row for row in ours["summary"].values()}
        matched = []
        for b in sorted_summary(base):
            o = ours_by_key.get(row_key(b))
            if o is not None:
                matched.append((b, o))
        lines += [
            "## Matched comparison",
            "| case | step | prompt toks | gen toks | base gen tok/s | ours gen tok/s | delta | base prompt tok/s | ours prompt tok/s | delta | base total s | ours total s | delta |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        if not matched:
            lines.append("| _No matching case/step/prompt/gen rows_ | | | | | | | | | | | | |")
        for b, o in matched:
            lines.append(
                "| {case_name} | {step} | {pt} | {gt} | {bg:.3f} | {og:.3f} | {gd:+.2f}% | {bp:.3f} | {op:.3f} | {pd:+.2f}% | {bt:.3f} | {ot:.3f} | {td:+.2f}% |".format(
                    case_name=b["case"],
                    step=b.get("prefill_step_size", ""),
                    pt=b["prompt_tokens"],
                    gt=b["generation_tokens"],
                    bg=b["generation_tps_median"],
                    og=o["generation_tps_median"],
                    gd=pct(o["generation_tps_median"], b["generation_tps_median"]),
                    bp=b["prompt_tps_median"],
                    op=o["prompt_tps_median"],
                    pd=pct(o["prompt_tps_median"], b["prompt_tps_median"]),
                    bt=b["total_s_median"],
                    ot=o["total_s_median"],
                    td=pct(o["total_s_median"], b["total_s_median"]),
                )
            )

    lines += [
        "",
        "Rule of thumb: treat <3% as noise unless it repeats across runs; prioritize >5-10% deltas.",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark two mlx-lm git refs for DS4 inference via uvx.")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--base", default=BASE_SPEC, help="uv requirement for baseline mlx-lm")
    ap.add_argument("--ours", default=OURS_SPEC, help="uv requirement for experiment mlx-lm")
    ap.add_argument("--cases", default="decode_128,csa_2k_decode,long_8k_decode")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument(
        "--prefill-step-sizes",
        default="2048",
        help="Comma-separated prefill step sizes to sweep, e.g. 16,64,128,256,512,2048",
    )
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmups", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--refresh", action="store_true", help="Ask uvx to refresh cached package data")
    ap.add_argument("--only", choices=("base", "ours", "both"), default="both")
    ap.add_argument(
        "--cache-dir",
        default="ds4-ben-runs/cache",
        help="Directory for fresh run JSON cache. Use --no-cache to disable.",
    )
    ap.add_argument("--no-cache", action="store_true", help="Do not write fresh run JSON.")
    ap.add_argument(
        "--baseline-json",
        help="Path to a previous base.json. When set, base is not re-run unless --only base is used.",
    )
    args = ap.parse_args()

    uvx = require_uvx()
    cases = parse_cases(args.cases, args.max_tokens)
    args.prefill_steps = parse_prefill_steps(args.prefill_step_sizes)
    results = {}
    if args.baseline_json:
        results["base"] = load_baseline(args.baseline_json)

    targets = []
    if args.only in ("base", "both") and not (args.baseline_json and args.only != "base"):
        targets.append(Target("base", args.base))
    if args.only in ("ours", "both"):
        targets.append(Target("ours", args.ours))

    meta = {
        "host": platform.node(),
        "model": args.model,
        "cases": args.cases,
        "prefill_steps": args.prefill_step_sizes,
        "repeats": args.repeats,
        "warmups": args.warmups,
        "created_at": time.strftime("%Y%m%d-%H%M%S"),
        "python": sys.version.split()[0],
    }

    saved = []
    for target in targets:
        result = run_target(uvx, target, args, cases)
        results[target.label] = result
        if not args.no_cache:
            saved.append(save_result(result, args.cache_dir, meta))

    print("\n\n===== COPY/PASTE SUMMARY =====")
    print(markdown_summary(results.get("base"), results.get("ours"), meta))
    if saved:
        print("")
        print("## Cached JSON")
        for path in saved:
            print(f"- `{path}`")
    print("===== END SUMMARY =====")


if __name__ == "__main__":
    main()
