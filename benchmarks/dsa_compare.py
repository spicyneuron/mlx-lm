#!/usr/bin/env python3
"""
Compare decode/prefill TPS for the GLM-MoE-DSA / DeepSeek-V3.2 path between
two mlx-lm refs (e.g. main vs. a perf branch). Each ref is installed into an
ephemeral uvx environment, so this script runs standalone — no project install
required.

Usage:
    python dsa_compare.py --model zai-org/GLM-4.6 \\
        [--baseline git+https://github.com/ml-explore/mlx-lm@main] \\
        [--candidate git+https://github.com/spicyneuron/mlx-lm@_glm-perf]

The baseline/candidate args accept anything `uvx --from` does: a git URL with
@ref, a PyPI version spec, or a local path. Defaults compare upstream main
against the perf branch.

For DSA-relevant signal you want a prompt that exceeds `index_topk` (~2048
tokens by default) so the indexer's sparse path actually kicks in. The
included default prompt repeats to ~4K tokens; override with --prompt-file.
"""

import argparse
import json
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_BASELINE = "git+https://github.com/ml-explore/mlx-lm@main"
DEFAULT_CANDIDATE = "git+https://github.com/spicyneuron/mlx-lm@_glm-perf"

# Roughly 80 tokens of filler. Repeated to hit the target prompt length.
FILLER = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
    "How vexingly quick daft zebras jump! "
    "Sphinx of black quartz, judge my vow. "
    "Two driven jocks help fax my big quiz. "
)


def build_prompt(target_tokens: int) -> str:
    # ~5 chars/token rough estimate; over-shoot then let the tokenizer trim.
    reps = max(1, (target_tokens * 5) // len(FILLER))
    body = (FILLER * reps).strip()
    return (
        "Summarize the following passage in exactly one sentence:\n\n"
        + body
        + "\n\nSummary:"
    )


PROMPT_RE = re.compile(r"Prompt:\s+(\d+)\s+tokens,\s+([\d.]+)\s+tokens-per-sec")
GEN_RE = re.compile(r"Generation:\s+(\d+)\s+tokens,\s+([\d.]+)\s+tokens-per-sec")
MEM_RE = re.compile(r"Peak memory:\s+([\d.]+)\s+GB")


def run_once(
    ref: str,
    model: str,
    prompt: str,
    max_tokens: int,
    seed: int,
    extra_args: list[str],
    quiet: bool,
) -> dict:
    cmd = [
        "uvx",
        "--from",
        ref,
        "--",
        "python",
        "-m",
        "mlx_lm.generate",
        "--model",
        model,
        "--prompt",
        "-",  # stdin
        "--max-tokens",
        str(max_tokens),
        "--temp",
        "0",
        "--seed",
        str(seed),
        "--verbose",
        "True",
        *extra_args,
    ]
    start = time.perf_counter()
    proc = subprocess.run(
        cmd,
        input=prompt,
        capture_output=True,
        text=True,
        check=False,
    )
    wall = time.perf_counter() - start
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise RuntimeError(f"mlx_lm.generate failed under {ref}")

    out = proc.stdout
    if not quiet:
        sys.stderr.write(out[-2000:])  # tail for sanity
    p = PROMPT_RE.search(out)
    g = GEN_RE.search(out)
    m = MEM_RE.search(out)
    if not (p and g and m):
        raise RuntimeError(
            f"Could not parse perf lines from output:\n{out[-1000:]}"
        )
    return {
        "ref": ref,
        "wall_seconds": wall,
        "prompt_tokens": int(p.group(1)),
        "prompt_tps": float(p.group(2)),
        "gen_tokens": int(g.group(1)),
        "gen_tps": float(g.group(2)),
        "peak_gb": float(m.group(1)),
    }


def summarize(runs: list[dict], label: str) -> dict:
    keys = ["prompt_tps", "gen_tps", "peak_gb", "wall_seconds"]
    out = {"label": label, "ref": runs[0]["ref"], "n": len(runs)}
    for k in keys:
        vals = [r[k] for r in runs]
        out[k] = {
            "mean": statistics.mean(vals),
            "min": min(vals),
            "max": max(vals),
        }
    out["prompt_tokens"] = runs[0]["prompt_tokens"]
    out["gen_tokens"] = runs[0]["gen_tokens"]
    return out


def fmt_row(label: str, s: dict) -> str:
    return (
        f"{label:<10s} "
        f"prefill {s['prompt_tps']['mean']:8.2f} tps  "
        f"decode {s['gen_tps']['mean']:7.2f} tps  "
        f"peak {s['peak_gb']['mean']:6.2f} GB  "
        f"wall {s['wall_seconds']['mean']:6.2f} s"
    )


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", required=True, help="HF repo id or local path")
    p.add_argument("--baseline", default=DEFAULT_BASELINE)
    p.add_argument("--candidate", default=DEFAULT_CANDIDATE)
    p.add_argument(
        "--prompt-file",
        type=Path,
        help="Read prompt from file instead of the generated filler",
    )
    p.add_argument(
        "--prompt-tokens",
        type=int,
        default=4096,
        help="Target prompt length when using the generated filler (default: 4096)",
    )
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--runs", type=int, default=3, help="Runs per ref (first is warmup)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--extra",
        nargs=argparse.REMAINDER,
        default=[],
        help=(
            "Anything after this flag is forwarded to mlx_lm.generate. "
            "Example: --extra --kv-bits 8 --max-kv-size 32768"
        ),
    )
    p.add_argument(
        "--json-out", type=Path, help="Write full per-run results to this JSON file"
    )
    p.add_argument(
        "--quiet", action="store_true", help="Suppress tail-of-stdout per run"
    )
    args = p.parse_args()

    if shutil.which("uvx") is None:
        sys.exit("uvx not found on PATH. Install uv: https://docs.astral.sh/uv/")

    if args.prompt_file:
        prompt = args.prompt_file.read_text()
    else:
        prompt = build_prompt(args.prompt_tokens)

    extra_args = list(args.extra)

    print("=" * 72)
    print(f"Model:      {args.model}")
    print(f"Baseline:   {args.baseline}")
    print(f"Candidate:  {args.candidate}")
    print(f"Prompt:     ~{len(prompt)} chars, target {args.prompt_tokens} tokens")
    print(f"Max-tokens: {args.max_tokens}")
    print(f"Runs/ref:   {args.runs} (first is warmup, discarded)")
    if extra_args:
        print(f"Extra:      {' '.join(extra_args)}")
    print("=" * 72)

    all_runs = {"baseline": [], "candidate": []}
    for label, ref in [("baseline", args.baseline), ("candidate", args.candidate)]:
        print(f"\n[{label}] {ref}")
        for i in range(args.runs):
            tag = "warmup" if i == 0 else f"run {i}"
            print(f"  {tag}... ", end="", flush=True)
            try:
                r = run_once(
                    ref, args.model, prompt, args.max_tokens,
                    args.seed, extra_args, args.quiet,
                )
            except Exception as e:
                print(f"FAILED: {e}")
                return 1
            print(
                f"{r['prompt_tokens']} prompt tok, "
                f"prefill {r['prompt_tps']:.1f} tps, "
                f"decode {r['gen_tps']:.1f} tps, "
                f"peak {r['peak_gb']:.2f} GB"
            )
            if i > 0:
                all_runs[label].append(r)

    base = summarize(all_runs["baseline"], "baseline")
    cand = summarize(all_runs["candidate"], "candidate")

    print("\n" + "=" * 72)
    print("Summary (mean over measured runs)")
    print("-" * 72)
    print(fmt_row("baseline", base))
    print(fmt_row("candidate", cand))
    print("-" * 72)
    if base["prompt_tokens"] < 2048:
        print(
            f"NOTE: prompt was {base['prompt_tokens']} tokens. The DSA indexer "
            "only triggers above index_topk (~2048). For decode-path signal, "
            "increase --prompt-tokens or pass --prompt-file."
        )

    def delta(a, b):
        if a == 0:
            return float("nan")
        return (b - a) / a * 100.0

    d_prefill = delta(base["prompt_tps"]["mean"], cand["prompt_tps"]["mean"])
    d_decode = delta(base["gen_tps"]["mean"], cand["gen_tps"]["mean"])
    d_peak = delta(base["peak_gb"]["mean"], cand["peak_gb"]["mean"])
    print(
        f"delta      prefill {d_prefill:+6.2f}%      "
        f"decode {d_decode:+6.2f}%        peak {d_peak:+6.2f}%"
    )
    print("=" * 72)

    if args.json_out:
        args.json_out.write_text(
            json.dumps(
                {
                    "model": args.model,
                    "prompt_tokens": base["prompt_tokens"],
                    "max_tokens": args.max_tokens,
                    "baseline": {"summary": base, "runs": all_runs["baseline"]},
                    "candidate": {"summary": cand, "runs": all_runs["candidate"]},
                    "delta_pct": {
                        "prefill_tps": d_prefill,
                        "decode_tps": d_decode,
                        "peak_gb": d_peak,
                    },
                },
                indent=2,
            )
        )
        print(f"Wrote {args.json_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
