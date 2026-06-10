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
import tempfile
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


def resolve_model_path(model: str) -> Path:
    path = Path(model).expanduser()
    if path.exists():
        return path
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise RuntimeError(
            "DSA policy variants need a local model path or huggingface_hub "
            "installed so the HF repo can be materialized and patched."
        ) from e
    return Path(snapshot_download(model))


def link_model_dir(src: Path, tmp_root: Path, config: dict) -> Path:
    dst = tmp_root / src.name
    dst.mkdir()
    for item in src.iterdir():
        target = dst / item.name
        if item.name == "config.json":
            target.write_text(json.dumps(config, indent=2))
        else:
            target.symlink_to(item, target_is_directory=item.is_dir())
    return dst


def make_indexer_types(config: dict, policy: str) -> list[str]:
    layers = config["num_hidden_layers"]
    if policy == "alternate":
        return ["full" if i % 2 == 0 else "shared" for i in range(layers)]
    if policy == "hf-freq2":
        offset = 2
        freq = 2
        return [
            "full" if (max(i - offset + 1, 0) % freq) == 0 else "shared"
            for i in range(layers)
        ]
    raise ValueError(f"Unknown indexer policy: {policy}")


def build_policy_model(
    model: str,
    indexer_policy: str | None,
    index_topk: int | None,
    tmp_root: Path,
) -> tuple[str, dict]:
    src = resolve_model_path(model)
    config_path = src / "config.json"
    if not config_path.exists():
        raise RuntimeError(f"No config.json found in {src}")
    config = json.loads(config_path.read_text())
    policy = {}

    if indexer_policy:
        config["indexer_types"] = make_indexer_types(config, indexer_policy)
        # Keep the generated explicit pattern authoritative even if the original
        # config had frequency fields.
        config.pop("index_topk_freq", None)
        config.pop("index_skip_topk_offset", None)
        policy["indexer_policy"] = indexer_policy
        policy["shared_layers"] = config["indexer_types"].count("shared")

    if index_topk is not None:
        config["index_topk"] = index_topk
        policy["index_topk"] = index_topk

    return str(link_model_dir(src, tmp_root, config)), policy


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


LOGIT_SCRIPT = r"""
import argparse
import json

import mlx.core as mx

from mlx_lm import load

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--prompt-file", required=True)
p.add_argument("--out", required=True)
p.add_argument("--top-k", type=int, default=20)
args = p.parse_args()

model, tokenizer = load(args.model)
prompt = open(args.prompt_file).read()
tokens = tokenizer.encode(prompt)
if not tokens:
    raise SystemExit("Prompt tokenized to zero tokens")
logits = model(mx.array([tokens]))[:, -1, :].astype(mx.float32)
top_vals, top_idx = mx.topk(logits, args.top_k, axis=-1)
mx.eval(logits, top_vals, top_idx)
mx.savez(args.out, logits=logits, top_vals=top_vals, top_idx=top_idx)
print(json.dumps({"tokens": len(tokens), "vocab": logits.shape[-1]}))
"""


COMPARE_LOGITS_SCRIPT = r"""
import argparse
import json

import mlx.core as mx

p = argparse.ArgumentParser()
p.add_argument("--a", required=True)
p.add_argument("--b", required=True)
p.add_argument("--top-k", type=int, default=20)
args = p.parse_args()

a = mx.load(args.a)
b = mx.load(args.b)
la = a["logits"]
lb = b["logits"]
diff = (lb - la).astype(mx.float32)
abs_diff = mx.abs(diff)
rmse = mx.sqrt(mx.mean(diff * diff))
max_abs = mx.max(abs_diff)
mean_abs = mx.mean(abs_diff)
a_top = a["top_idx"][0, : args.top_k]
b_top = b["top_idx"][0, : args.top_k]
overlap = mx.sum(a_top[:, None] == b_top[None, :])
base_argmax = mx.argmax(la, axis=-1)
policy_argmax = mx.argmax(lb, axis=-1)
base_argmax_delta = mx.take_along_axis(diff, base_argmax[:, None], axis=-1)[:, 0]
mx.eval(rmse, max_abs, mean_abs, overlap, base_argmax, policy_argmax, base_argmax_delta)
print(json.dumps({
    "rmse": float(rmse.item()),
    "max_abs": float(max_abs.item()),
    "mean_abs": float(mean_abs.item()),
    "top_k": args.top_k,
    "top_k_overlap": int(overlap.item()),
    "base_argmax": int(base_argmax.item()),
    "policy_argmax": int(policy_argmax.item()),
    "same_argmax": bool(base_argmax.item() == policy_argmax.item()),
    "base_argmax_delta": float(base_argmax_delta.item()),
}))
"""


def run_python_snippet(ref: str, script: str, args: list[str]) -> str:
    cmd = ["uvx", "--from", ref, "--", "python", "-c", script, *args]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise RuntimeError(f"Python snippet failed under {ref}")
    return proc.stdout


def run_logit_check(
    ref: str,
    model: str,
    policy_model: str,
    prompt: str,
    tmp_root: Path,
    top_k: int,
) -> dict:
    prompt_file = tmp_root / "logit_prompt.txt"
    base_logits = tmp_root / "candidate_logits.npz"
    policy_logits = tmp_root / "candidate_policy_logits.npz"
    prompt_file.write_text(prompt)

    common = ["--prompt-file", str(prompt_file), "--top-k", str(top_k)]
    base_meta = json.loads(
        run_python_snippet(
            ref,
            LOGIT_SCRIPT,
            ["--model", model, "--out", str(base_logits), *common],
        )
    )
    policy_meta = json.loads(
        run_python_snippet(
            ref,
            LOGIT_SCRIPT,
            ["--model", policy_model, "--out", str(policy_logits), *common],
        )
    )
    metrics = json.loads(
        run_python_snippet(
            ref,
            COMPARE_LOGITS_SCRIPT,
            ["--a", str(base_logits), "--b", str(policy_logits), "--top-k", str(top_k)],
        )
    )
    return {"base": base_meta, "policy": policy_meta, "metrics": metrics}


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
    p.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Run only the candidate (no A/B comparison). Useful for iterating on "
        "the candidate when the baseline hasn't moved since the last run.",
    )
    p.add_argument(
        "--candidate-indexer-policy",
        choices=("alternate", "hf-freq2"),
        help=(
            "Run an extra candidate variant with a patched config. "
            "'alternate' marks odd layers shared; 'hf-freq2' uses the HF-style "
            "freq=2, offset=2 pattern. This materially changes outputs."
        ),
    )
    p.add_argument(
        "--candidate-index-topk",
        type=int,
        help=(
            "Run an extra candidate variant with config.index_topk overridden. "
            "Setting this above the tested context effectively benchmarks dense "
            "attention for that workload. This materially changes outputs."
        ),
    )
    p.add_argument(
        "--logit-check",
        action="store_true",
        help=(
            "When running a candidate policy variant, also compare last-token "
            "logits between normal candidate and candidate-policy. This adds "
            "two extra model forwards outside the timed benchmark."
        ),
    )
    p.add_argument(
        "--logit-top-k",
        type=int,
        default=20,
        help="Top-k size used for --logit-check overlap reporting (default: 20).",
    )
    args = p.parse_args()

    if shutil.which("uvx") is None:
        sys.exit("uvx not found on PATH. Install uv: https://docs.astral.sh/uv/")

    if args.prompt_file:
        prompt = args.prompt_file.read_text()
    else:
        prompt = build_prompt(args.prompt_tokens)

    extra_args = list(args.extra)

    tmp_ctx = tempfile.TemporaryDirectory(prefix="mlx-lm-dsa-policy-")
    tmp_root = Path(tmp_ctx.name)

    refs = []
    if not args.skip_baseline:
        refs.append({"label": "baseline", "ref": args.baseline, "model": args.model})
    refs.append({"label": "candidate", "ref": args.candidate, "model": args.model})

    policy = None
    if args.candidate_indexer_policy or args.candidate_index_topk is not None:
        try:
            policy_model, policy = build_policy_model(
                args.model,
                args.candidate_indexer_policy,
                args.candidate_index_topk,
                tmp_root,
            )
        except Exception as e:
            print(f"Failed to prepare DSA policy variant: {e}", file=sys.stderr)
            return 1
        refs.append(
            {
                "label": "candidate-policy",
                "ref": args.candidate,
                "model": policy_model,
            }
        )

    print("=" * 72)
    print(f"Model:      {args.model}")
    if not args.skip_baseline:
        print(f"Baseline:   {args.baseline}")
    print(f"Candidate:  {args.candidate}")
    print(f"Prompt:     ~{len(prompt)} chars, target {args.prompt_tokens} tokens")
    print(f"Max-tokens: {args.max_tokens}")
    print(f"Runs/ref:   {args.runs} (first is warmup, discarded)")
    if extra_args:
        print(f"Extra:      {' '.join(extra_args)}")
    if policy:
        print(f"Policy:     {policy}")
    if args.logit_check:
        print(f"Logits:     check enabled, top-k {args.logit_top_k}")
    print("=" * 72)

    all_runs = {r["label"]: [] for r in refs}
    for ref_info in refs:
        label = ref_info["label"]
        ref = ref_info["ref"]
        model = ref_info["model"]
        print(f"\n[{label}] {ref}")
        if model != args.model:
            print(f"  model override: {model}")
        for i in range(args.runs):
            tag = "warmup" if i == 0 else f"run {i}"
            print(f"  {tag}... ", end="", flush=True)
            try:
                r = run_once(
                    ref, model, prompt, args.max_tokens,
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

    summaries = {r["label"]: summarize(all_runs[r["label"]], r["label"]) for r in refs}
    cand = summaries["candidate"]

    print("\n" + "=" * 72)
    print("Summary (mean over measured runs)")
    print("-" * 72)
    for r in refs:
        label = r["label"]
        print(fmt_row(label, summaries[label]))
    print("-" * 72)
    if cand["prompt_tokens"] < 2048:
        print(
            f"NOTE: prompt was {cand['prompt_tokens']} tokens. The DSA indexer "
            "only triggers above index_topk (~2048). For decode-path signal, "
            "increase --prompt-tokens or pass --prompt-file."
        )

    def delta(a, b):
        if a == 0:
            return float("nan")
        return (b - a) / a * 100.0

    deltas = None
    if not args.skip_baseline:
        base = summaries["baseline"]
        deltas = {
            "prefill_tps": delta(base["prompt_tps"]["mean"], cand["prompt_tps"]["mean"]),
            "decode_tps": delta(base["gen_tps"]["mean"], cand["gen_tps"]["mean"]),
            "peak_gb": delta(base["peak_gb"]["mean"], cand["peak_gb"]["mean"]),
        }
        print(
            f"delta      prefill {deltas['prefill_tps']:+6.2f}%      "
            f"decode {deltas['decode_tps']:+6.2f}%        "
            f"peak {deltas['peak_gb']:+6.2f}%"
        )
    if "candidate-policy" in summaries:
        policy_summary = summaries["candidate-policy"]
        print(
            f"policyΔ   prefill "
            f"{delta(cand['prompt_tps']['mean'], policy_summary['prompt_tps']['mean']):+6.2f}%      "
            f"decode "
            f"{delta(cand['gen_tps']['mean'], policy_summary['gen_tps']['mean']):+6.2f}%        "
            f"peak "
            f"{delta(cand['peak_gb']['mean'], policy_summary['peak_gb']['mean']):+6.2f}%"
        )
    logit_check = None
    if args.logit_check:
        if not policy:
            print("logits    skipped (no candidate policy variant requested)")
        else:
            print(
                "logits    running candidate vs candidate-policy check... ",
                end="",
                flush=True,
            )
            try:
                logit_check = run_logit_check(
                    args.candidate,
                    args.model,
                    policy_model,
                    prompt,
                    tmp_root,
                    args.logit_top_k,
                )
            except Exception as e:
                print(f"FAILED: {e}")
                return 1
            m = logit_check["metrics"]
            print(
                f"rmse {m['rmse']:.4g}, max_abs {m['max_abs']:.4g}, "
                f"mean_abs {m['mean_abs']:.4g}, top{m['top_k']} overlap "
                f"{m['top_k_overlap']}/{m['top_k']}, same_argmax {m['same_argmax']}"
            )
    print("=" * 72)

    if args.json_out:
        payload = {
            "model": args.model,
            "prompt_tokens": cand["prompt_tokens"],
            "max_tokens": args.max_tokens,
            "candidate": {"summary": cand, "runs": all_runs["candidate"]},
        }
        if policy:
            payload["policy"] = policy
            payload["candidate_policy"] = {
                "summary": summaries["candidate-policy"],
                "runs": all_runs["candidate-policy"],
            }
        if logit_check:
            payload["logit_check"] = logit_check
        if not args.skip_baseline:
            payload["baseline"] = {
                "summary": summaries["baseline"],
                "runs": all_runs["baseline"],
            }
            payload["delta_pct"] = deltas
        args.json_out.write_text(json.dumps(payload, indent=2))
        print(f"Wrote {args.json_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
