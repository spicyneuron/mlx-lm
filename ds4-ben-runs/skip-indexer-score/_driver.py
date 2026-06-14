
import argparse
import json
import os
import platform
import time
from statistics import median

import mlx.core as mx
from mlx_lm.generate import stream_generate
from mlx_lm.utils import load


def tokenize(tokenizer, text):
    return tokenizer.encode(text, add_special_tokens=False)


def build_prompt(tokenizer, target_tokens, seed_text):
    base = tokenize(tokenizer, seed_text)
    if not base:
        base = [0]
    reps = (target_tokens + len(base) - 1) // len(base)
    return (base * reps)[:target_tokens]


def run_once(model, tokenizer, prompt, max_tokens, seed, case_name):
    mx.random.seed(seed)
    t0 = time.perf_counter()
    last = None
    token_count = 0
    # Keep this call compatible across the DS4 refs. The default sampler is
    # deterministic argmax, so avoid CLI-only sampling kwargs like temp/top_p.
    for resp in stream_generate(
        model,
        tokenizer,
        prompt,
        max_tokens=max_tokens,
    ):
        last = resp
        token_count += 1
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
    ap.add_argument("--repeats", type=int, required=True)
    ap.add_argument("--warmups", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    started = time.perf_counter()
    model, tokenizer = load(
        args.model,
        tokenizer_config={"trust_remote_code": True if args.trust_remote_code else None},
    )
    mx.eval(model.parameters())
    load_s = time.perf_counter() - started

    cases = json.loads(args.cases_json)
    records = []
    seed_text = (
        "DeepSeek V4 performance benchmark prompt. "
        "This text is repeated to create a deterministic context. "
    )

    for case in cases:
        prompt = build_prompt(tokenizer, case["prompt_tokens"], seed_text)
        max_tokens = case["max_tokens"]
        name = case["name"]

        for i in range(args.warmups):
            run_once(model, tokenizer, prompt, max_tokens, args.seed + i, name)
            mx.clear_cache()

        for i in range(args.repeats):
            rec = run_once(model, tokenizer, prompt, max_tokens, args.seed + 1000 + i, name)
            rec["repeat"] = i
            records.append(rec)
            mx.clear_cache()

    by_case = {}
    for rec in records:
        by_case.setdefault(rec["case"], []).append(rec)

    summary = {}
    for name, rows in by_case.items():
        summary[name] = {
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
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    import sys
    main()
