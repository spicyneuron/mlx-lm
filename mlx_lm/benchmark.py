# Copyright © 2025 Apple Inc.

import argparse
import time

import mlx.core as mx

from mlx_lm import batch_generate, load, stream_generate
from mlx_lm.generate import (
    DEFAULT_MIN_P,
    DEFAULT_MODEL,
    DEFAULT_TEMP,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
)
from mlx_lm.utils import pipeline_load, sharded_load


def setup_arg_parser():
    """Set up and return the argument parser."""
    parser = argparse.ArgumentParser(description="LLM benchmarking script")
    parser.add_argument(
        "--model",
        type=str,
        help=(
            "The path to the local model directory or Hugging Face repo. "
            f"If no model is specified, then {DEFAULT_MODEL} is used."
        ),
        default=None,
    )
    parser.add_argument(
        "--prompt-tokens",
        "-p",
        default=512,
        help="Length of random prompt when --prompt is not provided",
        type=int,
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Text prompt to benchmark instead of a random token prompt.",
    )
    parser.add_argument(
        "--generation-tokens",
        "-g",
        default=1024,
        help="Length of completion",
        type=int,
    )
    parser.add_argument(
        "--temp", type=float, default=DEFAULT_TEMP, help="Sampling temperature"
    )
    parser.add_argument(
        "--top-p", type=float, default=DEFAULT_TOP_P, help="Sampling top-p"
    )
    parser.add_argument(
        "--min-p", type=float, default=DEFAULT_MIN_P, help="Sampling min-p"
    )
    parser.add_argument(
        "--top-k", type=int, default=DEFAULT_TOP_K, help="Sampling top-k"
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        default=1,
        help="Batch size",
        type=int,
    )
    parser.add_argument(
        "--num-trials",
        "-n",
        default=5,
        help="Number of timing trials",
        type=int,
    )
    parser.add_argument(
        "--pipeline",
        action="store_true",
        help="Use pipelining instead of tensor parallelism",
    )
    parser.add_argument(
        "--quantize-activations",
        "-qa",
        action="store_true",
        help="Quantize activations using the same quantization config as the corresponding layer.",
    )
    parser.add_argument(
        "--prefill-step-size",
        type=int,
        default=2048,
        help="Step size for prefill processing (default: 2048)",
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=0,
        help="Delay between each test in seconds (default: 0)",
    )
    parser.add_argument(
        "--draft-mtp",
        dest="mtp",
        action="store_true",
        help="Use native MTP layers as a speculative draft model.",
    )
    parser.add_argument(
        "--num-draft-tokens",
        type=int,
        default=3,
        help="Number of tokens to draft when using --draft-mtp.",
    )
    return parser


def main():
    parser = setup_arg_parser()
    args = parser.parse_args()
    if args.mtp and args.batch_size != 1:
        parser.error("--draft-mtp only supports --batch-size 1.")
    if args.prompt is not None and args.batch_size != 1:
        parser.error("--prompt only supports --batch-size 1.")
    mx.random.seed(0)

    group = mx.distributed.init()
    rank = group.rank()
    pipeline_group = group if args.pipeline else None
    tensor_group = group if not args.pipeline else None

    def rprint(*args, **kwargs):
        if rank == 0:
            print(*args, **kwargs)

    model_path = args.model or DEFAULT_MODEL

    if group.size() > 1:
        model, tokenizer, config = sharded_load(
            model_path, pipeline_group, tensor_group, return_config=True
        )
    else:
        model, tokenizer, config = load(
            model_path,
            return_config=True,
            tokenizer_config={"trust_remote_code": True},
            model_config={"quantize_activations": args.quantize_activations},
        )

    # Empty to avoid early stopping
    tokenizer._eos_token_ids = {}

    prompt_tokens = args.prompt_tokens
    generation_tokens = args.generation_tokens
    batch_size = args.batch_size
    if args.prompt is None:
        vocab_size = config.get("vocab_size") or config["text_config"]["vocab_size"]
        prompts = mx.random.randint(0, vocab_size, (batch_size, prompt_tokens)).tolist()
        prompt = prompts[0]
    else:
        prompt = tokenizer.encode(args.prompt)
        prompt_tokens = len(prompt)
        prompts = [prompt]

    def single_bench():
        mtp_stats = {}

        def update_mtp_stats(stats):
            mtp_stats.clear()
            mtp_stats.update(stats)

        for response in stream_generate(
            model,
            tokenizer,
            prompt,
            max_tokens=generation_tokens,
            prefill_step_size=args.prefill_step_size,
            mtp=args.mtp,
            num_draft_tokens=args.num_draft_tokens,
            mtp_stats_callback=update_mtp_stats if args.mtp else None,
            temp=args.temp,
            top_p=args.top_p,
            min_p=args.min_p,
            top_k=args.top_k,
        ):
            pass
        if args.mtp:
            response.mtp_stats = mtp_stats
        return response

    def batch_bench():
        return batch_generate(
            model,
            tokenizer,
            prompts,
            max_tokens=generation_tokens,
            prefill_step_size=args.prefill_step_size,
        ).stats

    if batch_size == 1:
        _bench = single_bench
    else:
        _bench = batch_bench

    rprint("Running warmup..")
    _bench()

    report_keys = ["prompt_tps", "generation_tps", "peak_memory"]
    rprint(f"Timing with {prompt_tokens=}, {generation_tokens=}, {batch_size=}.")
    responses = []

    def mtp_results(response):
        stats = getattr(response, "mtp_stats", {})
        accepted = stats.get("accepted", 0)
        proposed = stats.get("proposed", 0)
        acceptance_rate = accepted / proposed if proposed else 0
        return [f"mtp_acceptance={100 * acceptance_rate:.1f}%"]

    for i in range(args.num_trials):
        if args.delay > 0:
            time.sleep(args.delay)
        tic = time.perf_counter()
        response = _bench()
        toc = time.perf_counter()
        responses.append(response)
        results = [(k, getattr(response, k)) for k in report_keys]
        results = [f"{k}={v:.3f}" for k, v in results]
        results.append(f"total_time={toc - tic:.3f}")
        if args.mtp:
            results.extend(mtp_results(response))
        rprint(f"Trial {i+1}:  " + ", ".join(results))

    def avg(k):
        vals = (getattr(response, k) for response in responses)
        return sum(vals) / args.num_trials

    results = [(k, avg(k)) for k in report_keys]
    results = [f"{k}={v:.3f}" for k, v in results]
    if args.mtp:
        accepted = sum(getattr(r, "mtp_stats", {}).get("accepted", 0) for r in responses)
        proposed = sum(getattr(r, "mtp_stats", {}).get("proposed", 0) for r in responses)
        acceptance_rate = accepted / proposed if proposed else 0
        results.append(f"mtp_acceptance={100 * acceptance_rate:.1f}%")
    rprint(f"Averages: " + ", ".join(results))


if __name__ == "__main__":
    main()
