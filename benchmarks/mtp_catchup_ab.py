#!/usr/bin/env python3
"""Does fixing the full-accept mtp_cache drift recover D2 acceptance / throughput?

Bug: on a full-accept block the bonus path jumps to target_y[n_draft] and never
feeds the last accepted draft token through the MTP, so mtp_cache loses one
position per full-accept block. Its offset drifts -1 each time (model_cache, which
verify advances, does not), skewing the 1-layer draft head's RoPE positions and
plausibly depressing acceptance -- the symptom we'd otherwise pin on "head
capacity". Standard speculative decoding handles exactly this case (generate.py
_draft_generate: "include the last draft token in the next draft step"); the MTP
path did not. The fix (mtp_full_accept_catchup=True, now the default) feeds that
token via an update_only MTP step.

This is a controlled same-process A/B on real text:
  * catchup OFF reproduces the buggy baseline; max_cache_skew should CLIMB past 1.
  * catchup ON  applies the fix;              max_cache_skew should stay == 1.
Then compare D2 acceptance and net (wall-clock) generation_tps. The catch-up adds
one MTP forward per full-accept block, so generation_tps is the honest net metric
(it already includes that cost); acceptance has to clear it to be a real win. The
gain (if any) should grow with context -- drift accumulates over the generation
while the per-block catch-up cost is constant -- so 16k is where it shows up.

    cat mlx_lm/*.py > $TMPDIR/big.txt
    uv run --with-editable . benchmarks/mtp_catchup_ab.py \
        --model ../../glm-5.2/lm-01-mtp --prompt-file $TMPDIR/big.txt \
        --contexts 4096,16384 --num-draft-tokens 2
"""

import argparse

from mlx_lm.utils import load
from mlx_lm.generate import stream_generate


def run(model, tokenizer, prompt, n_draft, prefill_step_size, catchup):
    """Generate 256 MTP tokens; return (acceptance, generation_tps, max_cache_skew)."""
    stats = {}
    last = None
    for last in stream_generate(
        model,
        tokenizer,
        prompt,
        max_tokens=256,
        prefill_step_size=prefill_step_size,
        temp=0.0,
        mtp=True,
        num_draft_tokens=n_draft,
        mtp_full_accept_catchup=catchup,
        mtp_stats_callback=lambda s: (stats.clear(), stats.update(s)),
    ):
        pass
    acc, prop = stats.get("accepted", 0), stats.get("proposed", 0)
    rate = acc / prop if prop else 0.0
    return rate, last.generation_tps, stats.get("max_cache_skew", -1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--prompt-file", required=True)
    p.add_argument("--contexts", default="4096,16384")
    p.add_argument("--num-draft-tokens", type=int, default=2)
    p.add_argument("--prefill-step-size", type=int, default=2048)
    args = p.parse_args()

    model, tokenizer = load(args.model)
    tokenizer._eos_token_ids = {}  # force full 256-token generations
    ids_all = tokenizer.encode(open(args.prompt_file).read())
    contexts = [int(c) for c in args.contexts.split(",") if c]
    nd = args.num_draft_tokens
    print(f"loaded {args.model}  doc tokens={len(ids_all)}  D{nd}\n")

    hdr = f"{'context':>7} {'mode':>9} {'accept':>7} {'gen_tps':>8} {'skew':>5}"
    print(hdr)
    print("-" * len(hdr))
    for C in contexts:
        if len(ids_all) < C:
            print(f"{C:>7}  SKIP (need {C} tokens, have {len(ids_all)})")
            continue
        prompt = ids_all[:C]
        off = run(model, tokenizer, prompt, nd, args.prefill_step_size, False)
        on = run(model, tokenizer, prompt, nd, args.prefill_step_size, True)
        print(f"{C:>7} {'off(buggy)':>9} {off[0]:>6.1%} {off[1]:>8.2f} {off[2]:>5d}")
        print(f"{C:>7} {'on(fixed)':>9} {on[0]:>6.1%} {on[1]:>8.2f} {on[2]:>5d}")
        d_acc = on[0] - off[0]
        d_tps = (on[1] / off[1] - 1.0) if off[1] else 0.0
        print(f"{C:>7} {'delta':>9} {d_acc:>+6.1%} {d_tps:>+7.1%}  tps\n")

    print(
        "skew off >> 1 and skew on == 1  => the drift is real and the fix closes it.\n"
        "delta accept > 0 with delta tps >= 0  => the 'head-capacity' ceiling was\n"
        "  partly this bug; re-open the MTP shelve decision (esp. if 16k crosses 1.0x).\n"
        "delta ~ 0 everywhere  => drift was benign for draft quality; shelve stands."
    )


if __name__ == "__main__":
    main()
