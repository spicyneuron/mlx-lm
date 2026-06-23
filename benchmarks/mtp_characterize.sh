#!/usr/bin/env bash
# MTP characterization run (validated-plan step 1: characterize, no new code).
#
# Goal: decide whether the MTP bottleneck is draft-forward, verify-forward,
# or verify post-processing / host-sync -- BEFORE writing any history-policy code.
#
# It does NOT sweep draft depth. It sweeps *prompt length* (committed MTP history
# cost grows with context) and contrasts each MTP run against a no-MTP baseline at
# the same context, so you can read:
#   - draft_ms/tok rising with prompt length  -> committed-history cost is real
#   - verify_ms/tok vs baseline ms/tok         -> is verify nearly free (MoE) or
#                                                 sync/materialization bound?
#   - acceptance at D1                          -> is the drafter even useful here?
#
# Usage (on the server, after `git checkout` of the branch):
#   MODEL=/path/to/glm52 ./benchmarks/mtp_characterize.sh
#
# Optional overrides:
#   RUN="uv run python"     # how to invoke python (default below)
#   PROMPTS="512 4096 16384"
#   GEN=256                 # tokens generated per run
#   TRIALS=3
#   OUT=/path/to/results.txt
set -euo pipefail

MODEL="${MODEL:?set MODEL=/path/to/glm52}"
RUN="${RUN:-uv run python}"
PROMPTS="${PROMPTS:-512 4096 16384}"
GEN="${GEN:-256}"
TRIALS="${TRIALS:-3}"
OUT="${OUT:-mtp_characterize_$(date +%Y%m%d_%H%M%S).txt}"

bench() { $RUN -m mlx_lm.benchmark --model "$MODEL" --temp 0.0 \
            --generation-tokens "$GEN" --num-trials "$TRIALS" "$@"; }

{
  echo "=== MTP characterization ==="
  echo "date:    $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host:    $(hostname)"
  echo "git:     $(git rev-parse --short HEAD 2>/dev/null || echo n/a) $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo n/a)"
  echo "model:   $MODEL"
  echo "runner:  $RUN"
  echo "mlx:     $($RUN -c 'import mlx.core as mx; print(mx.__version__)' 2>/dev/null || echo n/a)"
  echo "matrix:  prompts=[$PROMPTS] gen=$GEN trials=$TRIALS temp=0.0"
  echo

  for P in $PROMPTS; do
    echo "########## prompt-tokens=$P ##########"

    echo "----- [baseline] no MTP -----"
    bench --prompt-tokens "$P" || echo "FAILED: baseline p=$P"
    echo

    echo "----- [mtp D1] committed history (current behavior) -----"
    bench --prompt-tokens "$P" --draft-mtp --num-draft-tokens 1 || echo "FAILED: mtp D1 p=$P"
    echo

    echo "----- [mtp D2] committed history -----"
    bench --prompt-tokens "$P" --draft-mtp --num-draft-tokens 2 || echo "FAILED: mtp D2 p=$P"
    echo

    echo "----- [mtp D1 no-bonus] -----"
    bench --prompt-tokens "$P" --draft-mtp --num-draft-tokens 1 --draft-mtp-no-bonus \
      || echo "FAILED: mtp D1 no-bonus p=$P"
    echo
  done

  echo "=== done ==="
} 2>&1 | tee "$OUT"

echo
echo "Results written to: $OUT"
echo "Copy back with:  scp <server>:$(pwd)/$OUT ."
