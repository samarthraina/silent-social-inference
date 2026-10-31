#!/usr/bin/env bash
# Scientific runner only: never starts, stops, or destroys cloud instances.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
RESULT_ROOT="${1:?Usage: bash run_pilot_both.sh /absolute/path/to/results}"
mkdir -p "$RESULT_ROOT"
RESULT_ROOT="$(cd "$RESULT_ROOT" && pwd)"
for MODEL_KEY in qwen llama; do
  for STAGE_KEY in engineering pilot; do
    RUN_DIR="$RESULT_ROOT/${MODEL_KEY}_${STAGE_KEY}"
    RUN_ARGS=(python run.py --plan frozen/plan.json --stage "$STAGE_KEY" --model "$MODEL_KEY"
              --out "$RUN_DIR" --max-seconds 14400)
    if [[ -f "$RUN_DIR/metadata.json" ]]; then RUN_ARGS+=(--resume); fi
    if [[ "$STAGE_KEY" == pilot ]]; then RUN_ARGS+=(--engineering "$RESULT_ROOT/${MODEL_KEY}_engineering"); fi
    "${RUN_ARGS[@]}"
  done
done
ANALYSIS_PATH="$RESULT_ROOT/pilot_summary.json"
if [[ ! -f "$ANALYSIS_PATH" ]]; then
  python analyze.py --run "$RESULT_ROOT/qwen_pilot" --run "$RESULT_ROOT/llama_pilot" --out "$ANALYSIS_PATH"
fi
printf '%s\n' "Both pilots complete: $RESULT_ROOT" "Validation has not run. Back up results. The instance is still rented."
