#!/usr/bin/env bash
# Requires an author-reviewed, frozen lock. Does not rent/stop/destroy any instance.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
LOCK_DIR="${1:?Usage: bash run_confirmation_both.sh /absolute/locks /absolute/results}"
RESULT_DIR="${2:?Usage: bash run_confirmation_both.sh /absolute/locks /absolute/results}"
mkdir -p "$RESULT_DIR"
for MODEL_KEY in qwen llama; do
  TARGET_DIR="$RESULT_DIR/$MODEL_KEY"
  RUN_ARGS=(python confirm.py --locks "$LOCK_DIR" --model "$MODEL_KEY" --out "$TARGET_DIR" --max-seconds 10800)
  if [[ -f "$TARGET_DIR/metadata.json" ]]; then RUN_ARGS+=(--resume); fi
  "${RUN_ARGS[@]}"
done
if [[ ! -f "$RESULT_DIR/confirmation_summary.json" ]]; then
  python summarize_confirmation.py --run "$RESULT_DIR/qwen" --run "$RESULT_DIR/llama" --out "$RESULT_DIR/confirmation_summary.json"
fi
printf '%s\n' 'Both confirmations complete. Back up the result directory. The GPU instance is still rented.'
