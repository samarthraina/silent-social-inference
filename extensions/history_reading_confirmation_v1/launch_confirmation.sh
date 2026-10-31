#!/usr/bin/env bash
# Run only after actual author exposure information has been supplied.
set -euo pipefail
REVIEW_FILE="${1:?Usage: bash launch_confirmation.sh /absolute/review.json /absolute/locks /absolute/results}"
LOCK_DIR="${2:?Missing locks directory}"
RESULT_DIR="${3:?Missing results directory}"
CONFIRM_VENV="${CONFIRM_VENV:-/workspace/generation-path-venv}"
if [[ ! -f "$CONFIRM_VENV/bin/activate" ]]; then
  printf '%s\n' 'The tested Python environment is missing. Create an isolated venv and install requirements-gpu.txt, then set CONFIRM_VENV to that path.' >&2
  exit 2
fi
source "$CONFIRM_VENV/bin/activate"
export HF_HOME="${HF_HOME:-/workspace/hf-cache}"
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ ! -f "$LOCK_DIR/lock.json" ]]; then
  python freeze_confirmation.py --audit "$REVIEW_FILE" --out "$LOCK_DIR"
else
  if ! cmp -s "$REVIEW_FILE" "$LOCK_DIR/exposure_audit.json"; then
    printf '%s\n' 'The exposure review differs from the frozen lock. Do not mix review versions into a started confirmation.' >&2
    exit 2
  fi
fi
exec bash run_confirmation_both.sh "$LOCK_DIR" "$RESULT_DIR"
