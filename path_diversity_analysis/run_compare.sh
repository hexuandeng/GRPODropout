#!/usr/bin/env bash
# ============================================================================
# For the same (model, dataset), compare per question whether different sampling
# sizes (e.g. k32 vs k256) solve the same questions.
#
# Usage: set FILE_A / FILE_B (or DIR) below, then run  bash run_compare.sh
# ============================================================================
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}"

# ------------------------------- Config ------------------------------------ #

# Option 1: point to two jsonl files (env-driven, required unless DIR is set below).
FILE_A="${FILE_A:-}"
FILE_B="${FILE_B:-}"

# Labels for the two files (used in the header; leave empty to infer from _k<N>_ in the filename)
LABELS="${LABELS:-}"

# Option 2 (optional): if DIR is set, FILE_A/FILE_B are ignored and test_*_k*.jsonl inside DIR are auto-paired
DIR="${DIR:-}"

# Path to save the result JSON (leave empty to only print, not save)
SAVE="${SAVE:-${SCRIPT_DIR}/outputs/compare_solved.json}"

PY="${PYTHON:-python}"

# --------------------------------------------------------------------------- #

mkdir -p "${SCRIPT_DIR}/outputs"
ARGS=()
if [[ -n "${DIR}" ]]; then
  ARGS+=("${DIR}")
else
  : "${FILE_A:?Set FILE_A=/path/to/run_a.jsonl (or set DIR=/path/to/dataset_dir)}"
  : "${FILE_B:?Set FILE_B=/path/to/run_b.jsonl (or set DIR=/path/to/dataset_dir)}"
  ARGS+=("${FILE_A}" "${FILE_B}")
  [[ -n "${LABELS}" ]] && ARGS+=(--labels "${LABELS}")
fi
[[ -n "${SAVE}" ]] && ARGS+=(--save "${SAVE}")

"${PY}" compare_solved.py "${ARGS[@]}"
