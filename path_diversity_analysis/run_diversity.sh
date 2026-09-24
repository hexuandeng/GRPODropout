#!/usr/bin/env bash
# ============================================================================
# Main analysis runner: path diversity + answer diversity + LLM-judge
#
# Usage: edit the config section below, then run  bash run_diversity.sh
# ============================================================================
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}"

# ------------------------------- Config ------------------------------------ #

# Parent directory of the sampling outputs: it should contain <model>/<dataset>/test_*.jsonl
# Defaults to data_root/ next to this script; override via the ROOT env var, e.g.
#   ROOT=/path/to/sampling_outputs bash run_diversity.sh
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/data_root}"

# ==== Which models to analyze (allowlist, exact directory-name match) ====
# List the model directory names to run below, one per line; only listed ones are analyzed.
# Comment out a line with # to skip it temporarily. An empty array (all commented) errors out to remind you to pick.
MODELS=(
  "qwen3-4B-GRPO_balance-neg1"
  "qwen3-4B-GRPO_base"
  "qwen3-4B-GRPO_clip-cov"
  "qwen3-4B-GRPO_kl-cov"
  "qwen3-4B-DAPO_base"
  "qwen3-4B-DAPO_clip"
  # "qwen3-4B-GRPO_balance-neg"   # <- keep commented to skip
  # "Qwen3-4B"                    # <- base model, uncomment to include
)

# ==== Which datasets to analyze (allowlist) ====
# An empty array means run every dataset under each model.
DATASETS=(
  # "aime24"
  # "aime2025"
  # "amc23"
  # "hmmt25"
)

# Metrics to run: cosine answer (answer diversity) length (response length) judge_cluster judge_pairwise
# - Just answer + length (no model/API, fastest): METRICS="answer,length"
# - answer + cosine (needs bge-m3): METRICS="answer,cosine"
# - Include local-model judge (needs the endpoint configured below): METRICS="cosine,answer,length,judge_cluster,judge_pairwise"
METRICS="cosine,answer,length"

# Path-text views: think (only <think> reasoning) full (whole response) -- used by cosine
PATH_TEXT="think,full"
# Views used by the judge (think only to save cost)
JUDGE_PATH_TEXT="think,full"

# Which rollouts to evaluate diversity on:
#   all     = all rollouts (reflects overall exploration, including wrong attempts)
#   correct = only correct rollouts (reflects diversity of "valid solutions"; all-wrong questions are skipped)
#   both    = both criteria, with __all / __correct column suffixes (recommended)
ROLLOUT_SUBSET="both"

# Max questions per group (2~3 for a smoke test; leave empty for the full run)
MAX_QUESTIONS=""

# ---- Embedding model (for cosine). Defaults to models/bge-m3 under this directory ----
export EMBED_MODEL="${EMBED_MODEL:-${SCRIPT_DIR}/models/bge-m3}"
export EMBED_DEVICE="${EMBED_DEVICE:-cuda}"     # use cpu if no GPU

# ---- LLM judge endpoint (OpenAI-compatible). If empty, judge metrics are skipped automatically ----
# Cloud API:  JUDGE_BASE_URL=https://xxx/v1        JUDGE_MODEL=gpt-4o-mini
# Local vLLM: JUDGE_BASE_URL=http://127.0.0.1:8000/v1  JUDGE_API_KEY=EMPTY  JUDGE_MODEL=<vllm model name>
# Local Ollama: JUDGE_BASE_URL=http://127.0.0.1:11434/v1 JUDGE_API_KEY=ollama JUDGE_MODEL=qwen2.5
export JUDGE_BASE_URL="${JUDGE_BASE_URL:-}"
export JUDGE_API_KEY="${JUDGE_API_KEY:-}"       # for local deployments, any non-empty string, e.g. EMPTY
export JUDGE_MODEL="${JUDGE_MODEL:-}"

# python interpreter
PY="${PYTHON:-python}"

# --------------------------------------------------------------------------- #

if [[ ${#MODELS[@]} -eq 0 ]]; then
  echo "Error: MODELS is empty; list at least one model directory name in the config section." >&2
  exit 1
fi

# array -> comma-separated string
join_by() { local IFS="$1"; shift; echo "$*"; }
MODELS_CSV=$(join_by , "${MODELS[@]}")
DATASETS_CSV="all"
[[ ${#DATASETS[@]} -gt 0 ]] && DATASETS_CSV=$(join_by , "${DATASETS[@]}")

ARGS=(--root "${ROOT}" --models "${MODELS_CSV}" --datasets "${DATASETS_CSV}"
      --metrics "${METRICS}" --path_text "${PATH_TEXT}"
      --judge_path_text "${JUDGE_PATH_TEXT}" --rollout_subset "${ROLLOUT_SUBSET}")
[[ -n "${MAX_QUESTIONS}" ]] && ARGS+=(--max_questions "${MAX_QUESTIONS}")

echo "==> ROOT      = ${ROOT}"
echo "==> MODELS    = ${MODELS_CSV}"
echo "==> DATASETS  = ${DATASETS_CSV}"
echo "==> METRICS   = ${METRICS}"
echo "==> SUBSET    = ${ROLLOUT_SUBSET}"
echo "==> EMBED     = ${EMBED_MODEL} (device=${EMBED_DEVICE})"
if [[ -n "${JUDGE_BASE_URL}" ]]; then
  echo "==> JUDGE     = ${JUDGE_BASE_URL} model=${JUDGE_MODEL}"
else
  echo "==> JUDGE     = not configured, judge metrics will be skipped"
fi
echo

"${PY}" run_analysis.py "${ARGS[@]}"

echo
echo "Done. Results in ${SCRIPT_DIR}/outputs/summary/ and outputs/per_question/"
