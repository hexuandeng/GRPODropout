#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EVAL_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
PROJECT_DIR=$(cd "${EVAL_DIR}/.." && pwd)

# Usage examples:
#   DATASETS="aime24,aime2025,amc23,hmmt25" N_SAMPLING=32 GPUS=0,1 bash run_sampling_eval.sh


MODELS=${MODELS:?"Set MODELS=/path/ckpt_a,/path/ckpt_b (comma-separated checkpoint paths)"}
DATASETS=${DATASETS:-"amc23,aime24,aime2025,hmmt25"}
GPUS=${GPUS:-"7"}
PROMPT=${PROMPT:-"qwen"}
MAX_TOKENS=${MAX_TOKENS:-8192}
SEED=${SEED:-3407}
GPU_UTIL=${GPU_UTIL:-0.90}
GRADE_MODE=${GRADE_MODE:-paper}
ENABLE_THINKING=${ENABLE_THINKING:-true}
USE_VERL_PROMPT=${USE_VERL_PROMPT:-false}
USE_VERL_REWARD=${USE_VERL_REWARD:-true}
USE_MARIO_MATH=${USE_MARIO_MATH:-false}
DISABLE_MARIO_MATH=${DISABLE_MARIO_MATH:-true}
MARIO_EVAL_PATH=${MARIO_EVAL_PATH:-"${PROJECT_DIR}/MARIO_EVAL"}
REPEAT_REQUESTS=${REPEAT_REQUESTS:-true}
TOP_K=${TOP_K:-20}

# SCOPE-RL Table 6 evaluation setting for Pass@k.
TEMPERATURE=${TEMPERATURE:-0.6}
TOP_P=${TOP_P:-0.95}
N_SAMPLING=${N_SAMPLING:-32}
K=${K:-${N_SAMPLING}}

OUTPUT_ROOT=${OUTPUT_ROOT:-"${EVAL_DIR}/eval_results/sampling_ouutputs_8k_qwen_test11"}
COMPLETIONS_ROOT=${COMPLETIONS_ROOT:-"${EVAL_DIR}/eval_results/sampling_outputs_8k_qwen_test11"}

IFS=',' read -ra MODEL_LIST <<< "${MODELS}"

if [[ "${DISABLE_MARIO_MATH}" != "true" && ( "${GRADE_MODE}" == "paper" || "${USE_MARIO_MATH}" == "true" ) ]]; then
  if [[ ! -d "${MARIO_EVAL_PATH}/math_evaluation" || ! -d "${MARIO_EVAL_PATH}/latex2sympy" ]]; then
    echo "MARIO math evaluator not found under ${MARIO_EVAL_PATH}"
    echo "Expected minimal layout: ${MARIO_EVAL_PATH}/math_evaluation and ${MARIO_EVAL_PATH}/latex2sympy"
    exit 1
  fi
fi

EXTRA_ARGS=()
[[ "${USE_VERL_PROMPT}" == "true" ]] && EXTRA_ARGS+=(--use_verl_prompt)
[[ "${USE_VERL_REWARD}" == "true" ]] && EXTRA_ARGS+=(--use_verl_reward)
[[ "${USE_MARIO_MATH}" == "true" ]] && EXTRA_ARGS+=(--use_mario_math)
[[ "${DISABLE_MARIO_MATH}" == "true" ]] && EXTRA_ARGS+=(--disable_mario_math)
[[ "${DISABLE_MARIO_MATH}" != "true" && -n "${MARIO_EVAL_PATH}" ]] && EXTRA_ARGS+=(--mario_eval_path "${MARIO_EVAL_PATH}")
[[ "${ENABLE_THINKING}" == "true" ]] && EXTRA_ARGS+=(--enable_thinking)
[[ "${REPEAT_REQUESTS}" == "true" ]] && EXTRA_ARGS+=(--repeat_requests)

for MODEL in "${MODEL_LIST[@]}"; do
  MODEL_NAME=$(basename "${MODEL%/}")
  CUDA_VISIBLE_DEVICES="${GPUS}" \
  python "${EVAL_DIR}/eval.py" \
    --model_name_or_path "${MODEL}" \
    --datasets "${DATASETS}" \
    --data_dir "${EVAL_DIR}/data" \
    --grade_mode "${GRADE_MODE}" \
    --prompt_type "${PROMPT}" \
    --temperature "${TEMPERATURE}" \
    --top_p "${TOP_P}" \
    --top_k "${TOP_K}" \
    --n_sampling "${N_SAMPLING}" \
    --k "${K}" \
    --max_tokens "${MAX_TOKENS}" \
    --seed "${SEED}" \
    --surround_with_messages \
    --output_dir "${OUTPUT_ROOT}/${MODEL_NAME}" \
    --completions_save_dir "${COMPLETIONS_ROOT}/${MODEL_NAME}" \
    --gpu_memory_utilization "${GPU_UTIL}" \
    "${EXTRA_ARGS[@]}"
done
