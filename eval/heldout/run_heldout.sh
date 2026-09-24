#!/usr/bin/env bash
# Held-out OOD eval + training-consistent actor/entropy for ONE single model
# (a plain model dir, NOT an all_check/global_step_* tree) across the 3 held-out
# groups. Runs the groups SEQUENTIALLY, each using all of CUDA_VISIBLE_DEVICES
# (default: 2 cards).
#
# Datasets (results saved separately) — all under eval/heldout/data/:
#   mmlu_pro : MMLU-Pro-100-nomath.jsonl   (choice, non-math OOD)
#   amc23    : AMC23_verl_val.parquet      (40 math questions, full AMC23)
#   dapo17k  : dapo17k_verl_val.parquet    (100 math questions)
#
# Output layout:
#   ${OUT_ROOT}/<model_tag>/<data_name>/summary.json
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ---- edit these to your environment ----
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}   # 2 cards
MODEL_PATH=${MODEL_PATH:?"Set MODEL_PATH=/path/to/model dir"}
OUT_ROOT=${OUT_ROOT:-${SCRIPT_DIR}/eval_out/heldout}

N_SAMPLING=${N_SAMPLING:-1}
TEMPERATURE=${TEMPERATURE:-1.0}
TOP_P=${TOP_P:-0.95}
TOP_K=${TOP_K:--1}
MAX_TOKENS=${MAX_TOKENS:-8192}
GPU_MEM=${GPU_MEM:-0.85}
# --------------------------------------

# The 3 held-out groups: "data_name|data_file". All under eval/heldout/data/.
DATASETS=(
    "mmlu_pro|${SCRIPT_DIR}/data/MMLU-Pro-100-nomath.jsonl"
    "amc23|${SCRIPT_DIR}/data/AMC23_verl_val.parquet"
    "dapo17k|${SCRIPT_DIR}/data/dapo17k_verl_val.parquet"
)

TAG=$(basename "${MODEL_PATH%/}")

echo "############################################################"
echo "# MODEL: ${TAG}  (${MODEL_PATH})   GPU=${CUDA_VISIBLE_DEVICES}"
echo "############################################################"

for entry in "${DATASETS[@]}"; do
    DATA_NAME="${entry%%|*}"
    DATA_FILE="${entry#*|}"
    if [ ! -f "${DATA_FILE}" ]; then
        echo "[warn] data file missing for ${DATA_NAME}: ${DATA_FILE} -- skipping group" >&2
        continue
    fi
    out_dir="${OUT_ROOT}/${TAG}/${DATA_NAME}"
    if [ -f "${out_dir}/summary.json" ]; then
        echo "[skip] ${TAG}/${DATA_NAME} already done -> ${out_dir}/summary.json"
        continue
    fi
    echo "==================== ${TAG} | ${DATA_NAME} ===================="
    python3 -u "${SCRIPT_DIR}/eval_heldout_entropy.py" \
        --model_name_or_path "${MODEL_PATH}" \
        --data_name "${DATA_NAME}" \
        --data_file "${DATA_FILE}" \
        --output_dir "${out_dir}" \
        --n_sampling "${N_SAMPLING}" \
        --temperature "${TEMPERATURE}" \
        --top_p "${TOP_P}" \
        --top_k "${TOP_K}" \
        --max_tokens "${MAX_TOKENS}" \
        --gpu_memory_utilization "${GPU_MEM}"
done

echo ""
echo "==================== SUMMARY (${TAG}) ===================="
printf '%-10s | %-10s | %-22s\n' "dataset" "acc_mean" "actor_entropy_token_mean"
for entry in "${DATASETS[@]}"; do
    DATA_NAME="${entry%%|*}"
    s="${OUT_ROOT}/${TAG}/${DATA_NAME}/summary.json"
    [ -f "${s}" ] || continue
    python3 - "$s" "$DATA_NAME" <<'PY'
import json, sys
s, dn = sys.argv[1], sys.argv[2]
d = json.load(open(s))
print(f"{dn:<10} | {d['acc_mean']:<10.4f} | {d['actor_entropy_token_mean']:<22.6f}")
PY
done
