#!/usr/bin/env bash
# Held-out OOD eval + training-consistent actor/entropy over ALL checkpoints
# (global_step_*) of ONE OR MORE models, across MULTIPLE held-out datasets.
#
# Datasets (3 groups, results saved separately) — all under eval/heldout/data/:
#   mmlu_pro : MMLU-Pro-100-nomath.jsonl   (choice, non-math OOD)
#   amc23    : AMC23_verl_val.parquet      (40 math questions, full AMC23)
#   dapo17k  : dapo17k_verl_val.parquet    (100 math questions)
#
# Output layout (per group kept separate):
#   ${OUT_ROOT}/<model_tag>/<data_name>/<global_step_x>/summary.json
#
# Two run modes (PARALLEL env var):
#   PARALLEL=1 (default): the 3 dataset groups run CONCURRENTLY, each pinned to
#                         its own GPU pair (GPU_GROUPS). 6 GPUs -> 2 per group.
#   PARALLEL=0          : the old behaviour — groups run sequentially, each using
#                         all of CUDA_VISIBLE_DEVICES.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ---- edit these to your environment ----
# One or more model roots (each expected to contain all_check/global_step_*).
# Provide them via the environment (space/newline separated):
#   MODEL_ROOTS_ENV="/path/a/all_check /path/b/all_check" bash run_heldout_all_checks.sh
MODEL_ROOTS=()
if [ -n "${MODEL_ROOTS_ENV:-}" ]; then
    read -r -a MODEL_ROOTS <<< "${MODEL_ROOTS_ENV}"
fi
if [ "${#MODEL_ROOTS[@]}" -eq 0 ]; then
    echo "Set MODEL_ROOTS_ENV=\"/path/a/all_check /path/b/all_check\"" >&2
    exit 1
fi

OUT_ROOT=${OUT_ROOT:-${SCRIPT_DIR}/eval_out/heldout}

# The 3 held-out groups: "data_name|data_file". All under eval/heldout/data/.
DATASETS=(
    "mmlu_pro|${SCRIPT_DIR}/data/MMLU-Pro-100-nomath.jsonl"
    "amc23|${SCRIPT_DIR}/data/AMC23_verl_val.parquet"
    "dapo17k|${SCRIPT_DIR}/data/dapo17k_verl_val.parquet"
)

# Parallel mode: which GPU(s) each dataset group above gets (index-aligned with
# DATASETS). Default: 6 cards, 2 per group. Override with e.g.
#   GPU_GROUPS_ENV="0,1 2,3 4,5"
GPU_GROUPS=("1" "2,3" "4,5")
if [ -n "${GPU_GROUPS_ENV:-}" ]; then
    read -r -a GPU_GROUPS <<< "${GPU_GROUPS_ENV}"
fi

PARALLEL=${PARALLEL:-1}

N_SAMPLING=${N_SAMPLING:-1}
TEMPERATURE=${TEMPERATURE:-1.0}
TOP_P=${TOP_P:-0.95}
TOP_K=${TOP_K:--1}
MAX_TOKENS=${MAX_TOKENS:-8192}
GPU_MEM=${GPU_MEM:-0.85}
# --------------------------------------

# model_tag: unique dir name for a model root (parent-of-all_check basename)
model_tag() {
    local root="$1" parent
    parent=$(basename "$(dirname "${root}")")
    if [ "${parent}" = "." ] || [ -z "${parent}" ]; then
        parent=$(basename "${root}")
    fi
    echo "${parent}"
}

list_ckpts() {  # echo checkpoints for a model root in numeric order
    ls -d "$1"/global_step_* 2>/dev/null | sort -t_ -k3 -n
}

# Run ONE model x ONE dataset group over all its checkpoints. Expects
# CUDA_VISIBLE_DEVICES to be already exported by the caller.
run_group() {
    local model_root="$1" tag="$2" data_name="$3" data_file="$4"
    local ckpts name out_dir
    mapfile -t ckpts < <(list_ckpts "${model_root}")
    for ckpt in "${ckpts[@]}"; do
        name=$(basename "${ckpt}")
        out_dir="${OUT_ROOT}/${tag}/${data_name}/${name}"
        if [ -f "${out_dir}/summary.json" ]; then
            echo "[skip] ${tag}/${data_name}/${name} already done (GPU=${CUDA_VISIBLE_DEVICES:-?})"
            continue
        fi
        echo "==== ${tag} | ${data_name} | ${name} | GPU=${CUDA_VISIBLE_DEVICES:-?} ===="
        python3 -u "${SCRIPT_DIR}/eval_heldout_entropy.py" \
            --model_name_or_path "${ckpt}" \
            --data_name "${data_name}" \
            --data_file "${data_file}" \
            --output_dir "${out_dir}" \
            --n_sampling "${N_SAMPLING}" \
            --temperature "${TEMPERATURE}" \
            --top_p "${TOP_P}" \
            --top_k "${TOP_K}" \
            --max_tokens "${MAX_TOKENS}" \
            --gpu_memory_utilization "${GPU_MEM}"
    done
}

LOG_DIR="${OUT_ROOT}/logs"
mkdir -p "${LOG_DIR}"

for MODEL_ROOT in "${MODEL_ROOTS[@]}"; do
    TAG=$(model_tag "${MODEL_ROOT}")
    echo "############################################################"
    echo "# MODEL: ${TAG}  (${MODEL_ROOT})   PARALLEL=${PARALLEL}"
    echo "############################################################"

    mapfile -t CKPTS < <(list_ckpts "${MODEL_ROOT}")
    if [ "${#CKPTS[@]}" -eq 0 ]; then
        echo "No global_step_* checkpoints found under ${MODEL_ROOT}, skipping." >&2
        continue
    fi
    echo "Found ${#CKPTS[@]} checkpoints:"
    printf '  %s\n' "${CKPTS[@]}"

    if [ "${PARALLEL}" = "1" ]; then
        # Launch the 3 dataset groups concurrently, each on its own GPU pair.
        pids=()
        plabels=()
        for i in "${!DATASETS[@]}"; do
            entry="${DATASETS[$i]}"
            DATA_NAME="${entry%%|*}"
            DATA_FILE="${entry#*|}"
            GPUS="${GPU_GROUPS[$i % ${#GPU_GROUPS[@]}]}"
            if [ ! -f "${DATA_FILE}" ]; then
                echo "[warn] data file missing for ${DATA_NAME}: ${DATA_FILE} -- skipping" >&2
                continue
            fi
            log="${LOG_DIR}/${TAG}__${DATA_NAME}.log"
            echo "[launch] ${DATA_NAME} on GPU ${GPUS} -> log: ${log}"
            (
                export CUDA_VISIBLE_DEVICES="${GPUS}"
                run_group "${MODEL_ROOT}" "${TAG}" "${DATA_NAME}" "${DATA_FILE}"
            ) > "${log}" 2>&1 &
            pids+=("$!")
            plabels+=("${DATA_NAME}@GPU${GPUS}")
        done
        # wait for all groups of this model before moving to the next model
        fail=0
        for j in "${!pids[@]}"; do
            if wait "${pids[$j]}"; then
                echo "[done] ${plabels[$j]}"
            else
                echo "[FAIL] ${plabels[$j]} (see log)" >&2
                fail=1
            fi
        done
        [ "${fail}" -eq 0 ] || echo "[warn] some groups failed for ${TAG}" >&2
    else
        # Sequential: each group uses all of CUDA_VISIBLE_DEVICES in turn.
        export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
        for entry in "${DATASETS[@]}"; do
            DATA_NAME="${entry%%|*}"
            DATA_FILE="${entry#*|}"
            if [ ! -f "${DATA_FILE}" ]; then
                echo "[warn] data file missing for ${DATA_NAME}: ${DATA_FILE} -- skipping" >&2
                continue
            fi
            echo "-------- dataset: ${DATA_NAME} --------"
            run_group "${MODEL_ROOT}" "${TAG}" "${DATA_NAME}" "${DATA_FILE}"
        done
    fi
done

echo ""
echo "==================== SUMMARY (all models x datasets x checkpoints) ===================="
printf '%-28s | %-10s | %-14s | %-10s | %-22s\n' \
    "model_tag" "dataset" "checkpoint" "acc_mean" "actor_entropy_token_mean"
for MODEL_ROOT in "${MODEL_ROOTS[@]}"; do
    TAG=$(model_tag "${MODEL_ROOT}")
    mapfile -t CKPTS < <(list_ckpts "${MODEL_ROOT}")
    for entry in "${DATASETS[@]}"; do
        DATA_NAME="${entry%%|*}"
        for ckpt in "${CKPTS[@]}"; do
            name=$(basename "${ckpt}")
            s="${OUT_ROOT}/${TAG}/${DATA_NAME}/${name}/summary.json"
            [ -f "${s}" ] || continue
            python3 - "$s" "$TAG" "$DATA_NAME" "$name" <<'PY'
import json, sys
s, tag, dn, name = sys.argv[1:5]
d = json.load(open(s))
print(f"{tag:<28} | {dn:<10} | {name:<14} | {d['acc_mean']:<10.4f} | {d['actor_entropy_token_mean']:<22.6f}")
PY
        done
    done
done
