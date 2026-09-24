#!/usr/bin/env bash
# =============================================================================
# GRPO-dropout unified training launcher.
#
# Covariance/entropy-guided rollout-deletion selector on top of standard GRPO
# (verl). One template drives every experiment; per-experiment differences live
# in small config files under configs/<model>/<dataset>/<variant>.env.
#
# Usage:
#   bash train.sh [configs/.../variant.env] [extra hydra overrides ...]
#
# Any variable below can also be overridden directly from the environment, e.g.
#   MODEL_PATH=/path/to/model DATA_ROOT=/path/to/data bash train.sh configs/...
# =============================================================================
set -euo pipefail
set -x

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)          # examples/grpo_dropout
EXAMPLES_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)         # examples
VERL_ROOT=$(cd "${EXAMPLES_DIR}/.." && pwd)          # verl repo root
WORKSPACE_ROOT=$(cd "${VERL_ROOT}/.." && pwd)        # grpodropout root
ROOT_DIR="${VERL_ROOT}"

# Any leading arguments that are existing files are sourced as config, in order
# (later files win). Compose model + dataset + variant, e.g.:
#   bash train.sh configs/models/qwen3-4b.env configs/datasets/dapo17k.env \
#                 configs/variants/balance.env
while [ "$#" -ge 1 ] && [ -f "$1" ]; do
    # shellcheck disable=SC1090
    source "$1"
    shift
done

# -----------------------------
# Roots (relative by default; override via environment)
# -----------------------------
DATA_ROOT=${DATA_ROOT:-${WORKSPACE_ROOT}/data}
MODEL_ROOT=${MODEL_ROOT:-${WORKSPACE_ROOT}/models}
SAVE_ROOT=${SAVE_ROOT:-${WORKSPACE_ROOT}/checkpoints}

# -----------------------------
# Custom modules (co-located with this launcher)
# -----------------------------
SELECTOR_MODULE="${SCRIPT_DIR}/selector.py"
MATH_REWARD_MODULE="${SCRIPT_DIR}/reward.py"
MMLU_PRO_REWARD_MODULE="${SCRIPT_DIR}/mmlu_pro_reward.py"
MMLU_PRO_DATASET_MODULE="${SCRIPT_DIR}/mmlu_pro_dataset.py"

# PLACEHOLDER_APPEND

# -----------------------------
# Experiment metadata / paths
# -----------------------------
MODEL_PATH=${MODEL_PATH:-${MODEL_ROOT}/Qwen3-4B}
TRAIN_FILE=${TRAIN_FILE:-${DATA_ROOT}/dapo17k/dapo17k_verl_train.parquet}
VAL_FILE=${VAL_FILE:-${DATA_ROOT}/AMC23/AMC23_verl_val.parquet}
PROJECT_NAME=${PROJECT_NAME:-Math_Reasoning_GRPO}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-${MODEL_TAG:-model}-GRPO_${VARIANT_TAG:-base}_${DATASET_TAG:-dapo17k}}
SAVE_DIR=${SAVE_DIR:-${SAVE_ROOT}/${EXPERIMENT_NAME}}
ROLLOUT_LOG_DIR=${ROLLOUT_LOG_DIR:-${ROOT_DIR}/rollout_logs/${EXPERIMENT_NAME}}

# -----------------------------
# Reward mode: math (ground-truth) | mmlu_pro
# -----------------------------
REWARD_MODE=${REWARD_MODE:-math}
DATASET_ARGS=()
if [ "${REWARD_MODE}" = "mmlu_pro" ]; then
    REWARD_MODULE=${REWARD_MODULE:-${MMLU_PRO_REWARD_MODULE}}
    REWARD_NAME=${REWARD_NAME:-compute_score_batch_mmlu_pro}
    VAL_REWARD_NAME=${VAL_REWARD_NAME:-compute_score_batch_mmlu_pro}
    DATASET_ARGS=(
        "data.custom_cls.path=${MMLU_PRO_DATASET_MODULE}"
        data.custom_cls.name=MMLUProDataset
    )
else
    REWARD_MODULE=${REWARD_MODULE:-${MATH_REWARD_MODULE}}
    REWARD_NAME=${REWARD_NAME:-compute_score_batch_gt_math}
    VAL_REWARD_NAME=${VAL_REWARD_NAME:-compute_score_batch_gt_math}
fi

# -----------------------------
# Data / sampling hyperparameters
# -----------------------------
ADV_ESTIMATOR=${ADV_ESTIMATOR:-grpo}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-8192}
FILTER_OVERLONG_PROMPTS=${FILTER_OVERLONG_PROMPTS:-True}
TRUNCATION_MODE=${TRUNCATION_MODE:-error}
ENABLE_THINKING=${ENABLE_THINKING:-True}

N_VOTES_PER_PROMPT=${N_VOTES_PER_PROMPT:-8}
ROLLOUT_N=${ROLLOUT_N:-${N_VOTES_PER_PROMPT}}
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-1.0}
ROLLOUT_TOP_P=${ROLLOUT_TOP_P:-1.0}
ROLLOUT_TOP_K=${ROLLOUT_TOP_K:--1}
VAL_TEMPERATURE=${VAL_TEMPERATURE:-1.0}
VAL_TOP_P=${VAL_TOP_P:-0.95}
VAL_TOP_K=${VAL_TOP_K:--1}
VAL_N=${VAL_N:-1}
VAL_DO_SAMPLE=${VAL_DO_SAMPLE:-True}

# -----------------------------
# Rollout selector (grpo-dropout) knobs
# -----------------------------
ENABLE_SELECTOR=${ENABLE_SELECTOR:-True}
N_UPDATE_SAMPLES_PER_PROMPT=${N_UPDATE_SAMPLES_PER_PROMPT:-8}
SELECT_METHOD=${SELECT_METHOD:-random}
SELECT_RANDOM_SEED=${SELECT_RANDOM_SEED:-3407}
SELECT_POSITIVE_RATIO=${SELECT_POSITIVE_RATIO:-0.5}
ENTROPY_LOGPROB_BASIS=${ENTROPY_LOGPROB_BASIS:-avg}
SELECT_LOGPROB_BASIS=${SELECT_LOGPROB_BASIS:-${ENTROPY_LOGPROB_BASIS}}
RECOMPUTE_ADV_AFTER_SELECT=${RECOMPUTE_ADV_AFTER_SELECT:-True}
ADV_P_CENTER_MODE=${ADV_P_CENTER_MODE:-none}
SAVE_TOKEN_LOGPROBS=${SAVE_TOKEN_LOGPROBS:-False}
SAVE_TOKEN_ENTROPIES=${SAVE_TOKEN_ENTROPIES:-False}
SELECTOR_DEBUG_PRINT_GROUPS=${SELECTOR_DEBUG_PRINT_GROUPS:-32}
SELECTOR_DEBUG_PRINT_TRAJS_PER_GROUP=${SELECTOR_DEBUG_PRINT_TRAJS_PER_GROUP:-2}

# -----------------------------
# Optimization hyperparameters
# -----------------------------
ACTOR_LR=${ACTOR_LR:-1e-6}
ACTOR_WEIGHT_DECAY=${ACTOR_WEIGHT_DECAY:-0.01}
ACTOR_LR_WARMUP_STEPS_RATIO=${ACTOR_LR_WARMUP_STEPS_RATIO:-0.0}
ACTOR_WARMUP_STYLE=${ACTOR_WARMUP_STYLE:-constant}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-64}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
ACTOR_PPO_EPOCHS=${ACTOR_PPO_EPOCHS:-1}
ACTOR_CLIP_RATIO=${ACTOR_CLIP_RATIO:-0.2}
LOSS_AGG_MODE=${LOSS_AGG_MODE:-token-mean}
ENTROPY_COEFF=${ENTROPY_COEFF:-0}
USE_KL_LOSS=${USE_KL_LOSS:-False}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.0}
KL_LOSS_TYPE=${KL_LOSS_TYPE:-low_var_kl}

# -----------------------------
# Policy-loss variant: vanilla | gspo | clip_cov | kl_cov | high_entropy
# -----------------------------
LOSS_MODE=${LOSS_MODE:-vanilla}
POLICY_LOSS_ARGS=()
case "${LOSS_MODE}" in
    vanilla) ;;
    gspo)
        POLICY_LOSS_ARGS=(actor_rollout_ref.actor.policy_loss.loss_mode=gspo) ;;
    clip_cov)
        POLICY_LOSS_ARGS=(
            actor_rollout_ref.actor.policy_loss.loss_mode=clip_cov
            actor_rollout_ref.actor.policy_loss.clip_cov_ratio=${CLIP_COV_RATIO:-0.0002}
            actor_rollout_ref.actor.policy_loss.clip_cov_lb=${CLIP_COV_LB:-1.0}
            actor_rollout_ref.actor.policy_loss.clip_cov_ub=${CLIP_COV_UB:-5.0}
        ) ;;
    kl_cov)
        POLICY_LOSS_ARGS=(
            actor_rollout_ref.actor.policy_loss.loss_mode=kl_cov
            actor_rollout_ref.actor.policy_loss.kl_cov_ratio=${KL_COV_RATIO:-0.002}
            actor_rollout_ref.actor.policy_loss.ppo_kl_coef=${PPO_KL_COEF:-1}
        ) ;;
    high_entropy)
        POLICY_LOSS_ARGS=(
            actor_rollout_ref.actor.policy_loss.loss_mode=high_entropy
            +actor_rollout_ref.actor.policy_loss.top_entropy_quantile=${TOP_ENTROPY_QUANTILE:-0.2}
        ) ;;
    *)
        echo "Unknown LOSS_MODE: ${LOSS_MODE}" >&2; exit 1 ;;
esac

# -----------------------------
# System / trainer
# -----------------------------
CUDA_DEVICES=${CUDA_DEVICES:-0,1,2,3}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-4}
NNODES=${NNODES:-1}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.45}
ROLLOUT_TP_SIZE=${ROLLOUT_TP_SIZE:-1}
ROLLOUT_LOGPROB_MICRO_BSZ_PER_GPU=${ROLLOUT_LOGPROB_MICRO_BSZ_PER_GPU:-2}
REF_LOGPROB_MICRO_BSZ_PER_GPU=${REF_LOGPROB_MICRO_BSZ_PER_GPU:-2}
TEST_FREQ=${TEST_FREQ:-20}
SAVE_FREQ=${SAVE_FREQ:-50}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
LOGGER=${LOGGER:-console}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True}

# Optional runtime-env hook (Ray, network, etc.); skipped if absent.
RUNTIME_ENV=${RUNTIME_ENV:-${VERL_ROOT}/scripts/runtime_env.sh}
if [ -f "${RUNTIME_ENV}" ]; then
    # shellcheck disable=SC1090
    source "${RUNTIME_ENV}"
fi

# -----------------------------
# Selector overrides (only when enabled)
# -----------------------------
SELECTOR_ARGS=()
if [ "${ENABLE_SELECTOR}" = "True" ] || [ "${ENABLE_SELECTOR}" = "true" ]; then
    SELECTOR_ARGS=(
        "+custom_training_module.path=${SELECTOR_MODULE}"
        +custom_training_module.name=build_grpo_dropout_selection
        +custom_training_module.kwargs.n_update_samples_per_prompt=${N_UPDATE_SAMPLES_PER_PROMPT}
        +custom_training_module.kwargs.select_method=${SELECT_METHOD}
        +custom_training_module.kwargs.select_positive_ratio=${SELECT_POSITIVE_RATIO}
        +custom_training_module.kwargs.select_random_seed=${SELECT_RANDOM_SEED}
        +custom_training_module.kwargs.entropy_logprob_basis=${ENTROPY_LOGPROB_BASIS}
        +custom_training_module.kwargs.select_logprob_basis=${SELECT_LOGPROB_BASIS}
        +custom_training_module.kwargs.recompute_adv_after_select=${RECOMPUTE_ADV_AFTER_SELECT}
        +custom_training_module.kwargs.adv_p_center_mode=${ADV_P_CENTER_MODE}
        +custom_training_module.kwargs.debug_print_groups=${SELECTOR_DEBUG_PRINT_GROUPS}
        +custom_training_module.kwargs.debug_print_trajs_per_group=${SELECTOR_DEBUG_PRINT_TRAJS_PER_GROUP}
        +custom_training_module.kwargs.save_token_logprobs=${SAVE_TOKEN_LOGPROBS}
        +custom_training_module.kwargs.save_token_entropies=${SAVE_TOKEN_ENTROPIES}
    )
else
    SELECTOR_ARGS=(+custom_training_module.path=null)
fi

CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} python3 -u -m verl.trainer.main_ppo \
    algorithm.adv_estimator=${ADV_ESTIMATOR} \
    algorithm.use_kl_in_reward=False \
    "data.train_files=${TRAIN_FILE}" \
    "data.val_files=${VAL_FILE}" \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.filter_overlong_prompts=${FILTER_OVERLONG_PROMPTS} \
    data.truncation=${TRUNCATION_MODE} \
    +data.chat_template_kwargs.enable_thinking=${ENABLE_THINKING} \
    "${DATASET_ARGS[@]}" \
    "actor_rollout_ref.model.path=${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR} \
    actor_rollout_ref.actor.optim.weight_decay=${ACTOR_WEIGHT_DECAY} \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=${ACTOR_LR_WARMUP_STEPS_RATIO} \
    actor_rollout_ref.actor.optim.warmup_style=${ACTOR_WARMUP_STYLE} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((${MAX_PROMPT_LENGTH}+${MAX_RESPONSE_LENGTH})) \
    actor_rollout_ref.actor.ppo_epochs=${ACTOR_PPO_EPOCHS} \
    actor_rollout_ref.actor.clip_ratio=${ACTOR_CLIP_RATIO} \
    actor_rollout_ref.actor.use_kl_loss=${USE_KL_LOSS} \
    actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF} \
    actor_rollout_ref.actor.kl_loss_type=${KL_LOSS_TYPE} \
    actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF} \
    actor_rollout_ref.actor.loss_agg_mode=${LOSS_AGG_MODE} \
    "${POLICY_LOSS_ARGS[@]}" \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP_SIZE} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${ROLLOUT_LOGPROB_MICRO_BSZ_PER_GPU} \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION} \
    actor_rollout_ref.rollout.max_model_len=$((${MAX_PROMPT_LENGTH}+${MAX_RESPONSE_LENGTH})) \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((${MAX_PROMPT_LENGTH}+${MAX_RESPONSE_LENGTH})) \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.temperature=${ROLLOUT_TEMPERATURE} \
    actor_rollout_ref.rollout.top_p=${ROLLOUT_TOP_P} \
    actor_rollout_ref.rollout.top_k=${ROLLOUT_TOP_K} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMPERATURE} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${VAL_TOP_P} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${VAL_TOP_K} \
    actor_rollout_ref.rollout.val_kwargs.n=${VAL_N} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=${VAL_DO_SAMPLE} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${REF_LOGPROB_MICRO_BSZ_PER_GPU} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.reward_manager=batch \
    "custom_reward_function.path=${REWARD_MODULE}" \
    custom_reward_function.name=${REWARD_NAME} \
    "+custom_val_reward_function.path=${REWARD_MODULE}" \
    +custom_val_reward_function.name=${VAL_REWARD_NAME} \
    "${SELECTOR_ARGS[@]}" \
    trainer.critic_warmup=0 \
    trainer.logger=${LOGGER} \
    "trainer.rollout_data_dir=${ROLLOUT_LOG_DIR}" \
    "trainer.default_local_dir=${SAVE_DIR}" \
    "trainer.project_name=${PROJECT_NAME}" \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=${TEST_FREQ} \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.val_before_train=${VAL_BEFORE_TRAIN} \
    "$@"
