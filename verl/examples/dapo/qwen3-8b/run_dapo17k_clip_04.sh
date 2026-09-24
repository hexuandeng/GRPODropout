set -x

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
DAPO_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
EXAMPLES_DIR=$(cd "${DAPO_DIR}/.." && pwd)
TTRL_DIR="${EXAMPLES_DIR}/ttrl"
ROOT_DIR=$(cd "${EXAMPLES_DIR}/.." && pwd)

if [ ! -f "${TTRL_DIR}/ttrl_reward.py" ]; then
    echo "Failed to locate ${TTRL_DIR}/ttrl_reward.py" >&2
    exit 1
fi

# -----------------------------
# Paths / Experiment Metadata
# -----------------------------
TRAIN_FILE=/root/yzh/data/dapo17k/dapo17k_verl_train.parquet
VAL_FILE=/root/yzh/data/AMC23/AMC23_verl_val.parquet
MODEL_PATH=/data/LLMs/Qwen3-8B
SAVE_DIR=/data/yzh/model/qwen3-8B-DAPO_clip_04/dapo17k
PROJECT_NAME=Math_Reasoning_DAPO
EXPERIMENT_NAME=DAPO_clip_dapo17k
ROLLOUT_LOG_DIR=${ROOT_DIR}/rollout_logs/qwen3-8B-DAPO_clip_04/dapo17k

# -----------------------------
# Data Hyperparameters
# -----------------------------
TRAIN_BATCH_SIZE=128
MAX_PROMPT_LENGTH=2048
MAX_RESPONSE_LENGTH=8192
FILTER_OVERLONG_PROMPTS=True
TRUNCATION_MODE=error
ENABLE_THINKING=True

# -----------------------------
# Rollout / Sampling Hyperparameters
# -----------------------------
ROLLOUT_N=8
ROLLOUT_TEMPERATURE=1.0
ROLLOUT_TOP_P=1.0
ROLLOUT_TOP_K=-1

VAL_TEMPERATURE=1.0
VAL_TOP_P=0.95
VAL_TOP_K=-1
VAL_N=16
VAL_DO_SAMPLE=True

# -----------------------------
# DAPO Clip-Higher Hyperparameters
# -----------------------------
# DAPO/LATR clip-higher baseline: epsilon_low=0.2, epsilon_high=0.28.
CLIP_RATIO_LOW=0.2
CLIP_RATIO_HIGH=0.4
CLIP_RATIO_C=10.0

# -----------------------------
# Optimization Hyperparameters
# -----------------------------
ACTOR_LR=1e-6
ACTOR_WEIGHT_DECAY=0.01
ACTOR_LR_WARMUP_STEPS_RATIO=0.0
ACTOR_WARMUP_STYLE=constant
PPO_MINI_BATCH_SIZE=64
PPO_MICRO_BATCH_SIZE_PER_GPU=1
ACTOR_PPO_EPOCHS=1
LOSS_AGG_MODE=token-mean

ENTROPY_COEFF=0
USE_KL_LOSS=False
KL_LOSS_COEF=0.0
KL_LOSS_TYPE=low_var_kl

# -----------------------------
# System / Throughput
# -----------------------------
CUDA_DEVICES=4,5,6,7
N_GPUS_PER_NODE=4
NNODES=1
ROLLOUT_GPU_MEMORY_UTILIZATION=0.45
ROLLOUT_TP_SIZE=1
ROLLOUT_LOGPROB_MICRO_BSZ_PER_GPU=2
REF_LOGPROB_MICRO_BSZ_PER_GPU=2

# -----------------------------
# Trainer
# -----------------------------
TEST_FREQ=10
SAVE_FREQ=11
TOTAL_EPOCHS=1
TOTAL_TRAINING_STEPS=110

source /root/yzh/verl/scripts/runtime_env.sh


CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} python3 -u -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files=${TRAIN_FILE} \
    data.val_files=${VAL_FILE} \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.filter_overlong_prompts=${FILTER_OVERLONG_PROMPTS} \
    data.truncation=${TRUNCATION_MODE} \
    +data.chat_template_kwargs.enable_thinking=${ENABLE_THINKING} \
    actor_rollout_ref.model.path=${MODEL_PATH} \
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
    actor_rollout_ref.actor.clip_ratio_low=${CLIP_RATIO_LOW} \
    actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO_HIGH} \
    actor_rollout_ref.actor.clip_ratio_c=${CLIP_RATIO_C} \
    actor_rollout_ref.actor.use_kl_loss=${USE_KL_LOSS} \
    actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF} \
    actor_rollout_ref.actor.kl_loss_type=${KL_LOSS_TYPE} \
    actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF} \
    actor_rollout_ref.actor.loss_agg_mode=${LOSS_AGG_MODE} \
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
    custom_reward_function.path=${TTRL_DIR}/ttrl_reward.py \
    custom_reward_function.name=compute_score_batch_gt_math \
    +custom_val_reward_function.path=${TTRL_DIR}/ttrl_reward.py \
    +custom_val_reward_function.name=compute_score_batch_gt_math \
    +custom_training_module.path=null \
    trainer.critic_warmup=0 \
    trainer.logger=console \
    trainer.rollout_data_dir=${ROLLOUT_LOG_DIR} \
    trainer.default_local_dir=${SAVE_DIR} \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=${TEST_FREQ} \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS} \
    "$@"
