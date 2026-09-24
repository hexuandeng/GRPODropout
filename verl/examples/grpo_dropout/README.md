# GRPO-Dropout

Covariance / entropy-guided **rollout deletion** on top of standard GRPO in
[verl](https://github.com/volcengine/verl). Instead of updating on every sampled
rollout, a selector drops a subset of each prompt group's rollouts before the
policy update, steering the group covariance between advantage and log-prob to
prevent entropy collapse.

The training reward is the **ground-truth** math grader
(`compute_score_batch_gt_math`); MMLU-Pro experiments use a multiple-choice
grader. Training and validation use the same reward.

## Layout

```
grpo_dropout/
  reward.py             # ground-truth math graders (train + val reward)
  mmlu_pro_reward.py    # MMLU-Pro multiple-choice reward
  mmlu_pro_dataset.py   # MMLU-Pro dataset class (data.custom_cls)
  selector.py           # build_grpo_dropout_selection (loaded as custom_training_module)
  selector_methods.py   # individual select_* strategies
  train.sh              # single parameterized launcher
  configs/
    models/   {qwen3-1.7b,qwen3-4b,qwen3-8b,dpsk-1.5b}.env
    datasets/ {dapo17k,mmlu_pro}.env
    variants/ {base,balance,push,...}.env
```

## Quick start

Point the launcher at your data/model roots (relative defaults shown), then
compose a model + dataset + variant config:

```bash
export DATA_ROOT=/path/to/data       # default: <repo>/../data
export MODEL_ROOT=/path/to/models    # default: <repo>/../models
export SAVE_ROOT=/path/to/checkpoints

bash train.sh \
  configs/models/qwen3-4b.env \
  configs/datasets/dapo17k.env \
  configs/variants/balance.env
```

Leading arguments that are existing files are sourced in order (later wins);
anything after them is passed through to Hydra, e.g.
`... configs/variants/entropy_reg.env ENTROPY_COEFF=0.02 trainer.total_epochs=2`.
Every variable in `train.sh` can also be overridden from the environment.

## Variants (config table)

Selector-based variants (`ENABLE_SELECTOR=True`) differ mainly in `SELECT_METHOD`
and `ADV_P_CENTER_MODE`:

| Variant | SELECT_METHOD | ADV_P_CENTER_MODE | Notes |
|---|---|---|---|
| base | random | none | keep all rollouts (baseline) |
| balance | balanced_push_zero | weighted_center | main method |
| push | positive_push_topk | weighted_center | |
| negpush | negative_push_topk | weighted_center | |
| threshold | balanced_push_threshold | weighted_center | step-scheduled covariance threshold |
| balance_neg | balanced_push_zero_add_negpush_negadv | weighted_center | |
| balance_pos | balanced_push_zero_add_negpush_posadv | weighted_center | |
| balance_drop_pos | balanced_push_zero_drop_negpush_posadv | weighted_center | |
| balance_global_mean | balanced_push_centered_global_mean | weighted_center | |
| balance_neg_no_readv | balanced_push_zero_add_negpush_negadv | none | no adv recompute/recenter |
| entropy_reg | random | none | + `ENTROPY_COEFF` (0.01/0.02/0.03) |

Policy-loss baselines (`ENABLE_SELECTOR=False`, no rollout deletion):

| Variant | LOSS_MODE | Extra knobs |
|---|---|---|
| gspo | gspo | — |
| clip_cov | clip_cov | `CLIP_COV_RATIO/LB/UB` |
| kl_cov | kl_cov | `KL_COV_RATIO`, `PPO_KL_COEF` |
| high_entropy | high_entropy | `TOP_ENTROPY_QUANTILE` (0.2) — "Beyond the 80/20 Rule" |

Combine any variant with any `configs/models/*.env` and `configs/datasets/*.env`.
`EXPERIMENT_NAME` defaults to `<MODEL_TAG>-GRPO_<VARIANT_TAG>_<DATASET_TAG>`.
