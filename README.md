# GRPODropout: Less is More for Online Reinforcement Learning Rollouts

Code release for our study of **covariance / entropy-guided rollout deletion**
on top of GRPO. During RL fine-tuning of LLMs, instead of updating on every
sampled rollout in a prompt group, a lightweight selector *drops* a subset of
rollouts before the policy update. Deleting rollouts along the advantage↔log-prob
covariance direction steers the group away from entropy collapse while keeping
the useful learning signal, improving exploration/exploitation trade-offs.

The method is built as a minimally-invasive extension of
[verl](https://github.com/volcengine/verl): the reward and the rollout selector
are plugged in through verl's `custom_reward_function` / `custom_training_module`
hooks, so the framework itself is essentially stock.

## Repository layout

```
GRPODropout/
├── verl/                              # verl fork (training framework)
│   ├── examples/grpo_dropout/         # ← our recipe
│   │   ├── reward.py                  # ground-truth math reward (train + val)
│   │   ├── mmlu_pro_reward.py         # MMLU-Pro reward + dataset class
│   │   ├── selector.py                # build_grpo_dropout_selection (rollout deletion)
│   │   ├── selector_methods.py        # individual selection strategies
│   │   ├── train.sh                   # single parameterized training launcher
│   │   ├── configs/                   # model / dataset / variant .env files
│   │   └── README.md                  # variants + how to run
│   ├── verl/trainer/ppo/              # integration points (selector call, high-entropy loss)
│   └── analyze/                       # post-hoc analysis & dashboard scripts
├── eval/                              # evaluation harness (greedy / sampling / held-out)
└── data/                             # dataset preparation scripts (data files not shipped)
```

## Quick start

1. Install verl and its dependencies (see `verl/README.md` / `verl/requirements.txt`).
2. Prepare datasets — the raw data files are **not** shipped; rebuild them with the
   scripts in [data/](data/) (see [data/README.md](data/README.md)).
3. Train, composing a model + dataset + method variant:

   ```bash
   cd verl/examples/grpo_dropout
   export DATA_ROOT=/path/to/data MODEL_ROOT=/path/to/models SAVE_ROOT=/path/to/checkpoints
   bash train.sh configs/models/qwen3-4b.env configs/datasets/dapo17k.env configs/variants/balance.env
   ```

   See [verl/examples/grpo_dropout/README.md](verl/examples/grpo_dropout/README.md)
   for the full list of variants (base / balance / push / threshold / … and the
   gspo / clip_cov / kl_cov / high_entropy baselines).
4. Evaluate trained checkpoints with the harness in [eval/](eval/):

   ```bash
   cd eval/scripts
   MODELS=/path/to/ckpt_a,/path/to/ckpt_b bash run_greedy_eval.sh
   ```

## Paths & configuration

All launchers derive the repository root from their own location and read data /
model / output roots from `DATA_ROOT`, `MODEL_ROOT`, `SAVE_ROOT` (with sensible
relative defaults). No absolute machine paths are hard-coded.
