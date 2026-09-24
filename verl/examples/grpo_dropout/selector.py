# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Selection and trajectory-statistics utilities for the GRPO-dropout experiment.

This module serves two goals:

1. Build per-trajectory / per-question statistics that are dumped into rollout JSONL.
2. Select which sampled trajectories are kept for the actor update.

The implementation keeps two parallel "log-probability basis" views:

`avglogprob`
    Treat `traj_avg_logprob` as the proxy for `log p_k`.
`sumlogprob`
    Treat `traj_sum_logprob` as the proxy for `log p_k`.

For each basis we log three related views:

`before`
    Quantities on the full prompt group before sample selection.
`selected_frozen`
    Quantities on the selected subset, but still compared against the original
    full-group entropy center.
`after_recentered`
    Quantities on the selected subset after recomputing the entropy center on
    that subset. This aligns with the empirical post-selection covariance view.

Important field families:

`traj_entropy_push_*`
    `A_k * (s_k - H_hat)` under one chosen basis.

`group_entropy_grad_proxy_*`
    Group-level Monte Carlo estimate of the entropy trend, implemented as the
    mean of the corresponding `traj_entropy_push_*` values.

`traj_entropy_grad_contrib_*`
    Per-trajectory share of the group entropy-trend estimate:

        traj_entropy_grad_contrib = traj_entropy_push / group_count

    Therefore these trajectory contributions sum back to the group-level
    gradient proxy.

`traj_entropy_push_weighted_*`
    Weight-preserving variant kept for future ablations:

        weight_k * A_k * (s_k - H_hat)

    where `weight_k = exp(logp_proxy_k)` under the chosen basis.

Legacy aliases such as `traj_entropy_gap`, `group_entropy_proxy`,
`group_entropy_grad_proxy_before_select`, etc. are still emitted and mapped to
the configured default basis so existing analysis scripts continue to run.

Selection methods:

`positive_push_topk`
    For each prompt group, keep only trajectories whose before-select
    `traj_entropy_push_before_*` is positive under the configured
    `entropy_logprob_basis`. If more than `n_update_samples_per_prompt` are
    positive, keep the top-k by push value. If none are positive, fall back to
    deterministic random sampling inside the group. This fallback is a
    practical safeguard against empty selected subsets.

`pseudo_label_ratio`
    For each prompt group, target a configurable positive/negative ratio based
    on the reward-side pseudo label (`score > 0` vs `score == 0`).

`push_sign_ratio`
    For each prompt group, target a configurable positive/negative ratio based
    on the sign of before-select `traj_entropy_push_before_*`. Zero-push
    trajectories are used only as backfill when one sign quota is exhausted.

`balanced_push_zero`
    For each prompt group, choose a subset whose before-select push sum is as
    close to zero as possible under the configured `entropy_logprob_basis`.
    This uses a practical greedy balancing heuristic on
    `traj_entropy_push_before_*`, then evaluates the actual post-selection
    trend using the existing `after_recentered` statistics.

`balanced_push_threshold`
    Start from all positive-push trajectories, then greedily add negative-push
    trajectories while the selected push sum stays strictly above
    `a(step) * positive_push_sum`. The threshold ratio `a(step)` decays to zero
    by the last training step, at which point this reduces to
    `balanced_push_zero`.
"""

from __future__ import annotations

import importlib.util
import math
from typing import Any
from pathlib import Path

import torch

_CURRENT_DIR = Path(__file__).resolve().parent
_METHODS_PATH = _CURRENT_DIR / "selector_methods.py"
_METHODS_SPEC = importlib.util.spec_from_file_location("grpo_dropout_selector_methods_runtime", _METHODS_PATH)
if _METHODS_SPEC is None or _METHODS_SPEC.loader is None:
    raise ImportError(f"Could not load selector methods module from '{_METHODS_PATH}'.")
_METHODS_MODULE = importlib.util.module_from_spec(_METHODS_SPEC)
_METHODS_SPEC.loader.exec_module(_METHODS_MODULE)

_basis_suffix = _METHODS_MODULE.basis_suffix
_select_indices = _METHODS_MODULE.select_indices


def _build_group_keys(extra_infos, uids, n: int) -> tuple[list[str], list[str | None]]:
    group_keys: list[str] = []
    question_ids: list[str | None] = []

    for i in range(n):
        question_id = None
        if extra_infos is not None and i < len(extra_infos):
            extra_info = extra_infos[i]
            if isinstance(extra_info, dict):
                question_id = extra_info.get("question_id") or extra_info.get("index")

        if question_id is not None:
            question_id = str(question_id)

        if uids is not None and i < len(uids):
            group_key = str(uids[i])
        elif question_id is not None:
            group_key = question_id
        else:
            group_key = str(i)

        group_keys.append(group_key)
        question_ids.append(question_id)

    return group_keys, question_ids


def _masked_row_sum(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(values.dtype)
    return (values * mask).sum(dim=-1)


def _masked_row_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(values.dtype)
    lengths = mask.sum(dim=-1).clamp_min(1.0)
    return (values * mask).sum(dim=-1) / lengths


def _masked_row_count(mask: torch.Tensor) -> torch.Tensor:
    return mask.to(torch.float32).sum(dim=-1)


def _covariance_and_contribs(x_values: list[float], y_values: list[float]) -> tuple[float, list[float]]:
    if not x_values:
        return 0.0, []

    mean_x = sum(x_values) / len(x_values)
    mean_y = sum(y_values) / len(y_values)
    contribs = [((x - mean_x) * (y - mean_y)) / len(x_values) for x, y in zip(x_values, y_values, strict=False)]
    return sum(contribs), contribs


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if not values:
        return 0.0
    mean_value = _mean(values)
    return math.sqrt(sum((value - mean_value) ** 2 for value in values) / len(values))


def _safe_exp(value: float) -> float:
    return float(math.exp(max(value, -745.0)))


def _to_python_list(values: Any, n: int) -> list[Any]:
    if values is None:
        return [None] * n
    if hasattr(values, "detach"):
        values = values.detach().cpu().tolist()
    elif hasattr(values, "tolist"):
        values = values.tolist()
    else:
        values = list(values)
    if len(values) != n:
        raise ValueError(f"Expected {n} values but received {len(values)}.")
    return list(values)


def _count_binary(values: list[Any], idxs: list[int], positive_if) -> tuple[int, int] | None:
    subset = [values[idx] for idx in idxs]
    if any(value is None for value in subset):
        return None
    positive = sum(1 for value in subset if positive_if(float(value)))
    return positive, len(subset) - positive


def _count_push_signs(push_values: list[float], idxs: list[int], eps: float = 1e-12) -> tuple[int, int, int]:
    subset = [float(push_values[idx]) for idx in idxs]
    positive = sum(1 for value in subset if value > eps)
    negative = sum(1 for value in subset if value < -eps)
    zero = len(subset) - positive - negative
    return positive, negative, zero


def _format_binary_counts(label: str, counts: tuple[int, int] | None) -> str:
    if counts is None:
        return f"{label}=na"
    positive, negative = counts
    return f"{label}=+{positive}/-{negative}"


def _format_traj_brief(idx: int, stats: dict[str, list[Any]], basis_suffix: str) -> str:
    return (
        f"[i={idx} tok={int(stats['traj_num_tokens'][idx])} "
        f"logp={stats[f'traj_logp_{basis_suffix}'][idx]:.4f} "
        f"surp={stats[f'traj_surprisal_{basis_suffix}'][idx]:.4f} "
        f"adv={stats['traj_adv'][idx]:.4f} "
        f"gap={stats[f'traj_entropy_gap_before_{basis_suffix}'][idx]:.4f} "
        f"push={stats[f'traj_entropy_push_before_{basis_suffix}'][idx]:.4f} "
        f"sel={int(bool(stats['selected_for_update'][idx]))}]"
    )


def _print_group_debug_summary(
    group_key: str,
    question_id: str | None,
    idxs: list[int],
    selected_group_indices: list[int],
    stats: dict[str, list[Any]],
    selection_meta: dict[str, Any],
    debug_print_trajs_per_group: int,
    debug_basis_suffix: str,
) -> None:
    rep = idxs[0]
    pseudo_before = _count_binary(stats["traj_pseudo_score"], idxs, lambda value: value > 0.0)
    pseudo_after = _count_binary(stats["traj_pseudo_score"], selected_group_indices, lambda value: value > 0.0)
    oracle_before = _count_binary(stats["traj_oracle_acc"], idxs, lambda value: value > 0.0)
    oracle_after = _count_binary(stats["traj_oracle_acc"], selected_group_indices, lambda value: value > 0.0)
    push_before = _count_push_signs(stats[f"traj_entropy_push_before_{debug_basis_suffix}"], idxs)
    push_after = _count_push_signs(stats[f"traj_entropy_push_before_{debug_basis_suffix}"], selected_group_indices)
    ratio_note = ""
    if "ratio_target_positive" in selection_meta:
        ratio_note = (
            f" target=+{int(selection_meta['ratio_target_positive'])}/-{int(selection_meta['ratio_target_negative'])}"
        )
    print(
        f"[GRPODropout][reward][group={group_key}][qid={question_id or '-'}] "
        f"size={int(stats['group_size'][rep])} "
        f"{_format_binary_counts('pseudo_before', pseudo_before)} "
        f"{_format_binary_counts('oracle_before', oracle_before)}"
    )
    print(
        f"[GRPODropout][selector][group={group_key}][qid={question_id or '-'}][method={stats['select_method'][rep]}]"
        f"[basis={debug_basis_suffix}] "
        f"selected={int(stats['group_selected_count'][rep])}{ratio_note} "
        f"{_format_binary_counts('pseudo_after', pseudo_after)} "
        f"{_format_binary_counts('oracle_after', oracle_after)} "
        f"push_before=+{push_before[0]}/-{push_before[1]}/0{push_before[2]} "
        f"push_after=+{push_after[0]}/-{push_after[1]}/0{push_after[2]} "
        f"token_H={stats['group_token_entropy_mean'][rep]:.4f} "
        f"H_before={stats[f'group_entropy_center_before_{debug_basis_suffix}'][rep]:.4f} "
        f"H_after={stats[f'group_entropy_center_after_{debug_basis_suffix}'][rep]:.4f} "
        f"logp_mean={stats[f'group_logp_mean_{debug_basis_suffix}'][rep]:.4f} "
        f"logp_std={stats[f'group_logp_std_{debug_basis_suffix}'][rep]:.4f} "
        f"adv_mean={stats['group_adv_mean'][rep]:.4f} adv_std={stats['group_adv_std'][rep]:.4f} "
        f"grad_before={stats[f'group_entropy_grad_proxy_before_{debug_basis_suffix}'][rep]:.4f} "
        f"grad_after={stats[f'group_entropy_grad_proxy_after_{debug_basis_suffix}'][rep]:.4f} "
        f"push_before={stats[f'group_entropy_push_mean_before_{debug_basis_suffix}'][rep]:.4f} "
        f"push_sel_frozen={stats[f'group_selected_entropy_push_mean_frozen_{debug_basis_suffix}'][rep]:.4f} "
        f"push_sel_recentered={stats[f'group_selected_entropy_push_mean_recentered_{debug_basis_suffix}'][rep]:.4f}"
    )

    top_k = max(int(debug_print_trajs_per_group), 0)
    if top_k <= 0:
        return

    preview_groups = [
        (
            "tail",
            sorted(idxs, key=lambda idx: stats[f"traj_entropy_gap_before_{debug_basis_suffix}"][idx], reverse=True)[
                :top_k
            ],
        ),
        (
            "head",
            sorted(idxs, key=lambda idx: stats[f"traj_entropy_gap_before_{debug_basis_suffix}"][idx])[:top_k],
        ),
        ("top_adv", sorted(idxs, key=lambda idx: stats["traj_adv"][idx], reverse=True)[:top_k]),
        (
            "selected",
            sorted(selected_group_indices, key=lambda idx: stats["traj_adv"][idx], reverse=True)[:top_k],
        ),
    ]
    for label, preview_idxs in preview_groups:
        if preview_idxs:
            print(
                f"  {label}: "
                + " ".join(_format_traj_brief(idx, stats=stats, basis_suffix=debug_basis_suffix) for idx in preview_idxs)
            )


def _init_basis_fields(stats: dict[str, list[Any]], n: int, suffix: str) -> None:
    stats[f"traj_logp_{suffix}"] = [0.0] * n
    stats[f"traj_surprisal_{suffix}"] = [0.0] * n
    stats[f"traj_weight_{suffix}"] = [0.0] * n
    stats[f"group_positive_push_count_before_{suffix}"] = [0] * n
    stats[f"group_positive_push_ratio_before_{suffix}"] = [0.0] * n

    stats[f"group_logp_mean_{suffix}"] = [0.0] * n
    stats[f"group_logp_std_{suffix}"] = [0.0] * n
    stats[f"group_entropy_center_before_{suffix}"] = [0.0] * n
    stats[f"group_entropy_center_after_{suffix}"] = [0.0] * n

    stats[f"traj_entropy_gap_before_{suffix}"] = [0.0] * n
    stats[f"traj_entropy_push_before_{suffix}"] = [0.0] * n
    stats[f"traj_entropy_push_weighted_before_{suffix}"] = [0.0] * n
    # This sample-mean decomposition is the main "single trajectory pushed the
    # group entropy trend by how much?" view used in analysis:
    # sum(traj_entropy_grad_contrib_*) == group_entropy_grad_proxy_*.
    stats[f"traj_entropy_grad_contrib_before_{suffix}"] = [0.0] * n
    stats[f"traj_entropy_grad_weighted_contrib_before_{suffix}"] = [0.0] * n

    stats[f"traj_entropy_gap_selected_frozen_{suffix}"] = [None] * n
    stats[f"traj_entropy_push_selected_frozen_{suffix}"] = [None] * n
    stats[f"traj_entropy_push_weighted_selected_frozen_{suffix}"] = [None] * n
    stats[f"traj_entropy_grad_contrib_selected_frozen_{suffix}"] = [None] * n
    stats[f"traj_entropy_grad_weighted_contrib_selected_frozen_{suffix}"] = [None] * n

    stats[f"traj_entropy_gap_after_recentered_{suffix}"] = [None] * n
    stats[f"traj_entropy_push_after_recentered_{suffix}"] = [None] * n
    stats[f"traj_entropy_push_weighted_after_recentered_{suffix}"] = [None] * n
    stats[f"traj_entropy_grad_contrib_after_{suffix}"] = [None] * n
    stats[f"traj_entropy_grad_weighted_contrib_after_{suffix}"] = [None] * n

    stats[f"group_entropy_push_mean_before_{suffix}"] = [0.0] * n
    stats[f"group_entropy_push_sum_before_{suffix}"] = [0.0] * n
    stats[f"group_entropy_push_weighted_mean_before_{suffix}"] = [0.0] * n
    stats[f"group_entropy_push_weighted_sum_before_{suffix}"] = [0.0] * n
    stats[f"group_selected_entropy_push_mean_frozen_{suffix}"] = [0.0] * n
    stats[f"group_selected_entropy_push_sum_frozen_{suffix}"] = [0.0] * n
    stats[f"group_selected_entropy_push_weighted_mean_frozen_{suffix}"] = [0.0] * n
    stats[f"group_selected_entropy_push_weighted_sum_frozen_{suffix}"] = [0.0] * n
    stats[f"group_selected_entropy_push_mean_recentered_{suffix}"] = [0.0] * n
    stats[f"group_selected_entropy_push_sum_recentered_{suffix}"] = [0.0] * n
    stats[f"group_selected_entropy_push_weighted_mean_recentered_{suffix}"] = [0.0] * n
    stats[f"group_selected_entropy_push_weighted_sum_recentered_{suffix}"] = [0.0] * n

    stats[f"group_entropy_grad_proxy_before_{suffix}"] = [0.0] * n
    stats[f"group_entropy_grad_proxy_after_{suffix}"] = [0.0] * n

    stats[f"group_cov_before_{suffix}"] = [0.0] * n
    stats[f"group_cov_after_{suffix}"] = [0.0] * n
    stats[f"traj_cov_contrib_before_{suffix}"] = [0.0] * n
    stats[f"traj_cov_contrib_after_{suffix}"] = [None] * n
    stats[f"traj_entropy_grad_cov_contrib_before_{suffix}"] = [0.0] * n
    stats[f"traj_entropy_grad_cov_contrib_after_{suffix}"] = [None] * n


def _set_legacy_aliases(stats: dict[str, list[Any]], basis_suffix: str) -> None:
    stats["traj_sum_logprob"] = stats["traj_logp_sumlogprob"]
    stats["traj_avg_logprob"] = stats["traj_logp_avglogprob"]
    stats["traj_avg_surprisal"] = stats["traj_surprisal_avglogprob"]
    stats["traj_prob"] = stats["traj_weight_sumlogprob"]
    stats["traj_prob_proxy"] = stats["traj_weight_avglogprob"]

    stats["group_entropy_proxy"] = stats[f"group_entropy_center_before_{basis_suffix}"]
    stats["traj_entropy_gap"] = stats[f"traj_entropy_gap_before_{basis_suffix}"]
    stats["traj_entropy_push_proxy"] = stats[f"traj_entropy_push_before_{basis_suffix}"]
    stats["traj_entropy_push_weighted_proxy"] = stats[f"traj_entropy_push_weighted_before_{basis_suffix}"]
    stats["group_entropy_push_mean"] = stats[f"group_entropy_push_mean_before_{basis_suffix}"]
    stats["group_entropy_push_weighted_mean"] = stats[f"group_entropy_push_weighted_mean_before_{basis_suffix}"]
    stats["group_selected_entropy_push_mean"] = stats[f"group_selected_entropy_push_mean_frozen_{basis_suffix}"]

    stats["group_cov_before_select"] = stats[f"group_cov_before_{basis_suffix}"]
    stats["group_cov_after_select"] = stats[f"group_cov_after_{basis_suffix}"]
    stats["group_entropy_grad_proxy_before_select"] = stats[f"group_entropy_grad_proxy_before_{basis_suffix}"]
    stats["group_entropy_grad_proxy_after_select"] = stats[f"group_entropy_grad_proxy_after_{basis_suffix}"]
    stats["traj_cov_contrib_before_select"] = stats[f"traj_cov_contrib_before_{basis_suffix}"]
    stats["traj_cov_contrib_after_select"] = stats[f"traj_cov_contrib_after_{basis_suffix}"]
    stats["traj_entropy_grad_contrib_before_select"] = stats[f"traj_entropy_grad_contrib_before_{basis_suffix}"]
    stats["traj_entropy_grad_contrib_after_select"] = stats[f"traj_entropy_grad_contrib_after_{basis_suffix}"]

    stats["group_avg_logprob_mean"] = stats["group_logp_mean_avglogprob"]
    stats["group_avg_logprob_std"] = stats["group_logp_std_avglogprob"]


def build_grpo_dropout_selection(
    batch,
    select_method: str = "random",
    n_update_samples_per_prompt: int | None = None,
    select_random_seed: int = 42,
    save_token_logprobs: bool = False,
    save_token_entropies: bool = False,
    debug_print_groups: int = 0,
    debug_print_trajs_per_group: int = 2,
    entropy_logprob_basis: str = "avg",
    select_logprob_basis: str | None = None,
    select_positive_ratio: float = 0.5,
    keep_legacy_aliases: bool = True,
    **kwargs,
):
    """Build rollout statistics and choose which samples update the actor.

    Args:
        batch:
            Current GRPO training batch. Requires `response_mask`, `old_log_probs`,
            `token_entropies`, and either `advantages_before_select` or
            `advantages`.
        select_method:
            Sample-selection method. Existing methods are kept;
            `topk_logprob` uses `select_logprob_basis`, `pseudo_label_ratio`
            uses reward pseudo labels, and `push_sign_ratio` uses push signs.
        n_update_samples_per_prompt:
            Number of responses kept per prompt group for the actor update.
        select_random_seed:
            Seed for deterministic group-local random sampling.
        save_token_logprobs:
            Whether to dump raw response-token log-prob sequences.
        save_token_entropies:
            Whether to dump raw response-token entropy sequences.
        debug_print_groups:
            Number of groups to print in the selector debug preview. Each
            preview emits a reward-style recap line followed by the selector
            summary line for easier log inspection.
        debug_print_trajs_per_group:
            Number of trajectories per debug preview bucket.
        entropy_logprob_basis:
            Which log-probability basis should drive the legacy no-suffix aliases.
            Supported values: `avg` / `avglogprob` / `sum` / `sumlogprob`.
        select_logprob_basis:
            Which log-probability basis should be used when
            `select_method=topk_logprob`. Defaults to `entropy_logprob_basis`.
        select_positive_ratio:
            Only used by `pseudo_label_ratio` and `push_sign_ratio`. This sets
            the target positive-class fraction inside each prompt group.
        keep_legacy_aliases:
            Emit old field names such as `group_entropy_proxy`,
            `traj_entropy_gap`, and `group_entropy_grad_proxy_before_select`.

    Logged field families:

    Base trajectory fields:
        `traj_num_tokens`
            Number of valid response tokens.
        `traj_adv`
            Sequence-level GRPO advantage copied across response tokens.
        `traj_logp_avglogprob`
            Mean response-token log-probability.
        `traj_logp_sumlogprob`
            Sum of response-token log-probabilities.
        `traj_surprisal_*`
            Negative log-probability under each basis.
        `traj_weight_*`
            `exp(traj_logp_*)` under each basis. Kept for future weight ablations.

    Before-select entropy trend fields under basis `*`:
        `group_entropy_center_before_*`
            Mean trajectory surprisal of the full group.
        `traj_entropy_gap_before_*`
            `traj_surprisal_* - group_entropy_center_before_*`.
        `traj_entropy_push_before_*`
            `traj_adv * traj_entropy_gap_before_*`. Positive means this sampled
            trajectory points in an entropy-increasing direction under the chosen
            basis.
        `traj_entropy_push_weighted_before_*`
            `traj_weight_* * traj_entropy_push_before_*`. This keeps an explicit
            weight term for future ablations and is closer to the proof's
            state-space term `p_k A_k (log(1/p_k) - H(p))`.
        `traj_entropy_grad_contrib_before_*`
            `traj_entropy_push_before_* / group_size`. This is the trajectory's
            share of the group entropy-gradient estimate, and the sum over the
            group equals `group_entropy_grad_proxy_before_*`.
        `group_entropy_grad_proxy_before_*`
            Mean of `traj_entropy_push_before_*`. This is the main Monte Carlo
            entropy trend estimator and equals `-Cov(A, logp_*)`.
        `group_positive_push_count_before_*`
            Number of trajectories in the group with positive before-select
            entropy push.
        `group_positive_push_ratio_before_*`
            Fraction of trajectories in the group with positive before-select
            entropy push.

    Selected-frozen fields:
        Same definitions as the before-select fields, but logged only on selected
        samples while keeping the original full-group entropy center fixed.

    After-recentered fields:
        Same definitions again, but recomputing the entropy center on the
        selected subset. This view aligns with the empirical post-selection
        covariance / gradient proxy.

    Debug covariance fields:
        `group_cov_*`, `traj_cov_contrib_*`, and
        `traj_entropy_grad_cov_contrib_*` keep the explicit covariance
        decomposition for inspection. They are secondary to the main push/grad
        fields above.
    """

    legacy_basis_suffix = _basis_suffix(entropy_logprob_basis)
    select_basis_suffix = _basis_suffix(select_logprob_basis or entropy_logprob_basis)
    select_positive_ratio = float(select_positive_ratio)
    if not 0.0 <= select_positive_ratio <= 1.0:
        raise ValueError(f"select_positive_ratio must be in [0, 1], got {select_positive_ratio!r}")

    response_mask = batch.batch["response_mask"].detach().cpu().bool()
    old_log_probs = batch.batch["old_log_probs"].detach().cpu()
    advantage_key = "advantages_before_select" if "advantages_before_select" in batch.batch.keys() else "advantages"
    advantages = batch.batch[advantage_key].detach().cpu()
    token_entropies = batch.batch["token_entropies"].detach().cpu()

    n = old_log_probs.shape[0]
    extra_infos = batch.non_tensor_batch.get("extra_info", [None] * n)
    uids = batch.non_tensor_batch.get("uid")
    group_keys, question_ids = _build_group_keys(extra_infos=extra_infos, uids=uids, n=n)

    traj_num_tokens = _masked_row_count(response_mask).tolist()
    traj_logp_sumlogprob = _masked_row_sum(old_log_probs, response_mask).tolist()
    traj_logp_avglogprob = _masked_row_mean(old_log_probs, response_mask).tolist()
    traj_surprisal_sumlogprob = [-float(v) for v in traj_logp_sumlogprob]
    traj_surprisal_avglogprob = [-float(v) for v in traj_logp_avglogprob]
    traj_weight_sumlogprob = [_safe_exp(float(v)) for v in traj_logp_sumlogprob]
    traj_weight_avglogprob = [_safe_exp(float(v)) for v in traj_logp_avglogprob]
    traj_entropy_mean = _masked_row_mean(token_entropies, response_mask).tolist()
    traj_entropy_sum = _masked_row_sum(token_entropies, response_mask).tolist()
    traj_adv_mean = _masked_row_mean(advantages, response_mask).tolist()
    traj_adv_sum = _masked_row_sum(advantages, response_mask).tolist()
    pseudo_score_source = batch.non_tensor_batch.get("score")
    if pseudo_score_source is None and "acc" in batch.batch.keys():
        pseudo_score_source = batch.batch["acc"]
    traj_pseudo_score = _to_python_list(pseudo_score_source, n=n)
    traj_oracle_acc = _to_python_list(batch.non_tensor_batch.get("oracle_acc"), n=n)

    stats: dict[str, list[Any]] = {
        "group_key": [str(group_key) for group_key in group_keys],
        "select_method": [select_method] * n,
        "advantage_key": [advantage_key] * n,
        "entropy_logprob_basis": [legacy_basis_suffix] * n,
        "select_logprob_basis": [select_basis_suffix] * n,
        "select_positive_ratio": [select_positive_ratio] * n,
        "selector_random_fallback": [False] * n,
        "selector_single_sign_only": [False] * n,
        "selector_ablation_mode": [""] * n,
        "selector_base_selected_for_update": [False] * n,
        "selector_ablation_added_for_update": [False] * n,
        "selector_ablation_added_count": [0] * n,
        "selector_ablation_negpush_unselected_count": [0] * n,
        "traj_num_tokens": [int(v) for v in traj_num_tokens],
        "traj_entropy_mean": [float(v) for v in traj_entropy_mean],
        "traj_entropy_sum": [float(v) for v in traj_entropy_sum],
        "traj_adv": [float(v) for v in traj_adv_mean],
        "traj_adv_mean": [float(v) for v in traj_adv_mean],
        "traj_adv_sum": [float(v) for v in traj_adv_sum],
        "traj_pseudo_score": [None if v is None else float(v) for v in traj_pseudo_score],
        "traj_oracle_acc": [None if v is None else float(v) for v in traj_oracle_acc],
        "group_size": [0] * n,
        "group_selected_count": [0] * n,
        "group_selected_ratio": [0.0] * n,
        "group_adv_mean": [0.0] * n,
        "group_adv_std": [0.0] * n,
        "group_adv_all_zero": [False] * n,
        "group_token_entropy_mean": [0.0] * n,
        "group_token_entropy_sum": [0.0] * n,
        "selected_for_update": [False] * n,
    }

    for suffix in ("avglogprob", "sumlogprob"):
        _init_basis_fields(stats, n=n, suffix=suffix)

    stats["traj_logp_avglogprob"] = [float(v) for v in traj_logp_avglogprob]
    stats["traj_logp_sumlogprob"] = [float(v) for v in traj_logp_sumlogprob]
    stats["traj_surprisal_avglogprob"] = [float(v) for v in traj_surprisal_avglogprob]
    stats["traj_surprisal_sumlogprob"] = [float(v) for v in traj_surprisal_sumlogprob]
    stats["traj_weight_avglogprob"] = [float(v) for v in traj_weight_avglogprob]
    stats["traj_weight_sumlogprob"] = [float(v) for v in traj_weight_sumlogprob]

    if any(qid is not None for qid in question_ids):
        stats["question_id"] = [qid for qid in question_ids]

    if save_token_logprobs:
        stats["token_log_probs"] = [old_log_probs[i][response_mask[i]].tolist() for i in range(n)]
    if save_token_entropies:
        stats["token_entropies"] = [token_entropies[i][response_mask[i]].tolist() for i in range(n)]

    groups: dict[str, list[int]] = {}
    for idx, group_key in enumerate(group_keys):
        groups.setdefault(group_key, []).append(idx)

    for group_key, idxs in groups.items():
        token_count = sum(int(response_mask[idx].sum().item()) for idx in idxs)
        group_token_entropy_sum = float(sum(stats["traj_entropy_sum"][idx] for idx in idxs))
        group_token_entropy_mean = group_token_entropy_sum / max(token_count, 1)

        group_size = len(idxs)
        before_adv = [stats["traj_adv"][idx] for idx in idxs]
        group_adv_mean = _mean(before_adv)
        group_adv_std = _std(before_adv)
        group_adv_all_zero = all(abs(value) <= 1e-12 for value in before_adv)

        for idx in idxs:
            stats["group_size"][idx] = int(group_size)
            stats["group_adv_mean"][idx] = float(group_adv_mean)
            stats["group_adv_std"][idx] = float(group_adv_std)
            stats["group_adv_all_zero"][idx] = bool(group_adv_all_zero)
            stats["group_token_entropy_mean"][idx] = float(group_token_entropy_mean)
            stats["group_token_entropy_sum"][idx] = float(group_token_entropy_sum)

        for suffix in ("avglogprob", "sumlogprob"):
            before_logp = [stats[f"traj_logp_{suffix}"][idx] for idx in idxs]
            before_surprisal = [stats[f"traj_surprisal_{suffix}"][idx] for idx in idxs]
            before_cov, before_cov_contribs = _covariance_and_contribs(before_adv, before_logp)

            center_before = _mean(before_surprisal)
            group_logp_mean = _mean(before_logp)
            group_logp_std = _std(before_logp)

            push_before_values: list[float] = []
            weighted_before_values: list[float] = []

            for local_pos, idx in enumerate(idxs):
                adv_value = stats["traj_adv"][idx]
                surprisal_value = stats[f"traj_surprisal_{suffix}"][idx]
                weight_value = stats[f"traj_weight_{suffix}"][idx]

                gap_before = surprisal_value - center_before
                push_before = adv_value * gap_before
                weighted_before = weight_value * push_before
                grad_contrib_before = push_before / max(group_size, 1)
                grad_weighted_contrib_before = weighted_before / max(group_size, 1)

                stats[f"group_logp_mean_{suffix}"][idx] = float(group_logp_mean)
                stats[f"group_logp_std_{suffix}"][idx] = float(group_logp_std)
                stats[f"group_entropy_center_before_{suffix}"][idx] = float(center_before)
                stats[f"traj_entropy_gap_before_{suffix}"][idx] = float(gap_before)
                stats[f"traj_entropy_push_before_{suffix}"][idx] = float(push_before)
                stats[f"traj_entropy_push_weighted_before_{suffix}"][idx] = float(weighted_before)
                stats[f"traj_entropy_grad_contrib_before_{suffix}"][idx] = float(grad_contrib_before)
                stats[f"traj_entropy_grad_weighted_contrib_before_{suffix}"][idx] = float(
                    grad_weighted_contrib_before
                )
                stats[f"group_cov_before_{suffix}"][idx] = float(before_cov)
                stats[f"traj_cov_contrib_before_{suffix}"][idx] = float(before_cov_contribs[local_pos])
                stats[f"traj_entropy_grad_cov_contrib_before_{suffix}"][idx] = float(-before_cov_contribs[local_pos])

                push_before_values.append(float(push_before))
                weighted_before_values.append(float(weighted_before))

            group_entropy_push_mean_before = _mean(push_before_values)
            group_entropy_push_sum_before = sum(push_before_values)
            group_entropy_push_weighted_mean_before = _mean(weighted_before_values)
            group_entropy_push_weighted_sum_before = sum(weighted_before_values)
            group_entropy_grad_proxy_before = group_entropy_push_mean_before
            group_positive_push_count_before = sum(1 for value in push_before_values if value > 0)
            group_positive_push_ratio_before = group_positive_push_count_before / max(group_size, 1)

            for idx in idxs:
                stats[f"group_entropy_push_mean_before_{suffix}"][idx] = float(group_entropy_push_mean_before)
                stats[f"group_entropy_push_sum_before_{suffix}"][idx] = float(group_entropy_push_sum_before)
                stats[f"group_entropy_push_weighted_mean_before_{suffix}"][idx] = float(
                    group_entropy_push_weighted_mean_before
                )
                stats[f"group_entropy_push_weighted_sum_before_{suffix}"][idx] = float(
                    group_entropy_push_weighted_sum_before
                )
                stats[f"group_entropy_grad_proxy_before_{suffix}"][idx] = float(group_entropy_grad_proxy_before)
                stats[f"group_positive_push_count_before_{suffix}"][idx] = int(group_positive_push_count_before)
                stats[f"group_positive_push_ratio_before_{suffix}"][idx] = float(group_positive_push_ratio_before)

    selected_indices: list[int] = []
    printed_groups = 0

    for group_key, idxs in groups.items():
        selected_group_indices, selection_meta = _select_indices(
            idxs=idxs,
            group_key=group_key,
            traj_adv_mean=stats["traj_adv_mean"],
            traj_logp_avglogprob=stats["traj_logp_avglogprob"],
            traj_logp_sumlogprob=stats["traj_logp_sumlogprob"],
            traj_entropy_mean=stats["traj_entropy_mean"],
            traj_pseudo_score=stats["traj_pseudo_score"],
            traj_entropy_push_before_avglogprob=stats["traj_entropy_push_before_avglogprob"],
            traj_entropy_push_before_sumlogprob=stats["traj_entropy_push_before_sumlogprob"],
            select_method=select_method,
            n_update_samples_per_prompt=n_update_samples_per_prompt,
            select_random_seed=select_random_seed,
            entropy_logprob_basis=legacy_basis_suffix,
            select_logprob_basis=select_basis_suffix,
            select_positive_ratio=select_positive_ratio,
            global_steps=kwargs.get("global_steps"),
            total_training_steps=kwargs.get("total_training_steps"),
            balance_target_ratio_start=float(kwargs.get("balance_target_ratio_start", 1.0)),
            balance_target_ratio_end=float(kwargs.get("balance_target_ratio_end", 0.0)),
            balance_target_ratio_schedule=str(kwargs.get("balance_target_ratio_schedule", "linear")),
            balance_push_center=float(kwargs.get("balance_push_center", 0.0)),
        )
        selected_group_set = set(selected_group_indices)
        base_selected_group_set = set(selection_meta.get("balance_base_selected_indices", selected_group_indices))
        ablation_added_group_set = set(selection_meta.get("balance_ablation_added_indices", []))
        selected_indices.extend(selected_group_indices)

        group_size = len(idxs)
        group_selected_count = len(selected_group_indices)
        group_selected_ratio = group_selected_count / max(group_size, 1)
        group_ablation_added_count = int(selection_meta.get("balance_ablation_added_count", 0))
        group_ablation_negpush_unselected_count = int(
            selection_meta.get("balance_ablation_negpush_unselected_count", 0)
        )
        group_ablation_mode = str(selection_meta.get("balance_ablation_mode", ""))

        for idx in idxs:
            stats["group_selected_count"][idx] = int(group_selected_count)
            stats["group_selected_ratio"][idx] = float(group_selected_ratio)
            stats["selected_for_update"][idx] = idx in selected_group_set
            stats["selector_random_fallback"][idx] = bool(selection_meta.get("balance_random_fallback", False))
            stats["selector_single_sign_only"][idx] = bool(selection_meta.get("balance_single_sign_only", False))
            stats["selector_ablation_mode"][idx] = group_ablation_mode
            stats["selector_base_selected_for_update"][idx] = idx in base_selected_group_set
            stats["selector_ablation_added_for_update"][idx] = idx in ablation_added_group_set
            stats["selector_ablation_added_count"][idx] = group_ablation_added_count
            stats["selector_ablation_negpush_unselected_count"][idx] = group_ablation_negpush_unselected_count

        for suffix in ("avglogprob", "sumlogprob"):
            after_adv = [stats["traj_adv"][idx] for idx in selected_group_indices]
            after_logp = [stats[f"traj_logp_{suffix}"][idx] for idx in selected_group_indices]
            after_cov, after_cov_contribs = _covariance_and_contribs(after_adv, after_logp)
            after_cov_contrib_map = {
                idx: contrib for idx, contrib in zip(selected_group_indices, after_cov_contribs, strict=False)
            }

            center_before = float(stats[f"group_entropy_center_before_{suffix}"][idxs[0]])
            center_after = _mean([stats[f"traj_surprisal_{suffix}"][idx] for idx in selected_group_indices])

            selected_push_frozen_values: list[float] = []
            selected_weighted_frozen_values: list[float] = []
            selected_push_recentered_values: list[float] = []
            selected_weighted_recentered_values: list[float] = []

            for idx in idxs:
                stats[f"group_entropy_center_after_{suffix}"][idx] = float(center_after)
                stats[f"group_cov_after_{suffix}"][idx] = float(after_cov)

                if idx not in selected_group_set:
                    continue

                adv_value = stats["traj_adv"][idx]
                surprisal_value = stats[f"traj_surprisal_{suffix}"][idx]
                weight_value = stats[f"traj_weight_{suffix}"][idx]
                push_frozen = float(stats[f"traj_entropy_push_before_{suffix}"][idx])
                weighted_frozen = float(stats[f"traj_entropy_push_weighted_before_{suffix}"][idx])
                gap_frozen = float(stats[f"traj_entropy_gap_before_{suffix}"][idx])
                grad_contrib_frozen = push_frozen / max(group_selected_count, 1)
                grad_weighted_contrib_frozen = weighted_frozen / max(group_selected_count, 1)

                gap_after = surprisal_value - center_after
                push_after = adv_value * gap_after
                weighted_after = weight_value * push_after
                grad_contrib_after = push_after / max(group_selected_count, 1)
                grad_weighted_contrib_after = weighted_after / max(group_selected_count, 1)

                stats[f"traj_entropy_gap_selected_frozen_{suffix}"][idx] = float(gap_frozen)
                stats[f"traj_entropy_push_selected_frozen_{suffix}"][idx] = float(push_frozen)
                stats[f"traj_entropy_push_weighted_selected_frozen_{suffix}"][idx] = float(weighted_frozen)
                stats[f"traj_entropy_grad_contrib_selected_frozen_{suffix}"][idx] = float(grad_contrib_frozen)
                stats[f"traj_entropy_grad_weighted_contrib_selected_frozen_{suffix}"][idx] = float(
                    grad_weighted_contrib_frozen
                )
                stats[f"traj_entropy_gap_after_recentered_{suffix}"][idx] = float(gap_after)
                stats[f"traj_entropy_push_after_recentered_{suffix}"][idx] = float(push_after)
                stats[f"traj_entropy_push_weighted_after_recentered_{suffix}"][idx] = float(weighted_after)
                stats[f"traj_entropy_grad_contrib_after_{suffix}"][idx] = float(grad_contrib_after)
                stats[f"traj_entropy_grad_weighted_contrib_after_{suffix}"][idx] = float(
                    grad_weighted_contrib_after
                )
                stats[f"traj_cov_contrib_after_{suffix}"][idx] = float(after_cov_contrib_map[idx])
                stats[f"traj_entropy_grad_cov_contrib_after_{suffix}"][idx] = float(-after_cov_contrib_map[idx])

                selected_push_frozen_values.append(float(push_frozen))
                selected_weighted_frozen_values.append(float(weighted_frozen))
                selected_push_recentered_values.append(float(push_after))
                selected_weighted_recentered_values.append(float(weighted_after))

            group_selected_entropy_push_mean_frozen = _mean(selected_push_frozen_values)
            group_selected_entropy_push_sum_frozen = sum(selected_push_frozen_values)
            group_selected_entropy_push_weighted_mean_frozen = _mean(selected_weighted_frozen_values)
            group_selected_entropy_push_weighted_sum_frozen = sum(selected_weighted_frozen_values)
            group_selected_entropy_push_mean_recentered = _mean(selected_push_recentered_values)
            group_selected_entropy_push_sum_recentered = sum(selected_push_recentered_values)
            group_selected_entropy_push_weighted_mean_recentered = _mean(selected_weighted_recentered_values)
            group_selected_entropy_push_weighted_sum_recentered = sum(selected_weighted_recentered_values)
            group_entropy_grad_proxy_after = group_selected_entropy_push_mean_recentered

            for idx in idxs:
                stats[f"group_selected_entropy_push_mean_frozen_{suffix}"][idx] = float(
                    group_selected_entropy_push_mean_frozen
                )
                stats[f"group_selected_entropy_push_sum_frozen_{suffix}"][idx] = float(
                    group_selected_entropy_push_sum_frozen
                )
                stats[f"group_selected_entropy_push_weighted_mean_frozen_{suffix}"][idx] = float(
                    group_selected_entropy_push_weighted_mean_frozen
                )
                stats[f"group_selected_entropy_push_weighted_sum_frozen_{suffix}"][idx] = float(
                    group_selected_entropy_push_weighted_sum_frozen
                )
                stats[f"group_selected_entropy_push_mean_recentered_{suffix}"][idx] = float(
                    group_selected_entropy_push_mean_recentered
                )
                stats[f"group_selected_entropy_push_sum_recentered_{suffix}"][idx] = float(
                    group_selected_entropy_push_sum_recentered
                )
                stats[f"group_selected_entropy_push_weighted_mean_recentered_{suffix}"][idx] = float(
                    group_selected_entropy_push_weighted_mean_recentered
                )
                stats[f"group_selected_entropy_push_weighted_sum_recentered_{suffix}"][idx] = float(
                    group_selected_entropy_push_weighted_sum_recentered
                )
                stats[f"group_entropy_grad_proxy_after_{suffix}"][idx] = float(group_entropy_grad_proxy_after)

        if debug_print_groups > 0 and printed_groups < debug_print_groups:
            printed_groups += 1
            _print_group_debug_summary(
                group_key=group_key,
                question_id=question_ids[idxs[0]],
                idxs=idxs,
                selected_group_indices=selected_group_indices,
                stats=stats,
                selection_meta=selection_meta,
                debug_print_trajs_per_group=debug_print_trajs_per_group,
                debug_basis_suffix=legacy_basis_suffix,
            )

    if keep_legacy_aliases:
        _set_legacy_aliases(stats=stats, basis_suffix=legacy_basis_suffix)

    metrics = {
        "selection/selected_count": float(len(selected_indices)),
        "selection/selected_ratio": float(len(selected_indices) / max(n, 1)),
        "selection/group_count": float(len(groups)),
        "selection/traj_adv_mean": float(_mean(stats["traj_adv"])),
        "selection/traj_num_tokens_mean": float(_mean(stats["traj_num_tokens"])),
    }

    for suffix in ("avglogprob", "sumlogprob"):
        metrics[f"selection/traj_logp_{suffix}_mean"] = float(_mean(stats[f"traj_logp_{suffix}"]))
        metrics[f"selection/traj_surprisal_{suffix}_mean"] = float(_mean(stats[f"traj_surprisal_{suffix}"]))
        metrics[f"selection/traj_entropy_push_before_{suffix}_mean"] = float(
            _mean(stats[f"traj_entropy_push_before_{suffix}"])
        )
        metrics[f"selection/traj_entropy_push_weighted_before_{suffix}_mean"] = float(
            _mean(stats[f"traj_entropy_push_weighted_before_{suffix}"])
        )

    if selected_indices:
        metrics["selection/selected_traj_adv_mean"] = float(_mean([stats["traj_adv"][idx] for idx in selected_indices]))
        metrics["selection/selected_traj_num_tokens_mean"] = float(
            _mean([stats["traj_num_tokens"][idx] for idx in selected_indices])
        )
        for suffix in ("avglogprob", "sumlogprob"):
            metrics[f"selection/selected_traj_logp_{suffix}_mean"] = float(
                _mean([stats[f"traj_logp_{suffix}"][idx] for idx in selected_indices])
            )
            metrics[f"selection/selected_traj_surprisal_{suffix}_mean"] = float(
                _mean([stats[f"traj_surprisal_{suffix}"][idx] for idx in selected_indices])
            )

    if groups:
        group_representatives = [idxs[0] for idxs in groups.values()]
        metrics["selection/group_token_entropy_mean"] = float(
            sum(stats["group_token_entropy_mean"][idx] for idx in group_representatives) / len(group_representatives)
        )
        metrics["selection/group_adv_all_zero_ratio"] = float(
            sum(1 for idx in group_representatives if stats["group_adv_all_zero"][idx]) / len(group_representatives)
        )
        metrics["selection/selector_random_fallback_ratio"] = float(
            sum(1 for idx in group_representatives if stats["selector_random_fallback"][idx]) / len(group_representatives)
        )
        metrics["selection/selector_single_sign_only_ratio"] = float(
            sum(1 for idx in group_representatives if stats["selector_single_sign_only"][idx]) / len(group_representatives)
        )
        metrics["selection/selector_ablation_added_count_mean"] = float(
            sum(stats["selector_ablation_added_count"][idx] for idx in group_representatives)
            / len(group_representatives)
        )
        metrics["selection/selector_ablation_negpush_unselected_count_mean"] = float(
            sum(stats["selector_ablation_negpush_unselected_count"][idx] for idx in group_representatives)
            / len(group_representatives)
        )
        metrics["selection/selector_ablation_added_ratio"] = float(
            sum(1 for idx in range(n) if stats["selector_ablation_added_for_update"][idx]) / max(n, 1)
        )

        for suffix in ("avglogprob", "sumlogprob"):
            metrics[f"selection/group_entropy_center_before_{suffix}_mean"] = float(
                sum(stats[f"group_entropy_center_before_{suffix}"][idx] for idx in group_representatives)
                / len(group_representatives)
            )
            metrics[f"selection/group_entropy_center_after_{suffix}_mean"] = float(
                sum(stats[f"group_entropy_center_after_{suffix}"][idx] for idx in group_representatives)
                / len(group_representatives)
            )
            metrics[f"selection/group_entropy_grad_proxy_before_{suffix}_mean"] = float(
                sum(stats[f"group_entropy_grad_proxy_before_{suffix}"][idx] for idx in group_representatives)
                / len(group_representatives)
            )
            metrics[f"selection/group_entropy_grad_proxy_after_{suffix}_mean"] = float(
                sum(stats[f"group_entropy_grad_proxy_after_{suffix}"][idx] for idx in group_representatives)
                / len(group_representatives)
            )
            metrics[f"selection/group_entropy_push_mean_before_{suffix}_mean"] = float(
                sum(stats[f"group_entropy_push_mean_before_{suffix}"][idx] for idx in group_representatives)
                / len(group_representatives)
            )
            metrics[f"selection/group_selected_entropy_push_mean_frozen_{suffix}_mean"] = float(
                sum(
                    stats[f"group_selected_entropy_push_mean_frozen_{suffix}"][idx]
                    for idx in group_representatives
                )
                / len(group_representatives)
            )
            metrics[f"selection/group_selected_abs_entropy_push_sum_frozen_{suffix}_mean"] = float(
                sum(
                    abs(stats[f"group_selected_entropy_push_sum_frozen_{suffix}"][idx])
                    for idx in group_representatives
                )
                / len(group_representatives)
            )
            metrics[f"selection/group_selected_abs_entropy_push_mean_frozen_{suffix}_mean"] = float(
                sum(
                    abs(stats[f"group_selected_entropy_push_mean_frozen_{suffix}"][idx])
                    for idx in group_representatives
                )
                / len(group_representatives)
            )
            metrics[f"selection/group_selected_entropy_push_mean_recentered_{suffix}_mean"] = float(
                sum(
                    stats[f"group_selected_entropy_push_mean_recentered_{suffix}"][idx]
                    for idx in group_representatives
                )
                / len(group_representatives)
            )
            metrics[f"selection/group_entropy_push_weighted_mean_before_{suffix}_mean"] = float(
                sum(stats[f"group_entropy_push_weighted_mean_before_{suffix}"][idx] for idx in group_representatives)
                / len(group_representatives)
            )
            metrics[f"selection/group_positive_push_count_before_{suffix}_mean"] = float(
                sum(stats[f"group_positive_push_count_before_{suffix}"][idx] for idx in group_representatives)
                / len(group_representatives)
            )
            metrics[f"selection/group_positive_push_ratio_before_{suffix}_mean"] = float(
                sum(stats[f"group_positive_push_ratio_before_{suffix}"][idx] for idx in group_representatives)
                / len(group_representatives)
            )
            metrics[f"selection/group_selected_entropy_push_weighted_mean_frozen_{suffix}_mean"] = float(
                sum(
                    stats[f"group_selected_entropy_push_weighted_mean_frozen_{suffix}"][idx]
                    for idx in group_representatives
                )
                / len(group_representatives)
            )
            metrics[f"selection/group_selected_entropy_push_weighted_mean_recentered_{suffix}_mean"] = float(
                sum(
                    stats[f"group_selected_entropy_push_weighted_mean_recentered_{suffix}"][idx]
                    for idx in group_representatives
                )
                / len(group_representatives)
            )

    if keep_legacy_aliases:
        metrics["selection/traj_avg_logprob_mean"] = float(_mean(stats["traj_avg_logprob"]))
        metrics["selection/traj_avg_surprisal_mean"] = float(_mean(stats["traj_avg_surprisal"]))
        metrics["selection/traj_entropy_push_proxy_mean"] = float(_mean(stats["traj_entropy_push_proxy"]))
        metrics["selection/traj_entropy_push_proxy_pos_ratio"] = float(
            sum(1 for value in stats["traj_entropy_push_proxy"] if value > 0) / max(n, 1)
        )
        metrics["selection/traj_entropy_push_weighted_proxy_mean"] = float(
            _mean(stats["traj_entropy_push_weighted_proxy"])
        )
        if groups:
            group_representatives = [idxs[0] for idxs in groups.values()]
            metrics["selection/group_cov_before_mean"] = float(
                sum(stats["group_cov_before_select"][idx] for idx in group_representatives) / len(group_representatives)
            )
            metrics["selection/group_cov_after_mean"] = float(
                sum(stats["group_cov_after_select"][idx] for idx in group_representatives) / len(group_representatives)
            )
            metrics["selection/group_entropy_proxy_mean"] = float(
                sum(stats["group_entropy_proxy"][idx] for idx in group_representatives) / len(group_representatives)
            )
            metrics["selection/group_entropy_grad_proxy_before_mean"] = float(
                sum(stats["group_entropy_grad_proxy_before_select"][idx] for idx in group_representatives)
                / len(group_representatives)
            )
            metrics["selection/group_entropy_grad_proxy_after_mean"] = float(
                sum(stats["group_entropy_grad_proxy_after_select"][idx] for idx in group_representatives)
                / len(group_representatives)
            )
            metrics["selection/group_entropy_grad_proxy_delta_mean"] = float(
                sum(
                    stats["group_entropy_grad_proxy_after_select"][idx]
                    - stats["group_entropy_grad_proxy_before_select"][idx]
                    for idx in group_representatives
                )
                / len(group_representatives)
            )
            metrics["selection/group_entropy_push_mean"] = float(
                sum(stats["group_entropy_push_mean"][idx] for idx in group_representatives) / len(group_representatives)
            )
            metrics["selection/group_selected_entropy_push_mean"] = float(
                sum(stats["group_selected_entropy_push_mean"][idx] for idx in group_representatives)
                / len(group_representatives)
            )
            metrics["selection/group_entropy_push_weighted_mean"] = float(
                sum(stats["group_entropy_push_weighted_mean"][idx] for idx in group_representatives)
                / len(group_representatives)
            )

    return {
        "selected_indices": sorted(selected_indices),
        "stats": stats,
        "metrics": metrics,
    }

