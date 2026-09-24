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

"""Selection-method helpers for the GRPO-dropout selector.

This module intentionally contains only method-local logic:

- log-probability basis normalization
- per-group sample selection methods
- selector registry and dispatch

The main statistics-building workflow remains in `selector.py`.
"""

from __future__ import annotations

import random
from typing import Any


BASIS_ALIASES = {
    "avg": "avglogprob",
    "avglogprob": "avglogprob",
    "sum": "sumlogprob",
    "sumlogprob": "sumlogprob",
}


def basis_suffix(name: str) -> str:
    try:
        return BASIS_ALIASES[str(name).lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported log-probability basis: {name!r}. Use one of {sorted(BASIS_ALIASES)}") from exc


def select_random(idxs: list[int], update_count: int, group_key: str, select_random_seed: int, **kwargs) -> list[int]:
    rng = random.Random(f"{select_random_seed}:{group_key}")
    return sorted(rng.sample(idxs, update_count))


def select_topk_adv(idxs: list[int], update_count: int, traj_adv_mean: list[float], **kwargs) -> list[int]:
    return sorted(idxs, key=lambda idx: traj_adv_mean[idx], reverse=True)[:update_count]


def select_topk_avg_logprob(
    idxs: list[int], update_count: int, traj_logp_avglogprob: list[float], **kwargs
) -> list[int]:
    return sorted(idxs, key=lambda idx: traj_logp_avglogprob[idx], reverse=True)[:update_count]


def select_topk_logprob(
    idxs: list[int],
    update_count: int,
    traj_logp_avglogprob: list[float],
    traj_logp_sumlogprob: list[float],
    select_logprob_basis: str,
    **kwargs,
) -> list[int]:
    suffix = basis_suffix(select_logprob_basis)
    values = traj_logp_sumlogprob if suffix == "sumlogprob" else traj_logp_avglogprob
    return sorted(idxs, key=lambda idx: values[idx], reverse=True)[:update_count]


def select_low_entropy(idxs: list[int], update_count: int, traj_entropy_mean: list[float], **kwargs) -> list[int]:
    return sorted(idxs, key=lambda idx: traj_entropy_mean[idx])[:update_count]


def select_positive_push_topk(
    idxs: list[int],
    update_count: int,
    group_key: str,
    select_random_seed: int,
    traj_entropy_push_before_avglogprob: list[float],
    traj_entropy_push_before_sumlogprob: list[float],
    entropy_logprob_basis: str,
    **kwargs,
) -> list[int]:
    suffix = basis_suffix(entropy_logprob_basis)
    push_values = (
        traj_entropy_push_before_sumlogprob if suffix == "sumlogprob" else traj_entropy_push_before_avglogprob
    )
    positive_indices = [idx for idx in idxs if push_values[idx] > 0]

    if not positive_indices:
        return select_random(
            idxs=idxs,
            update_count=update_count,
            group_key=group_key,
            select_random_seed=select_random_seed,
        )

    if len(positive_indices) <= update_count:
        return sorted(positive_indices)

    return sorted(positive_indices, key=lambda idx: push_values[idx], reverse=True)[:update_count]

def select_negative_push_topk(
    idxs: list[int],
    update_count: int,
    group_key: str,
    select_random_seed: int,
    traj_entropy_push_before_avglogprob: list[float],
    traj_entropy_push_before_sumlogprob: list[float],
    entropy_logprob_basis: str,
    **kwargs,
) -> list[int]:
    suffix = basis_suffix(entropy_logprob_basis)
    push_values = (
        traj_entropy_push_before_sumlogprob if suffix == "sumlogprob" else traj_entropy_push_before_avglogprob
    )
    eps = 1e-12
    negative_indices = [idx for idx in idxs if push_values[idx] < -eps]

    if not negative_indices:
        return select_random(
            idxs=idxs,
            update_count=update_count,
            group_key=group_key,
            select_random_seed=select_random_seed,
        )

    if len(negative_indices) >= update_count:
        return sorted(negative_indices, key=lambda idx: push_values[idx])[:update_count]

    selected = list(sorted(negative_indices, key=lambda idx: push_values[idx]))
    selected_set = set(selected)
    non_negative_sorted = sorted(
        [idx for idx in idxs if idx not in selected_set],
        key=lambda idx: push_values[idx],
    )
    selected.extend(non_negative_sorted[: update_count - len(selected)])
    return sorted(selected)


def _sample_subset(pool: list[int], count: int, rng: random.Random) -> list[int]:
    if count <= 0 or not pool:
        return []
    if count >= len(pool):
        return list(pool)
    return rng.sample(pool, count)


def select_pseudo_label_ratio(
    idxs: list[int],
    update_count: int,
    group_key: str,
    select_random_seed: int,
    traj_pseudo_score: list[float],
    select_positive_ratio: float,
    **kwargs,
) -> tuple[list[int], dict[str, Any]]:
    rng = random.Random(f"{select_random_seed}:{group_key}:pseudo_label_ratio")
    positive_indices = [idx for idx in idxs if float(traj_pseudo_score[idx]) > 0.0]
    negative_indices = [idx for idx in idxs if float(traj_pseudo_score[idx]) <= 0.0]

    target_positive = int(round(update_count * float(select_positive_ratio)))
    target_positive = max(0, min(update_count, target_positive))
    target_negative = update_count - target_positive

    selected_positive = _sample_subset(positive_indices, min(target_positive, len(positive_indices)), rng)
    selected_negative = _sample_subset(negative_indices, min(target_negative, len(negative_indices)), rng)
    selected = list(selected_positive) + list(selected_negative)

    selected_set = set(selected)
    remaining = [idx for idx in idxs if idx not in selected_set]
    selected.extend(_sample_subset(remaining, update_count - len(selected), rng))

    return sorted(selected), {
        "ratio_target_positive": target_positive,
        "ratio_target_negative": target_negative,
        "ratio_selected_positive": sum(1 for idx in selected if float(traj_pseudo_score[idx]) > 0.0),
        "ratio_selected_negative": sum(1 for idx in selected if float(traj_pseudo_score[idx]) <= 0.0),
    }


def select_push_sign_ratio(
    idxs: list[int],
    update_count: int,
    group_key: str,
    select_random_seed: int,
    traj_entropy_push_before_avglogprob: list[float],
    traj_entropy_push_before_sumlogprob: list[float],
    entropy_logprob_basis: str,
    select_positive_ratio: float,
    **kwargs,
) -> tuple[list[int], dict[str, Any]]:
    suffix = basis_suffix(entropy_logprob_basis)
    push_values = (
        traj_entropy_push_before_sumlogprob if suffix == "sumlogprob" else traj_entropy_push_before_avglogprob
    )
    eps = 1e-12
    rng = random.Random(f"{select_random_seed}:{group_key}:push_sign_ratio")

    positive_indices = [idx for idx in idxs if float(push_values[idx]) > eps]
    negative_indices = [idx for idx in idxs if float(push_values[idx]) < -eps]
    zero_indices = [idx for idx in idxs if abs(float(push_values[idx])) <= eps]

    target_positive = int(round(update_count * float(select_positive_ratio)))
    target_positive = max(0, min(update_count, target_positive))
    target_negative = update_count - target_positive

    selected_positive = _sample_subset(positive_indices, min(target_positive, len(positive_indices)), rng)
    selected_negative = _sample_subset(negative_indices, min(target_negative, len(negative_indices)), rng)
    selected = list(selected_positive) + list(selected_negative)

    selected_set = set(selected)
    zero_remaining = [idx for idx in zero_indices if idx not in selected_set]
    selected.extend(_sample_subset(zero_remaining, update_count - len(selected), rng))

    selected_set = set(selected)
    remaining = [idx for idx in idxs if idx not in selected_set]
    selected.extend(_sample_subset(remaining, update_count - len(selected), rng))

    return sorted(selected), {
        "ratio_target_positive": target_positive,
        "ratio_target_negative": target_negative,
        "ratio_selected_positive": sum(1 for idx in selected if float(push_values[idx]) > eps),
        "ratio_selected_negative": sum(1 for idx in selected if float(push_values[idx]) < -eps),
        "ratio_selected_zero": sum(1 for idx in selected if abs(float(push_values[idx])) <= eps),
    }


def select_balanced_push_zero(
    idxs: list[int],
    update_count: int,
    group_key: str,
    select_random_seed: int,
    traj_entropy_push_before_avglogprob: list[float],
    traj_entropy_push_before_sumlogprob: list[float],
    entropy_logprob_basis: str,
    **kwargs,
) -> tuple[list[int], dict[str, Any]]:
    suffix = basis_suffix(entropy_logprob_basis)
    push_values = (
        traj_entropy_push_before_sumlogprob if suffix == "sumlogprob" else traj_entropy_push_before_avglogprob
    )
    eps = 1e-12
    group_pushes = {idx: float(push_values[idx]) for idx in idxs}
    positive_indices = [idx for idx in idxs if group_pushes[idx] > eps]
    negative_indices = [idx for idx in idxs if group_pushes[idx] < -eps]
    zero_indices = [idx for idx in idxs if abs(group_pushes[idx]) <= eps]
    min_select_count = max(1, len(idxs) // 4)

    # v1 kept intentionally for reference; do not delete.
    # if len(zero_indices) == len(idxs):
    #     return select_random(
    #         idxs=idxs,
    #         update_count=update_count,
    #         group_key=group_key,
    #         select_random_seed=select_random_seed,
    #     ), {
    #         "balance_random_fallback": True,
    #         "balance_single_sign_only": False,
    #     }
    #
    # if not positive_indices or not negative_indices:
    #     chosen = list(sorted(positive_indices, key=lambda idx: group_pushes[idx], reverse=True))
    #     if len(chosen) > update_count:
    #         return sorted(chosen[:update_count]), {
    #             "balance_random_fallback": False,
    #             "balance_single_sign_only": True,
    #             "balance_positive_overflow": True,
    #         }
    #     remaining_slots = max(update_count - len(chosen), 0)
    #     chosen.extend(list(sorted(zero_indices))[:remaining_slots])
    #     remaining_slots = max(update_count - len(chosen), 0)
    #     same_sign_pool = positive_indices if positive_indices else negative_indices
    #     same_sign_sorted = sorted(
    #         same_sign_pool,
    #         key=lambda idx: group_pushes[idx],
    #         reverse=bool(negative_indices),
    #     )
    #     chosen.extend(same_sign_sorted[:remaining_slots])
    #     return sorted(chosen), {
    #         "balance_random_fallback": False,
    #         "balance_single_sign_only": True,
    #     }
    #
    # selected = list(sorted(positive_indices, key=lambda idx: group_pushes[idx], reverse=True))
    # positive_overflow = len(selected) > update_count
    # if positive_overflow:
    #     selected = selected[:update_count]
    #     return sorted(selected), {
    #         "balance_random_fallback": False,
    #         "balance_single_sign_only": False,
    #         "balance_positive_overflow": True,
    #         "balance_negative_prefix_len": 0,
    #     }
    #
    # positive_sum = sum(group_pushes[idx] for idx in selected)
    # available_slots = update_count - len(selected)
    # negative_sorted = sorted(negative_indices, key=lambda idx: group_pushes[idx], reverse=True)
    #
    # best_prefix_len = 0
    # best_abs_sum = abs(positive_sum)
    # running_prefix_sum = 0.0
    # for prefix_len, idx in enumerate(negative_sorted[:available_slots], start=1):
    #     running_prefix_sum += group_pushes[idx]
    #     total_abs_sum = abs(positive_sum + running_prefix_sum)
    #     if total_abs_sum < best_abs_sum:
    #         best_abs_sum = total_abs_sum
    #         best_prefix_len = prefix_len
    #
    # selected.extend(negative_sorted[:best_prefix_len])
    # selected_set = set(selected)
    # remaining_slots = max(update_count - len(selected), 0)
    # remaining_zeros = [idx for idx in sorted(zero_indices) if idx not in selected_set]
    # selected.extend(remaining_zeros[:remaining_slots])
    # selected_set = set(selected)
    # remaining_slots = max(update_count - len(selected), 0)
    # remaining_negative = [idx for idx in negative_sorted[best_prefix_len:] if idx not in selected_set]
    # selected.extend(remaining_negative[:remaining_slots])
    # selected_set = set(selected)
    # remaining_slots = max(update_count - len(selected), 0)
    # remaining_positive = [idx for idx in positive_indices if idx not in selected_set]
    # selected.extend(sorted(remaining_positive, key=lambda idx: group_pushes[idx], reverse=True)[:remaining_slots])
    # return sorted(selected), {
    #     "balance_random_fallback": False,
    #     "balance_single_sign_only": False,
    #     "balance_positive_overflow": False,
    #     "balance_negative_prefix_len": best_prefix_len,
    # }

    if len(zero_indices) == len(idxs):
        rng = random.Random(f"{select_random_seed}:{group_key}:balanced_push_zero:all_zero")
        return sorted(_sample_subset(idxs, min_select_count, rng)), {
            "balance_random_fallback": True,
            "balance_single_sign_only": False,
            "balance_min_select_count": min_select_count,
            "balance_selected_push_sum": 0.0,
        }

    if not positive_indices or not negative_indices:
        chosen = list(sorted(positive_indices, key=lambda idx: group_pushes[idx], reverse=True))

        if len(chosen) < min_select_count:
            remaining_slots = min_select_count - len(chosen)
            chosen.extend(list(sorted(zero_indices))[:remaining_slots])

        if len(chosen) < min_select_count:
            remaining_slots = min_select_count - len(chosen)
            same_sign_pool = negative_indices if negative_indices else positive_indices
            same_sign_sorted = sorted(same_sign_pool, key=lambda idx: group_pushes[idx], reverse=True)
            chosen_set = set(chosen)
            chosen.extend([idx for idx in same_sign_sorted if idx not in chosen_set][:remaining_slots])

        chosen_push_sum = sum(group_pushes[idx] for idx in chosen)
        return sorted(chosen), {
            "balance_random_fallback": False,
            "balance_single_sign_only": True,
            "balance_min_select_count": min_select_count,
            "balance_negative_prefix_len": 0,
            "balance_selected_push_sum": chosen_push_sum,
            "balance_min_select_satisfied": len(chosen) >= min_select_count,
        }

    # v2 behavior:
    # 1. select all positive pushes
    # 2. sort negatives from large to small, e.g. -0.1, -0.2, ...
    # 3. greedily add negatives while the running sum stays strictly positive
    # 4. once the next negative would make the sum non-positive, stop immediately
    selected = list(sorted(positive_indices, key=lambda idx: group_pushes[idx], reverse=True))
    running_sum = sum(group_pushes[idx] for idx in selected)
    negative_sorted = sorted(negative_indices, key=lambda idx: group_pushes[idx], reverse=True)
    selected_negative_prefix: list[int] = []

    for idx in negative_sorted:
        candidate_sum = running_sum + group_pushes[idx]
        if candidate_sum <= eps:
            break
        selected.append(idx)
        selected_negative_prefix.append(idx)
        running_sum = candidate_sum

    if len(selected) < min_select_count:
        selected_set = set(selected)
        remaining_slots = min_select_count - len(selected)
        remaining_zeros = [idx for idx in sorted(zero_indices) if idx not in selected_set]
        selected.extend(remaining_zeros[:remaining_slots])

    # Do not add more negative pushes here: that would violate the guarantee
    # that the selected push sum stays strictly positive and approaches zero
    # from the positive side.

    selected_push_sum = sum(group_pushes[idx] for idx in selected)
    return sorted(selected), {
        "balance_random_fallback": False,
        "balance_single_sign_only": False,
        "balance_negative_prefix_len": len(selected_negative_prefix),
        "balance_min_select_count": min_select_count,
        "balance_selected_push_sum": selected_push_sum,
        "balance_min_select_satisfied": len(selected) >= min_select_count,
    }

def _select_balanced_push_zero_add_negpush_adv_sign(
    idxs: list[int],
    update_count: int,
    group_key: str,
    select_random_seed: int,
    traj_adv_mean: list[float],
    traj_entropy_push_before_avglogprob: list[float],
    traj_entropy_push_before_sumlogprob: list[float],
    entropy_logprob_basis: str,
    adv_sign: str,
    **kwargs,
) -> tuple[list[int], dict[str, Any]]:
    base_selected, base_meta = select_balanced_push_zero(
        idxs=idxs,
        update_count=update_count,
        group_key=group_key,
        select_random_seed=select_random_seed,
        traj_entropy_push_before_avglogprob=traj_entropy_push_before_avglogprob,
        traj_entropy_push_before_sumlogprob=traj_entropy_push_before_sumlogprob,
        entropy_logprob_basis=entropy_logprob_basis,
    )

    suffix = basis_suffix(entropy_logprob_basis)
    push_values = (
        traj_entropy_push_before_sumlogprob if suffix == "sumlogprob" else traj_entropy_push_before_avglogprob
    )
    eps = 1e-12
    base_selected_set = set(base_selected)
    negative_push_unselected = [
        idx for idx in idxs if idx not in base_selected_set and float(push_values[idx]) < -eps
    ]

    if adv_sign == "positive":
        added = [idx for idx in negative_push_unselected if float(traj_adv_mean[idx]) > eps]
        mode = "negpush_posadv"
    elif adv_sign == "negative":
        added = [idx for idx in negative_push_unselected if float(traj_adv_mean[idx]) < -eps]
        mode = "negpush_negadv"
    else:
        raise ValueError(f"Unsupported adv_sign: {adv_sign!r}")

    selected = sorted(base_selected + added)
    meta = dict(base_meta)
    meta.update(
        {
            "balance_ablation_mode": mode,
            "balance_base_selected_indices": sorted(base_selected),
            "balance_base_selected_count": len(base_selected),
            "balance_ablation_added_indices": sorted(added),
            "balance_ablation_added_count": len(added),
            "balance_ablation_negpush_unselected_count": len(negative_push_unselected),
            "balance_ablation_selected_count": len(selected),
        }
    )
    return selected, meta


def select_balanced_push_zero_add_negpush_posadv(
    idxs: list[int],
    update_count: int,
    group_key: str,
    select_random_seed: int,
    traj_adv_mean: list[float],
    traj_entropy_push_before_avglogprob: list[float],
    traj_entropy_push_before_sumlogprob: list[float],
    entropy_logprob_basis: str,
    **kwargs,
) -> tuple[list[int], dict[str, Any]]:
    return _select_balanced_push_zero_add_negpush_adv_sign(
        idxs=idxs,
        update_count=update_count,
        group_key=group_key,
        select_random_seed=select_random_seed,
        traj_adv_mean=traj_adv_mean,
        traj_entropy_push_before_avglogprob=traj_entropy_push_before_avglogprob,
        traj_entropy_push_before_sumlogprob=traj_entropy_push_before_sumlogprob,
        entropy_logprob_basis=entropy_logprob_basis,
        adv_sign="positive",
    )


def select_balanced_push_zero_add_negpush_negadv(
    idxs: list[int],
    update_count: int,
    group_key: str,
    select_random_seed: int,
    traj_adv_mean: list[float],
    traj_entropy_push_before_avglogprob: list[float],
    traj_entropy_push_before_sumlogprob: list[float],
    entropy_logprob_basis: str,
    **kwargs,
) -> tuple[list[int], dict[str, Any]]:
    return _select_balanced_push_zero_add_negpush_adv_sign(
        idxs=idxs,
        update_count=update_count,
        group_key=group_key,
        select_random_seed=select_random_seed,
        traj_adv_mean=traj_adv_mean,
        traj_entropy_push_before_avglogprob=traj_entropy_push_before_avglogprob,
        traj_entropy_push_before_sumlogprob=traj_entropy_push_before_sumlogprob,
        entropy_logprob_basis=entropy_logprob_basis,
        adv_sign="negative",
    )


def select_balanced_push_zero_drop_negpush_posadv(
    idxs: list[int],
    update_count: int,
    group_key: str,
    select_random_seed: int,
    traj_adv_mean: list[float],
    traj_entropy_push_before_avglogprob: list[float],
    traj_entropy_push_before_sumlogprob: list[float],
    entropy_logprob_basis: str,
    **kwargs,
) -> tuple[list[int], dict[str, Any]]:
    suffix = basis_suffix(entropy_logprob_basis)
    push_values = (
        traj_entropy_push_before_sumlogprob if suffix == "sumlogprob" else traj_entropy_push_before_avglogprob
    )
    eps = 1e-12
    group_pushes = {idx: float(push_values[idx]) for idx in idxs}
    group_advs = {idx: float(traj_adv_mean[idx]) for idx in idxs}
    positive_indices = [idx for idx in idxs if group_pushes[idx] > eps]
    negative_indices = [idx for idx in idxs if group_pushes[idx] < -eps]
    zero_indices = [idx for idx in idxs if abs(group_pushes[idx]) <= eps]
    min_select_count = max(1, len(idxs) // 4)

    if len(zero_indices) == len(idxs) or not positive_indices:
        selected, meta = select_balanced_push_zero(
            idxs=idxs,
            update_count=update_count,
            group_key=group_key,
            select_random_seed=select_random_seed,
            traj_entropy_push_before_avglogprob=traj_entropy_push_before_avglogprob,
            traj_entropy_push_before_sumlogprob=traj_entropy_push_before_sumlogprob,
            entropy_logprob_basis=entropy_logprob_basis,
        )
        meta = dict(meta)
        meta["balance_ablation_mode"] = "drop_negpush_posadv"
        meta["balance_base_selected_indices"] = sorted(selected)
        meta["balance_ablation_added_indices"] = []
        meta["balance_ablation_added_count"] = 0
        meta["balance_dropped_negpush_posadv_count"] = 0
        meta["balance_dropped_pospush_posadv_count"] = 0
        return selected, meta

    selected = list(sorted(positive_indices, key=lambda idx: group_pushes[idx], reverse=True))
    running_sum = sum(group_pushes[idx] for idx in selected)
    negative_sorted = sorted(
        [idx for idx in negative_indices if group_advs[idx] <= eps],
        key=lambda idx: group_pushes[idx],
        reverse=True,
    )
    selected_negative_prefix: list[int] = []

    for idx in negative_sorted:
        candidate_sum = running_sum + group_pushes[idx]
        if candidate_sum <= eps:
            break
        selected.append(idx)
        selected_negative_prefix.append(idx)
        running_sum = candidate_sum

    if len(selected) < min_select_count:
        selected_set = set(selected)
        remaining_slots = min_select_count - len(selected)
        remaining_zeros = [idx for idx in sorted(zero_indices) if idx not in selected_set]
        selected.extend(remaining_zeros[:remaining_slots])
        running_sum = sum(group_pushes[idx] for idx in selected)

    prune_candidates = [
        idx for idx in selected if group_pushes[idx] > eps and group_advs[idx] > eps
    ]
    best_drop: list[int] = []
    best_sum = running_sum

    # Groups are small (8 rollouts in the current scripts), so exhaustive
    # subset search keeps the "preserve positives when tied" rule simple.
    for mask in range(1, 1 << len(prune_candidates)):
        drop = [idx for bit, idx in enumerate(prune_candidates) if mask & (1 << bit)]
        if len(selected) - len(drop) < min_select_count:
            continue
        candidate_sum = running_sum - sum(group_pushes[idx] for idx in drop)
        if candidate_sum <= eps:
            continue
        if candidate_sum < best_sum - eps or (
            abs(candidate_sum - best_sum) <= eps and best_drop and len(drop) < len(best_drop)
        ):
            best_sum = candidate_sum
            best_drop = drop

    if best_drop:
        drop_set = set(best_drop)
        selected = [idx for idx in selected if idx not in drop_set]

    selected_push_sum = sum(group_pushes[idx] for idx in selected)
    dropped_negpush_posadv = [
        idx for idx in negative_indices if group_advs[idx] > eps
    ]
    return sorted(selected), {
        "balance_random_fallback": False,
        "balance_single_sign_only": False,
        "balance_negative_prefix_len": len(selected_negative_prefix),
        "balance_min_select_count": min_select_count,
        "balance_selected_push_sum": selected_push_sum,
        "balance_min_select_satisfied": len(selected) >= min_select_count,
        "balance_ablation_mode": "drop_negpush_posadv",
        "balance_base_selected_indices": sorted(selected),
        "balance_ablation_added_indices": [],
        "balance_ablation_added_count": 0,
        "balance_ablation_selected_count": len(selected),
        "balance_dropped_negpush_posadv_count": len(dropped_negpush_posadv),
        "balance_dropped_pospush_posadv_count": len(best_drop),
    }

def select_balanced_push_centered_global_mean(
    idxs: list[int],
    update_count: int,
    group_key: str,
    select_random_seed: int,
    traj_entropy_push_before_avglogprob: list[float],
    traj_entropy_push_before_sumlogprob: list[float],
    entropy_logprob_basis: str,
    balance_push_center: float = 0.0,
    **kwargs,
) -> tuple[list[int], dict[str, Any]]:
    suffix = basis_suffix(entropy_logprob_basis)
    center = float(balance_push_center)
    raw_push_values = (
        traj_entropy_push_before_sumlogprob if suffix == "sumlogprob" else traj_entropy_push_before_avglogprob
    )

    centered_avg_push = list(traj_entropy_push_before_avglogprob)
    centered_sum_push = list(traj_entropy_push_before_sumlogprob)
    if suffix == "sumlogprob":
        centered_sum_push = [float(value) - center for value in traj_entropy_push_before_sumlogprob]
    else:
        centered_avg_push = [float(value) - center for value in traj_entropy_push_before_avglogprob]

    selected, meta = select_balanced_push_zero(
        idxs=idxs,
        update_count=update_count,
        group_key=group_key,
        select_random_seed=select_random_seed,
        traj_entropy_push_before_avglogprob=centered_avg_push,
        traj_entropy_push_before_sumlogprob=centered_sum_push,
        entropy_logprob_basis=entropy_logprob_basis,
    )

    meta = dict(meta)
    meta.update(
        {
            "balance_ablation_mode": "centered_global_mean",
            "balance_push_center": center,
            "balance_base_selected_indices": sorted(selected),
            "balance_ablation_added_indices": [],
            "balance_ablation_added_count": 0,
            "balance_selected_raw_push_sum": sum(float(raw_push_values[idx]) for idx in selected),
            "balance_selected_centered_push_sum": sum(float(raw_push_values[idx]) - center for idx in selected),
        }
    )
    return selected, meta

def _compute_balance_target_ratio(
    global_steps: int | None,
    total_training_steps: int | None,
    balance_target_ratio_start: float,
    balance_target_ratio_end: float,
    balance_target_ratio_schedule: str,
) -> float:
    start = float(balance_target_ratio_start)
    end = float(balance_target_ratio_end)
    schedule = str(balance_target_ratio_schedule).lower()
    if schedule != "linear":
        raise ValueError(
            "Unsupported balance_target_ratio_schedule: "
            f"{balance_target_ratio_schedule!r}. Use 'linear'."
        )

    if global_steps is None or total_training_steps is None or total_training_steps <= 1:
        return start

    progress = (float(global_steps) - 1.0) / max(float(total_training_steps) - 1.0, 1.0)
    progress = min(max(progress, 0.0), 1.0)
    return start + (end - start) * progress


def select_balanced_push_threshold(
    idxs: list[int],
    update_count: int,
    group_key: str,
    select_random_seed: int,
    traj_entropy_push_before_avglogprob: list[float],
    traj_entropy_push_before_sumlogprob: list[float],
    entropy_logprob_basis: str,
    global_steps: int | None = None,
    total_training_steps: int | None = None,
    balance_target_ratio_start: float = 1.0,
    balance_target_ratio_end: float = 0.0,
    balance_target_ratio_schedule: str = "linear",
    **kwargs,
) -> tuple[list[int], dict[str, Any]]:
    suffix = basis_suffix(entropy_logprob_basis)
    push_values = (
        traj_entropy_push_before_sumlogprob if suffix == "sumlogprob" else traj_entropy_push_before_avglogprob
    )
    eps = 1e-12
    group_pushes = {idx: float(push_values[idx]) for idx in idxs}
    positive_indices = [idx for idx in idxs if group_pushes[idx] > eps]
    negative_indices = [idx for idx in idxs if group_pushes[idx] < -eps]
    zero_indices = [idx for idx in idxs if abs(group_pushes[idx]) <= eps]
    min_select_count = max(1, len(idxs) // 4)

    if len(zero_indices) == len(idxs):
        rng = random.Random(f"{select_random_seed}:{group_key}:balanced_push_threshold:all_zero")
        return sorted(_sample_subset(idxs, min_select_count, rng)), {
            "balance_random_fallback": True,
            "balance_single_sign_only": False,
            "balance_min_select_count": min_select_count,
            "balance_selected_push_sum": 0.0,
            "balance_target_ratio": _compute_balance_target_ratio(
                global_steps=global_steps,
                total_training_steps=total_training_steps,
                balance_target_ratio_start=balance_target_ratio_start,
                balance_target_ratio_end=balance_target_ratio_end,
                balance_target_ratio_schedule=balance_target_ratio_schedule,
            ),
            "balance_target_sum": 0.0,
        }

    if not positive_indices or not negative_indices:
        chosen = list(sorted(positive_indices, key=lambda idx: group_pushes[idx], reverse=True))

        if len(chosen) < min_select_count:
            remaining_slots = min_select_count - len(chosen)
            chosen.extend(list(sorted(zero_indices))[:remaining_slots])

        if len(chosen) < min_select_count:
            remaining_slots = min_select_count - len(chosen)
            same_sign_pool = negative_indices if negative_indices else positive_indices
            same_sign_sorted = sorted(same_sign_pool, key=lambda idx: group_pushes[idx], reverse=True)
            chosen_set = set(chosen)
            chosen.extend([idx for idx in same_sign_sorted if idx not in chosen_set][:remaining_slots])

        chosen_push_sum = sum(group_pushes[idx] for idx in chosen)
        positive_push_sum = sum(group_pushes[idx] for idx in positive_indices)
        target_ratio = _compute_balance_target_ratio(
            global_steps=global_steps,
            total_training_steps=total_training_steps,
            balance_target_ratio_start=balance_target_ratio_start,
            balance_target_ratio_end=balance_target_ratio_end,
            balance_target_ratio_schedule=balance_target_ratio_schedule,
        )
        return sorted(chosen), {
            "balance_random_fallback": False,
            "balance_single_sign_only": True,
            "balance_min_select_count": min_select_count,
            "balance_negative_prefix_len": 0,
            "balance_selected_push_sum": chosen_push_sum,
            "balance_min_select_satisfied": len(chosen) >= min_select_count,
            "balance_target_ratio": target_ratio,
            "balance_target_sum": target_ratio * positive_push_sum,
        }

    target_ratio = _compute_balance_target_ratio(
        global_steps=global_steps,
        total_training_steps=total_training_steps,
        balance_target_ratio_start=balance_target_ratio_start,
        balance_target_ratio_end=balance_target_ratio_end,
        balance_target_ratio_schedule=balance_target_ratio_schedule,
    )

    selected = list(sorted(positive_indices, key=lambda idx: group_pushes[idx], reverse=True))
    positive_push_sum = sum(group_pushes[idx] for idx in selected)
    running_sum = positive_push_sum
    target_sum = target_ratio * positive_push_sum
    negative_sorted = sorted(negative_indices, key=lambda idx: group_pushes[idx], reverse=True)
    selected_negative_prefix: list[int] = []

    for idx in negative_sorted:
        candidate_sum = running_sum + group_pushes[idx]
        if candidate_sum <= target_sum + eps:
            break
        selected.append(idx)
        selected_negative_prefix.append(idx)
        running_sum = candidate_sum

    if len(selected) < min_select_count:
        selected_set = set(selected)
        remaining_slots = min_select_count - len(selected)
        remaining_zeros = [idx for idx in sorted(zero_indices) if idx not in selected_set]
        selected.extend(remaining_zeros[:remaining_slots])

    selected_push_sum = sum(group_pushes[idx] for idx in selected)
    return sorted(selected), {
        "balance_random_fallback": False,
        "balance_single_sign_only": False,
        "balance_negative_prefix_len": len(selected_negative_prefix),
        "balance_min_select_count": min_select_count,
        "balance_selected_push_sum": selected_push_sum,
        "balance_min_select_satisfied": len(selected) >= min_select_count,
        "balance_target_ratio": target_ratio,
        "balance_target_sum": target_sum,
    }


SELECTORS = {
    "random": select_random,
    "topk_adv": select_topk_adv,
    "topk_avg_logprob": select_topk_avg_logprob,
    "topk_logprob": select_topk_logprob,
    "low_entropy": select_low_entropy,
    "pseudo_label_ratio": select_pseudo_label_ratio,
    "positive_push_topk": select_positive_push_topk,
    "push_sign_ratio": select_push_sign_ratio,
    "negative_push_topk": select_negative_push_topk,
    "balanced_push_zero_add_negpush_posadv": select_balanced_push_zero_add_negpush_posadv,
    "balanced_push_zero_add_negpush_negadv": select_balanced_push_zero_add_negpush_negadv,
    "balanced_push_zero_drop_negpush_posadv": select_balanced_push_zero_drop_negpush_posadv,
    "balanced_push_zero": select_balanced_push_zero,
    "balanced_push_threshold": select_balanced_push_threshold,
    "balanced_push_centered_global_mean": select_balanced_push_centered_global_mean,
    "all": None,
}


def select_indices(
    idxs: list[int],
    group_key: str,
    traj_adv_mean: list[float],
    traj_logp_avglogprob: list[float],
    traj_logp_sumlogprob: list[float],
    traj_entropy_mean: list[float],
    traj_pseudo_score: list[float],
    traj_entropy_push_before_avglogprob: list[float],
    traj_entropy_push_before_sumlogprob: list[float],
    select_method: str,
    n_update_samples_per_prompt: int | None,
    select_random_seed: int,
    entropy_logprob_basis: str,
    select_logprob_basis: str,
    select_positive_ratio: float,
    global_steps: int | None = None,
    total_training_steps: int | None = None,
    balance_target_ratio_start: float = 1.0,
    balance_target_ratio_end: float = 0.0,
    balance_target_ratio_schedule: str = "linear",
    balance_push_center: float = 0.0,
) -> tuple[list[int], dict[str, Any]]:
    if n_update_samples_per_prompt is None or n_update_samples_per_prompt <= 0:
        return list(idxs), {}

    update_count = min(len(idxs), n_update_samples_per_prompt)
    if update_count >= len(idxs) or select_method == "all":
        return list(idxs), {}

    selector = SELECTORS.get(select_method)
    if selector is None:
        raise ValueError(f"Unsupported select_method: {select_method}")

    result = selector(
        idxs=idxs,
        update_count=update_count,
        group_key=group_key,
        traj_adv_mean=traj_adv_mean,
        traj_logp_avglogprob=traj_logp_avglogprob,
        traj_logp_sumlogprob=traj_logp_sumlogprob,
        traj_entropy_mean=traj_entropy_mean,
        traj_pseudo_score=traj_pseudo_score,
        traj_entropy_push_before_avglogprob=traj_entropy_push_before_avglogprob,
        traj_entropy_push_before_sumlogprob=traj_entropy_push_before_sumlogprob,
        select_random_seed=select_random_seed,
        entropy_logprob_basis=entropy_logprob_basis,
        select_logprob_basis=select_logprob_basis,
        select_positive_ratio=select_positive_ratio,
        global_steps=global_steps,
        total_training_steps=total_training_steps,
        balance_target_ratio_start=balance_target_ratio_start,
        balance_target_ratio_end=balance_target_ratio_end,
        balance_target_ratio_schedule=balance_target_ratio_schedule,
        balance_push_center=balance_push_center,
    )
    if isinstance(result, tuple):
        return result
    return result, {}
