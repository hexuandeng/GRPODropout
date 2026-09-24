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

"""Ground-truth math reward utilities for the GRPO-dropout experiment.

These graders score generated solutions against the dataset ground truth and are
used as both the training and validation reward (``reward_model.reward_manager=batch``).
"""

from __future__ import annotations

import re
from typing import Any

from verl.utils.reward_score import math as math_reward
from verl.utils.reward_score import math_dapo


def _normalize_answer_text(answer: str) -> str:
    answer = answer.strip().strip("$").strip()
    if answer.endswith("."):
        answer = answer[:-1].rstrip()
    return math_dapo.normalize_final_answer(answer)


def _extract_by_markers(text: str) -> dict[str, str] | None:
    candidate_lines = [line.strip() for line in text.splitlines() if line.strip()]
    candidate_lines = candidate_lines[-4:]

    marker_patterns = [
        r"(?i)(?:final\s+answer|answer)\s*[:=]\s*(.+)$",
        r"(?i)(?:final\s+answer|answer)\s+is\s+(.+)$",
        r"(?i)(?:therefore|thus|hence|so)[,:]?\s+(?:the\s+)?answer\s+is\s+(.+)$",
        r"(?i)(.+?)\s+(?:is|are)\s+the\s+answer(?:s)?\.?$",
    ]

    for line in reversed(candidate_lines):
        for pattern in marker_patterns:
            match = re.search(pattern, line)
            if match:
                raw_span = match.group(1).strip()
                if raw_span:
                    return {
                        "pred": _normalize_answer_text(raw_span),
                        "extract_source": "answer_marker",
                        "raw_extracted_span": raw_span,
                    }
    return None


def _extract_boxed_with_meta(text: str) -> dict[str, str] | None:
    boxed = math_dapo.last_boxed_only_string(text)
    if boxed is None:
        boxed = math_reward.last_boxed_only_string(text)
    if boxed is None:
        return None

    try:
        answer = math_dapo.remove_boxed(boxed)
    except Exception:
        try:
            answer = math_reward.remove_boxed(boxed)
        except Exception:
            return {
                "pred": "",
                "extract_source": "invalid_boxed",
                "raw_extracted_span": boxed,
            }

    return {
        "pred": _normalize_answer_text(answer),
        "extract_source": "boxed",
        "raw_extracted_span": boxed,
    }


def _extract_final_answer_with_meta(text: str) -> dict[str, str]:
    if not text:
        return {"pred": "", "extract_source": "empty", "raw_extracted_span": ""}

    boxed_result = _extract_boxed_with_meta(text)
    if boxed_result is not None:
        return boxed_result

    marker_result = _extract_by_markers(text)
    if marker_result is not None:
        return marker_result

    return {"pred": "", "extract_source": "none", "raw_extracted_span": ""}


def _extract_final_answer(text: str) -> str:
    return _extract_final_answer_with_meta(text)["pred"]


def _answers_equiv(left: str, right: str) -> bool:
    if left == right:
        return True
    if left == "" or right == "":
        return False
    try:
        return math_reward.is_equiv(left, right)
    except Exception:
        return False


def _normalize_ground_truths(ground_truth: Any) -> list[str]:
    values = ground_truth if isinstance(ground_truth, list) else [ground_truth]
    normalized: list[str] = []
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        boxed = math_dapo.last_boxed_only_string(text)
        if boxed is not None:
            try:
                text = math_dapo.remove_boxed(boxed)
            except Exception:
                pass
        normalized.append(_normalize_answer_text(text))
    return [value for value in normalized if value != ""]


def _grade_with_ground_truth(solution_str: str, ground_truth: Any) -> dict[str, Any]:
    # Keep a heuristic extraction path because `math_dapo.compute_score` is strict and
    # may return pred='[INVALID]' when the model omits its preferred answer template.
    extract_meta = _extract_final_answer_with_meta(solution_str)
    pred = extract_meta["pred"]
    gt_candidates = _normalize_ground_truths(ground_truth)
    acc = False
    try:
        if ground_truth is None or isinstance(ground_truth, list):
            raise ValueError("Use the fallback grader for None/list ground-truth values.")
        result = math_dapo.compute_score(solution_str=solution_str, ground_truth=str(ground_truth))
        dapo_pred = result.get("pred", None)
        acc = bool(result.get("acc", False))

        # Prefer the boxed/marker extraction above. Minerva-style "Answer:" extraction
        # can capture markdown labels such as "**" from "Final Answer:" and overwrite
        # a valid \boxed{} answer.
        if acc and dapo_pred not in (None, "", "[INVALID]") and pred == "":
            pred = _normalize_answer_text(str(dapo_pred))

        # If strict DAPO parsing fails, fall back to the heuristic extraction/equivalence path.
        if not acc and pred != "":
            acc = any(_answers_equiv(pred, gt) for gt in gt_candidates)
    except Exception:
        acc = pred != "" and any(_answers_equiv(pred, gt) for gt in gt_candidates)

    return {
        "score": float(acc),
        "acc": float(acc),
        "pred": pred,
        "extract_source": extract_meta["extract_source"],
        "raw_extracted_span": extract_meta["raw_extracted_span"],
    }


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


def compute_score_batch_gt_math(
    data_sources,
    solution_strs,
    ground_truths,
    extra_infos,
    **kwargs,
):
    """Compute validation reward against the original ground truth."""

    n = len(solution_strs)
    _, question_ids = _build_group_keys(extra_infos=extra_infos, uids=None, n=n)
    results: list[dict[str, Any]] = []

    for idx, (solution_str, ground_truth) in enumerate(zip(solution_strs, ground_truths, strict=False)):
        item = _grade_with_ground_truth(solution_str=solution_str, ground_truth=ground_truth)
        if question_ids[idx] is not None:
            item["question_id"] = question_ids[idx]
        results.append(item)

    return results

def compute_score_gt_math(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    **kwargs,
):
    """Compute a single-sample ground-truth math reward for DAPO-style reward managers."""

    return _grade_with_ground_truth(solution_str=solution_str, ground_truth=ground_truth)
