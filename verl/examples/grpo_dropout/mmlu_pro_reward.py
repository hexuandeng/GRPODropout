"""Strict binary reward for MMLU-Pro multiple-choice responses."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

_VALID_LABEL_RE = re.compile(r"^[A-J]$")
_BOXED_LABEL_RE = re.compile(r"^\\(?:text|mathrm|mathbf)\{\s*([A-J])\s*\}$", re.IGNORECASE)
_FINAL_MARKER_RE = re.compile(
    r"(?:^|\n)[ \t]*(?:the[ \t]+)?(?:final[ \t]+answer|answer)"
    r"[ \t]*(?:is|:|=)[ \t]*\(?([A-J])\)?[.]?[ \t\r\n]*$",
    re.IGNORECASE,
)
_FINAL_LINE_RE = re.compile(r"^\s*\(?([A-J])\)?[.]?\s*$", re.IGNORECASE)


def _last_boxed_content(text: str) -> str | None:
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    cursor = start + len("\\boxed{")
    depth = 1
    while cursor < len(text):
        if text[cursor] == "{":
            depth += 1
        elif text[cursor] == "}":
            depth -= 1
            if depth == 0:
                return text[start + len("\\boxed{") : cursor]
        cursor += 1
    return None


def extract_mmlu_pro_answer(solution_str: str) -> tuple[str | None, str]:
    """Extract only an explicit final A-J label and report its source."""
    if not isinstance(solution_str, str):
        return None, "invalid_type"
    text = solution_str.strip()
    if not text:
        return None, "missing"

    boxed = _last_boxed_content(text)
    if boxed is not None:
        candidate = boxed.strip().upper()
        wrapped = _BOXED_LABEL_RE.fullmatch(boxed.strip())
        if wrapped is not None:
            candidate = wrapped.group(1).upper()
        if _VALID_LABEL_RE.fullmatch(candidate):
            return candidate, "boxed"
        return None, "invalid_boxed"

    marker = _FINAL_MARKER_RE.search(text)
    if marker is not None:
        return marker.group(1).upper(), "final_marker"

    final_line = text.splitlines()[-1]
    line_match = _FINAL_LINE_RE.fullmatch(final_line)
    if line_match is not None:
        return line_match.group(1).upper(), "final_line"
    return None, "missing"


def compute_score_mmlu_pro(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
    **_: Any,
) -> dict[str, Any]:
    if data_source != "mmlu_pro":
        raise ValueError(f"MMLU-Pro reward received unsupported data_source={data_source!r}")
    if not isinstance(ground_truth, str):
        raise ValueError("MMLU-Pro ground truth must be a string label")
    ground_truth = ground_truth.strip().upper()
    if not _VALID_LABEL_RE.fullmatch(ground_truth):
        raise ValueError(f"Invalid MMLU-Pro ground truth: {ground_truth!r}")

    pred, extract_source = extract_mmlu_pro_answer(solution_str)
    correct = pred == ground_truth
    extra_info = extra_info or {}
    question_id = extra_info.get("question_id", extra_info.get("index"))
    category = extra_info.get("category")
    return {
        "score": 1.0 if correct else 0.0,
        "acc": float(correct),
        "oracle_acc": float(correct),
        # Validation metrics support categorical strings but cannot aggregate None.
        "pred": pred if pred is not None else "INVALID",
        "extract_source": extract_source,
        "question_id": "" if question_id is None else str(question_id),
        "category": "" if category is None else str(category),
    }


def _as_list(name: str, values: Sequence[Any]) -> list[Any]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence")
    try:
        return list(values)
    except TypeError as exc:
        raise TypeError(f"{name} must be a sequence") from exc


def compute_score_batch_mmlu_pro(
    data_sources: Sequence[str],
    solution_strs: Sequence[str],
    ground_truths: Sequence[str],
    extra_infos: Sequence[dict[str, Any] | None] | None = None,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    data_sources = _as_list("data_sources", data_sources)
    solution_strs = _as_list("solution_strs", solution_strs)
    ground_truths = _as_list("ground_truths", ground_truths)
    if extra_infos is None:
        extra_infos = [None] * len(solution_strs)
    else:
        extra_infos = _as_list("extra_infos", extra_infos)

    lengths = {
        "data_sources": len(data_sources),
        "solution_strs": len(solution_strs),
        "ground_truths": len(ground_truths),
        "extra_infos": len(extra_infos),
    }
    if len(set(lengths.values())) != 1:
        raise ValueError(f"MMLU-Pro batch inputs have different lengths: {lengths}")

    return [
        compute_score_mmlu_pro(data_source, solution, ground_truth, extra_info, **kwargs)
        for data_source, solution, ground_truth, extra_info in zip(
            data_sources, solution_strs, ground_truths, extra_infos
        )
    ]
