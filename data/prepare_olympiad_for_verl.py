#!/usr/bin/env python
"""Offline cleaner for OlympiadBench jsonl -> verl RL parquet format."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = r"""Please reason step by step, and put your final answer within \boxed{}."""


def _require_pyarrow() -> None:
    try:
        import pyarrow as pa  # noqa: F401
        import pyarrow.parquet as pq  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "This script requires pyarrow. Please install it first, e.g. `pip install pyarrow`."
        ) from exc


def _pick(row: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return default


def _read_jsonl_rows(jsonl_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {jsonl_path}") from exc
    return rows


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize_rows(rows: list[dict[str, Any]], data_source: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        qid = _pick(row, ["id", "ID", "question_id", "qid", "idx"], i)
        problem = _pick(row, ["problem", "Problem", "question", "Question", "prompt"], "")
        answer = _pick(row, ["answer", "Answer", "final_answer", "solution"], "")

        problem_text = _normalize_text(problem)
        answer_text = _normalize_text(answer)
        qid_text = _normalize_text(qid if qid is not None else i)

        if not problem_text:
            continue

        item = {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT.strip()},
                {"role": "user", "content": problem_text},
            ],
            "data_source": data_source,
            "reward_model": {"ground_truth": answer_text},
            "extra_info": {
                "index": qid_text,
                "question_id": qid_text,
                "raw_problem": problem_text,
            },
        }
        out.append(item)
    return out


def _write_rows(rows: list[dict[str, Any]], out_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    out_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, str(out_path))


def main() -> None:
    _require_pyarrow()

    DATA_ROOT = os.environ.get("DATA_ROOT", os.path.dirname(os.path.abspath(__file__)))

    input_path = Path(os.path.join(DATA_ROOT, "olympiadbench.jsonl"))
    train_out = Path(os.path.join(DATA_ROOT, "Olympiad", "Olympiad_verl_train.parquet"))
    val_out = Path(os.path.join(DATA_ROOT, "Olympiad", "Olympiad_verl_val.parquet"))

    print("Reading OlympiadBench jsonl...")
    rows = _read_jsonl_rows(input_path)

    print("Normalizing rows for verl...")
    normalized = _normalize_rows(rows, data_source="olympiadbench")

    print("Writing full data to both train and val parquet files...")
    _write_rows(normalized, train_out)
    _write_rows(normalized, val_out)

    print("-" * 40)
    print(f"[Done] Train rows: {len(normalized)} -> {train_out}")
    print(f"[Done] Val rows:   {len(normalized)} -> {val_out}")
    print("Both train and val use the full normalized dataset.")


if __name__ == "__main__":
    main()
