#!/usr/bin/env python
"""Offline cleaner for DAPO17K -> verl RL parquet format."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = r"""Please reason step by step, and put your final answer within \boxed{}."""
VAL_SIZE = 100


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


def _read_rows(parquet_path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    table = pq.read_table(str(parquet_path))
    return table.to_pylist()


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize_rows(rows: list[dict[str, Any]], data_source: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        problem = _pick(row, ["prompt", "question", "problem", "Question", "Problem"], "")
        reward_model = row.get("reward_model") or {}
        answer = _pick(row, ["solution", "answer", "final_answer"], reward_model.get("ground_truth", ""))

        problem_text = _normalize_text(problem)
        answer_text = _normalize_text(answer)
        question_id = str(i)

        if not problem_text:
            continue

        out.append(
            {
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT.strip()},
                    {"role": "user", "content": problem_text},
                ],
                "data_source": data_source,
                "question_id": question_id,
                "reward_model": {"ground_truth": answer_text},
                "extra_info": {
                    "index": question_id,
                    "question_id": question_id,
                    "raw_problem": problem_text,
                },
            }
        )
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

    input_path = Path(os.path.join(DATA_ROOT, "dapo17k", "dapo17k.parquet"))
    train_out = Path(os.path.join(DATA_ROOT, "dapo17k", "dapo17k_verl_train.parquet"))
    val_out = Path(os.path.join(DATA_ROOT, "dapo17k", "dapo17k_verl_val.parquet"))

    print("Reading DAPO17K parquet...")
    rows = _read_rows(input_path)

    print("Normalizing rows for verl...")
    normalized = _normalize_rows(rows, data_source="dapo17k")
    val_rows = normalized[: min(VAL_SIZE, len(normalized))]

    print("Writing parquet files...")
    _write_rows(normalized, train_out)
    _write_rows(val_rows, val_out)

    print("-" * 40)
    print(f"[Done] Train rows: {len(normalized)} -> {train_out}")
    print(f"[Done] Val rows:   {len(val_rows)} -> {val_out}")
    print(f"Validation split uses the first {len(val_rows)} normalized rows.")


if __name__ == "__main__":
    main()
