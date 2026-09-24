#!/usr/bin/env python
"""Convert cleaned JSON/JSONL files into verl-compatible parquet files."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

DEFAULT_SYSTEM_PROMPT = r"Please reason step by step, and put your final answer within \boxed{}."

def _require_pyarrow():
    try:
        import pyarrow as pa  # noqa: F401
        import pyarrow.parquet as pq  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("This script requires pyarrow. Install it with `pip install pyarrow`.") from exc


def _load_rows(input_path: Path) -> list[dict[str, Any]]:
    text = input_path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    if input_path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    data = json.loads(text)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if "data" in data and isinstance(data["data"], list):
            return data["data"]
        return [data]
    raise ValueError(f"Unsupported JSON structure in {input_path}")


def _pick(row: dict[str, Any], keys: list[str], default=None):
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return default


def _normalize_prompt(row: dict[str, Any], system_prompt: str) -> list[dict[str, str]]:
    prompt = row.get("prompt")
    if isinstance(prompt, list) and prompt:
        return prompt

    problem = _pick(row, ["question", "problem", "prompt_text", "input", "query"], "")
    problem = "" if problem is None else str(problem).strip()
    if not problem:
        raise ValueError(f"Missing question/problem field in row: {row}")

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": problem},
    ]


def _normalize_row(row: dict[str, Any], idx: int, data_source: str, system_prompt: str) -> dict[str, Any]:
    qid = _pick(row, ["id", "ID", "question_id", "qid", "idx"], idx)
    answer = _pick(row, ["answer", "Answer", "solution", "final_answer", "ground_truth"], "")
    raw_problem = _pick(row, ["question", "problem", "prompt_text", "input", "query"], "")
    url = row.get("url")

    item = {
        "prompt": _normalize_prompt(row, system_prompt=system_prompt),
        "data_source": data_source,
        "reward_model": {"ground_truth": "" if answer is None else str(answer).strip()},
        "extra_info": {
            "index": str(qid),
            "question_id": str(qid),
            "raw_problem": "" if raw_problem is None else str(raw_problem).strip(),
        },
    }
    if url is not None:
        item["extra_info"]["url"] = str(url)
    return item


def _write_rows(rows: list[dict[str, Any]], output_path: Path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, str(output_path))


def main():
    _require_pyarrow()

    DATA_ROOT = os.environ.get("DATA_ROOT", os.path.dirname(os.path.abspath(__file__)))

    # ==========================================
    # 1. Specify your input file path (supports .json or .jsonl)
    # ==========================================
    # Note: change the extension here to match your actual file (e.g. .jsonl)
    dataset1_input_path = os.path.join(DATA_ROOT, "AIME24", "AIME24_view.json")

    # ==========================================
    # 2. Specify your output file paths (4 Parquet files total)
    # ==========================================
    dataset1_train_out = os.path.join(DATA_ROOT, "AIME24", "AIME24_verl_train.parquet")
    dataset1_test_out  = os.path.join(DATA_ROOT, "AIME24", "AIME24_verl_val.parquet")

    # # ==========================================
    # Process Dataset 1 (e.g. AMC23)
    # ==========================================
    print("Reading Dataset 1...")
    d1_rows = _load_rows(Path(dataset1_input_path))
    d1_normalized = [
        _normalize_row(row, idx=i, data_source="aime24", system_prompt=DEFAULT_SYSTEM_PROMPT)
        for i, row in enumerate(d1_rows)
    ]

    print("Saving all Dataset 1 data to the train and test sets...")
    _write_rows(d1_normalized, Path(dataset1_train_out))
    _write_rows(d1_normalized, Path(dataset1_test_out))

    # ==========================================
    # Process Dataset 2 (e.g. AIME24)
    # ==========================================
    # print("Reading Dataset 2...")
    # d2_rows = _load_rows(Path(dataset2_input_path))
    # d2_normalized = [
    #     _normalize_row(row, idx=i, data_source="aime24", system_prompt=DEFAULT_SYSTEM_PROMPT)
    #     for i, row in enumerate(d2_rows)
    # ]

    # print("Saving all Dataset 2 data to the train and test sets...")
    # _write_rows(d2_normalized, Path(dataset2_train_out))
    # _write_rows(d2_normalized, Path(dataset2_test_out))

    # Print the final result
    print("-" * 40)
    print(f"[Done] Dataset 1: train={len(d1_normalized)} rows, test={len(d1_normalized)} rows (full copy)")
    # print(f"[Done] Dataset 2: train={len(d2_normalized)} rows, test={len(d2_normalized)} rows (full copy)")
    print("All JSON/JSONL files were successfully converted to Parquet and saved to the target directory!")


if __name__ == "__main__":
    main()