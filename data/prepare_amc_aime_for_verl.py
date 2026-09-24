#!/usr/bin/env python
"""Offline cleaner for AMC23/AIME24 -> verl RL parquet format."""

from __future__ import annotations
import os
from pathlib import Path

# qwen3 prompt from huggingface
SYSTEM_PROMPT = r"""Please reason step by step, and put your final answer within \boxed{}."""

def _require_pyarrow():
    try:
        import pyarrow as pa  # noqa: F401
        import pyarrow.parquet as pq  # noqa: F401
    except Exception as exc:  # pragma: no cover - dependency gate
        raise RuntimeError(
            "This script requires pyarrow. Please install it first, e.g. `pip install pyarrow`."
        ) from exc


def _pick(row: dict, keys: list[str], default=None):
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
    return default


def _read_rows(parquet_path: Path) -> list[dict]:
    import pyarrow.parquet as pq
    table = pq.read_table(str(parquet_path))
    return table.to_pylist()


def _normalize_rows(rows: list[dict], data_source: str) -> list[dict]:
    out = []
    for i, row in enumerate(rows):
        qid = _pick(row, ["id", "ID", "question_id", "qid", "idx"], i)
        problem = _pick(row, ["problem", "Problem", "question", "Question"], "")
        answer = _pick(row, ["answer", "Answer", "solution", "final_answer"], "")

        if problem is None or str(problem).strip() == "":
            # Skip malformed rows.
            continue

        item = {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT.strip()},
                {"role": "user", "content": str(problem).strip()},
            ],
            "data_source": data_source,
            # Keep this field to satisfy reward manager conventions in verl.
            "reward_model": {"ground_truth": "" if answer is None else str(answer)},
            "extra_info": {
                "index": str(qid),
                "question_id": str(qid),
                "raw_problem": str(problem).strip(),
            },
        }
        out.append(item)
    return out


def _write_rows(rows: list[dict], out_path: Path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    out_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, str(out_path))


def main():
    _require_pyarrow()

    DATA_ROOT = os.environ.get("DATA_ROOT", os.path.dirname(os.path.abspath(__file__)))

    # ==========================================
    # 1. Specify your input file paths
    # ==========================================
    amc23_input_path = os.path.join(DATA_ROOT, "AMC23", "AMC23.parquet")
    aime24_input_path = os.path.join(DATA_ROOT, "AIME24", "AIME24.parquet")

    # ==========================================
    # 2. Specify your output file paths (4 files total)
    # ==========================================
    amc23_train_out = os.path.join(DATA_ROOT, "AMC23", "AMC23_verl_train.parquet")
    amc23_test_out  = os.path.join(DATA_ROOT, "AMC23", "AMC23_verl_val.parquet")

    aime24_train_out = os.path.join(DATA_ROOT, "AIME24", "AIME24_verl_train.parquet")
    aime24_test_out  = os.path.join(DATA_ROOT, "AIME24", "AIME24_verl_val.parquet")

    # Read the raw data
    print("Reading raw data...")
    amc_rows = _read_rows(Path(amc23_input_path))
    aime_rows = _read_rows(Path(aime24_input_path))

    # Normalize the data format
    print("Processing data format...")
    amc_normalized = _normalize_rows(amc_rows, data_source="amc23")
    aime_normalized = _normalize_rows(aime_rows, data_source="aime24")

    # ==========================================
    # Save all data as both the train and test sets
    # ==========================================
    print("Saving all data to the train and test sets...")

    # AMC23: store the same full data twice
    _write_rows(amc_normalized, Path(amc23_train_out))
    _write_rows(amc_normalized, Path(amc23_test_out))

    # AIME24: store the same full data twice
    _write_rows(aime_normalized, Path(aime24_train_out))
    _write_rows(aime_normalized, Path(aime24_test_out))

    # Print the final result
    print("-" * 40)
    print(f"[Done] AMC23: train={len(amc_normalized)} rows, test={len(amc_normalized)} rows (full copy)")
    print(f"[Done] AIME24: train={len(aime_normalized)} rows, test={len(aime_normalized)} rows (full copy)")
    print("All files were successfully saved to the target directory!")


if __name__ == "__main__":
    main()