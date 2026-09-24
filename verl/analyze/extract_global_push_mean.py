#!/usr/bin/env python3
"""Compute global rollout push means from saved rollout JSONL logs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path(__file__).resolve().parents[1] / "rollout_logs/qwen3-1.7B-GRPO_balance/dapo17k"
DEFAULT_BASIS = "avg"
DEFAULT_FIELD: str | None = None
DEFAULT_PATTERN = "*.jsonl"
DEFAULT_SELECTED_ONLY = False
DEFAULT_OUTPUT: Path | None = None


def _parse_step(path: Path) -> int:
    try:
        return int(path.stem)
    except ValueError:
        return 0


def _iter_jsonl_files(input_path: Path, pattern: str) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    files = sorted(input_path.rglob(pattern), key=lambda path: (str(path.parent), _parse_step(path), path.name))
    return [path for path in files if path.is_file() and path.stat().st_size > 0]


def _iter_rows(file_path: Path):
    with file_path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {file_path}:{line_no}: {exc}") from exc


def _as_finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _field_for_basis(basis: str) -> str:
    normalized = basis.lower()
    if normalized in {"avg", "avglogprob"}:
        return "traj_entropy_push_before_avglogprob"
    if normalized in {"sum", "sumlogprob"}:
        return "traj_entropy_push_before_sumlogprob"
    if normalized in {"legacy", "proxy"}:
        return "traj_entropy_push_proxy"
    raise ValueError("Unsupported basis. Use avg, sum, or legacy.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute global mean/std for GRPO-dropout rollout push values. The input can be a single JSONL file, "
            "one rollout run directory, a dataset subdirectory, or the rollout_logs root."
        )
    )
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=DEFAULT_INPUT,
        help=f"Rollout JSONL file or directory. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--basis",
        choices=("avg", "sum", "legacy"),
        default=DEFAULT_BASIS,
        help=f"Which push field to aggregate. Default: {DEFAULT_BASIS}.",
    )
    parser.add_argument(
        "--field",
        default=DEFAULT_FIELD,
        help="Explicit JSON field to aggregate. Overrides --basis.",
    )
    parser.add_argument(
        "--pattern",
        default=DEFAULT_PATTERN,
        help=f"Glob pattern used under directory inputs. Default: {DEFAULT_PATTERN}.",
    )
    parser.add_argument(
        "--selected-only",
        action="store_true",
        default=DEFAULT_SELECTED_ONLY,
        help="Aggregate only rows with selected_for_update=true.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Optional path to write a JSON summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input does not exist: {input_path}")

    field = args.field or _field_for_basis(args.basis)
    files = _iter_jsonl_files(input_path, args.pattern)
    if not files:
        raise FileNotFoundError(f"No non-empty JSONL files found under {input_path}")

    count = 0
    skipped_missing = 0
    skipped_nonfinite = 0
    skipped_unselected = 0
    total_rows = 0
    mean = 0.0
    m2 = 0.0
    min_value: float | None = None
    max_value: float | None = None

    for file_path in files:
        for row in _iter_rows(file_path):
            total_rows += 1
            if args.selected_only and not bool(row.get("selected_for_update")):
                skipped_unselected += 1
                continue
            if field not in row:
                skipped_missing += 1
                continue
            value = _as_finite_float(row.get(field))
            if value is None:
                skipped_nonfinite += 1
                continue

            count += 1
            delta = value - mean
            mean += delta / count
            m2 += delta * (value - mean)
            min_value = value if min_value is None else min(min_value, value)
            max_value = value if max_value is None else max(max_value, value)

    if count == 0:
        raise ValueError(f"No finite values found for field {field!r}.")

    variance = m2 / count
    summary = {
        "input": str(input_path),
        "field": field,
        "basis": args.basis,
        "selected_only": bool(args.selected_only),
        "file_count": len(files),
        "total_rows": total_rows,
        "count": count,
        "mean": mean,
        "std": math.sqrt(variance),
        "min": min_value,
        "max": max_value,
        "skipped_missing": skipped_missing,
        "skipped_nonfinite": skipped_nonfinite,
        "skipped_unselected": skipped_unselected,
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.output is not None:
        output_path = args.output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
