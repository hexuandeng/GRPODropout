from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
STEP_RE = re.compile(r"\bstep:(\d+)\s+-\s+(.+)")
METRIC_RE = re.compile(r"([A-Za-z0-9_.@/\-]+):([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)", re.IGNORECASE)
ROLLOUT_DIR_RE = re.compile(r"trainer\.rollout_data_dir=([^\s]+)")
EXPERIMENT_RE = re.compile(r"trainer\.experiment_name=([^\s]+)")
SELECT_METHOD_RE = re.compile(r"\+custom_training_module\.kwargs\.select_method=([^\s]+)")


DEFAULT_LOG_DIR = Path(__file__).resolve().parents[1] / "logs" / "qwen3-1.7b"
DEFAULT_ROLLOUT_ROOT = Path(__file__).resolve().parents[2] / "rollout_logs"


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def parse_run_name(run_name: str) -> tuple[str, str]:
    match = re.match(r"^(?P<model>.+?)-(?:GRPO|TTRL)_(?P<family>.+)$", run_name)
    if match:
        marker = "GRPO" if "-GRPO_" in run_name else "TTRL"
        return match.group("model"), f"{marker}_{match.group('family')}"
    return run_name, ""


def row_key(row: dict[str, Any]) -> tuple[str, str, int]:
    run = str(row.get("run_name") or row.get("experiment") or row.get("source_log") or "")
    dataset = str(row.get("dataset") or "")
    step = int(float(row.get("step", 0)))
    return run, dataset, step


def rollout_key_from_path(path: Path) -> tuple[str, str, int]:
    return path.parent.parent.name, path.parent.name, int(path.stem)


def discover_rollout_files(input_dir: Path) -> tuple[Path, list[Path]]:
    input_dir = input_dir.resolve()
    if not input_dir.exists():
        return input_dir, []
    direct_files = sorted(input_dir.glob("*.jsonl"), key=lambda path: int(path.stem))
    if direct_files:
        return input_dir.parent.parent, direct_files
    run_files = sorted(input_dir.glob("*/*.jsonl"), key=lambda path: (path.parent.name, int(path.stem)))
    if run_files:
        return input_dir.parent, run_files
    root_files = sorted(
        input_dir.glob("*/*/*.jsonl"),
        key=lambda path: (path.parent.parent.name, path.parent.name, int(path.stem)),
    )
    return input_dir, root_files


def infer_log_identity(path: Path, text_head: str) -> dict[str, Any]:
    experiment = None
    rollout_relpath = None
    select_method = None
    for pattern, setter in [
        (EXPERIMENT_RE, "experiment"),
        (ROLLOUT_DIR_RE, "rollout_dir"),
        (SELECT_METHOD_RE, "select_method"),
    ]:
        match = pattern.search(text_head)
        if not match:
            continue
        if setter == "experiment":
            experiment = match.group(1)
        elif setter == "rollout_dir":
            raw = match.group(1).replace("\\", "/").strip()
            marker = "/rollout_logs/"
            lower = raw.lower()
            rollout_relpath = raw[lower.index(marker) + len(marker) :] if marker in lower else raw
        elif setter == "select_method":
            select_method = match.group(1)

    stem = path.stem
    if experiment is None:
        experiment = stem

    run_name = ""
    dataset = ""
    if rollout_relpath:
        parts = rollout_relpath.strip("/").split("/")
        if len(parts) >= 2:
            run_name, dataset = parts[0], parts[1]
    model, run_method = parse_run_name(run_name) if run_name else ("", "")
    return {
        "source_log": path.name,
        "experiment": experiment,
        "rollout_relpath": rollout_relpath or "",
        "run_name": run_name,
        "model": model,
        "run_method": run_method,
        "dataset": dataset,
        "method": select_method or run_method or stem,
    }


def parse_log_file(path: Path) -> list[dict[str, Any]]:
    # Read once. These logs are moderate and parsing line-oriented metrics is cheap.
    text = path.read_text(encoding="utf-8", errors="ignore")
    identity = infer_log_identity(path, text[:12000])
    rows: list[dict[str, Any]] = []
    for raw_line in text.splitlines():
        line = strip_ansi(raw_line)
        match = STEP_RE.search(line)
        if not match:
            continue
        step = int(match.group(1))
        tail = match.group(2)
        metrics = {name: safe_float(value) for name, value in METRIC_RE.findall(tail)}
        metrics = {name: value for name, value in metrics.items() if value is not None}
        if not metrics:
            continue
        row = dict(identity)
        row.update(
            {
                "step": step,
                "log_has_step": True,
                "rollout_has_step": False,
                "metric_source": "log",
            }
        )
        row.update(metrics)
        rows.append(row)
    return rows


def parse_all_logs(log_dir: Path) -> list[dict[str, Any]]:
    if not log_dir.exists():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(log_dir.glob("*.log")):
        rows.extend(parse_log_file(path))
    return rows


def adv_value(row: dict[str, Any]) -> float | None:
    for key in ("traj_adv_before_select", "traj_adv"):
        value = safe_float(row.get(key))
        if value is not None:
            return value
    return None


def sign(value: float | None, eps: float = 1e-12) -> str:
    if value is None or abs(value) <= eps:
        return "zero"
    return "pos" if value > 0 else "neg"


def selected_entropy_push(row: dict[str, Any]) -> float | None:
    basis = str(row.get("entropy_logprob_basis") or "avglogprob").lower()
    suffix = "sumlogprob" if basis in {"sum", "sumlogprob"} else "avglogprob"
    for key in (
        f"traj_entropy_push_selected_frozen_{suffix}",
        f"traj_entropy_push_before_{suffix}",
        "traj_entropy_push_proxy",
    ):
        value = safe_float(row.get(key))
        if value is not None:
            return value
    return None


def summarize_rollout_file(path: Path, rollout_root: Path) -> dict[str, Any] | None:
    try:
        step = int(path.stem)
    except ValueError:
        return None
    dataset_dir = path.parent
    run_dir = dataset_dir.parent
    run_name = run_dir.name
    dataset = dataset_dir.name
    model, run_method = parse_run_name(run_name)
    relpath = str(path.relative_to(rollout_root)).replace("\\", "/")
    mtime = path.stat().st_mtime

    total_entropy_sum = 0.0
    total_tokens = 0.0
    selected_pushes: list[float] = []
    cov_values: list[float] = []
    counts = defaultdict(int)
    score_values: list[float] = []
    reward_values: list[float] = []
    first: dict[str, Any] | None = None
    group_reps: dict[str, dict[str, Any]] = {}

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if first is None:
                first = row
            group_key = str(row.get("group_key", ""))
            group_reps.setdefault(group_key, row)

            selected = safe_bool(row.get("selected_for_update"))
            bucket = sign(adv_value(row))
            counts[f"{'selected' if selected else 'deleted'}_adv_{bucket}_count"] += 1
            if selected:
                push = selected_entropy_push(row)
                if push is not None:
                    selected_pushes.append(push)

            entropy_sum = safe_float(row.get("traj_entropy_sum"))
            token_count = safe_float(row.get("traj_num_tokens"))
            if entropy_sum is not None and token_count is not None and token_count > 0:
                total_entropy_sum += entropy_sum
                total_tokens += token_count

            score = safe_float(row.get("score"))
            if score is not None:
                score_values.append(score)
            reward = safe_float(row.get("reward"))
            if reward is None:
                reward = score
            if reward is not None:
                reward_values.append(reward)

    if first is None:
        return None
    for rep in group_reps.values():
        cov = safe_float(rep.get("group_cov_before_select"))
        if cov is not None:
            cov_values.append(cov)

    method = str(first.get("select_method") or run_method or "unknown")
    ablation = str(first.get("selector_ablation_mode") or "").strip()
    if ablation:
        method = f"{method}:{ablation}"

    return {
        "source_log": "",
        "experiment": run_name,
        "rollout_relpath": relpath,
        "rollout_mtime": mtime,
        "run_name": run_name,
        "model": model,
        "run_method": run_method,
        "dataset": dataset,
        "method": method,
        "step": step,
        "log_has_step": False,
        "rollout_has_step": True,
        "metric_source": "rollout",
        "actor/entropy_rollout": (total_entropy_sum / total_tokens) if total_tokens > 0 else None,
        "selection/group_cov_before_mean_rollout": mean(cov_values),
        "selected_entropy_push_sum_frozen_total": sum(selected_pushes),
        "selected_entropy_push_mean_frozen": mean(selected_pushes),
        "critic/score/mean_rollout": mean(score_values),
        "critic/rewards/mean_rollout": mean(reward_values),
        **counts,
    }


def parse_rollouts(rollout_root: Path, cached_rows: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    logical_root, files = discover_rollout_files(rollout_root)
    if not files:
        return []
    cached_by_key = {row_key(row): row for row in (cached_rows or []) if safe_bool(row.get("rollout_has_step"))}
    rows: list[dict[str, Any]] = []
    for path in files:
        key = rollout_key_from_path(path)
        cached = cached_by_key.get(key)
        if cached is not None:
            cached_mtime = safe_float(cached.get("rollout_mtime"))
            if cached_mtime is not None and abs(cached_mtime - path.stat().st_mtime) < 1e-6:
                rows.append(dict(cached))
                continue
        summary = summarize_rollout_file(path, logical_root)
        if summary:
            rows.append(summary)
    return rows


def merge_log_and_rollout(log_rows: list[dict[str, Any]], rollout_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str, int], dict[str, Any]] = {}

    for row in rollout_rows:
        merged[row_key(row)] = dict(row)
    for row in log_rows:
        key = row_key(row)
        base = merged.get(key, {})
        combined = dict(base)
        combined.update(row)
        combined["log_has_step"] = True
        combined["rollout_has_step"] = bool(base)
        combined["metric_source"] = "log+rollout" if base else "log"
        merged[key] = combined

    # Fill preferred columns from log first, rollout fallback second.
    for row in merged.values():
        if safe_float(row.get("actor/entropy")) is None:
            row["actor/entropy"] = row.get("actor/entropy_rollout")
        if safe_float(row.get("critic/score/mean")) is None:
            row["critic/score/mean"] = row.get("critic/score/mean_rollout")
        if safe_float(row.get("critic/rewards/mean")) is None:
            row["critic/rewards/mean"] = row.get("critic/rewards/mean_rollout")
        if safe_float(row.get("selection/group_cov_before_mean")) is None:
            row["selection/group_cov_before_mean"] = row.get("selection/group_cov_before_mean_rollout")

    return sorted(merged.values(), key=lambda r: (str(r.get("model")), str(r.get("run_name")), str(r.get("dataset")), int(float(r.get("step", 0)))))


def update_cache(
    log_dir: Path = DEFAULT_LOG_DIR,
    rollout_root: Path = DEFAULT_ROLLOUT_ROOT,
    output_csv: Path | None = None,
) -> list[dict[str, Any]]:
    if output_csv is None:
        output_csv = Path(__file__).resolve().parent / "outputs" / "live_step_summary.csv"
    cached_rows = read_csv(output_csv)
    log_rows = parse_all_logs(log_dir)
    rollout_rows = parse_rollouts(rollout_root, cached_rows=cached_rows)
    rows = merge_log_and_rollout(log_rows, rollout_rows)
    write_csv(output_csv, rows)
    return rows


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build live metric cache from training logs and rollout JSONL files.")
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--rollout-root", type=Path, default=DEFAULT_ROLLOUT_ROOT)
    parser.add_argument("--output-csv", type=Path, default=Path(__file__).resolve().parent / "outputs" / "live_step_summary.csv")
    args = parser.parse_args()
    rows = update_cache(log_dir=args.log_dir, rollout_root=args.rollout_root, output_csv=args.output_csv)
    print(f"Wrote {len(rows)} rows to {args.output_csv}")
