from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


EPS_DEFAULT = 1e-12
DEFAULT_ROLLOUT_ROOT = Path(__file__).resolve().parents[2] / "rollout_logs"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "nan"}:
        return None
    try:
        number = float(text)
    except ValueError:
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

    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def discover_rollout_files(input_dir: Path) -> tuple[Path, list[Path]]:
    """Return a logical rollout root plus JSONL files under a flexible input dir.

    Accepted inputs:
    - rollout_logs root:             root / run / dataset / step.jsonl
    - one run directory:             run / dataset / step.jsonl
    - one dataset/step directory:    dataset / step.jsonl
    """
    input_dir = input_dir.resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path must be a directory: {input_dir}")

    direct_files = sorted(input_dir.glob("*.jsonl"), key=parse_step)
    if direct_files:
        # The parent of the run directory is the logical root when possible.
        # For .../run/dataset this returns .../rollout_logs.
        logical_root = input_dir.parent.parent if input_dir.parent != input_dir else input_dir
        return logical_root, direct_files

    run_files = sorted(
        input_dir.glob("*/*.jsonl"),
        key=lambda path: (path.parent.name, parse_step(path)),
    )
    if run_files:
        # For .../run this returns .../rollout_logs.
        return input_dir.parent, run_files

    root_files = sorted(
        input_dir.glob("*/*/*.jsonl"),
        key=lambda path: (path.parent.parent.name, path.parent.name, parse_step(path)),
    )
    if root_files:
        return input_dir, root_files

    raise FileNotFoundError(
        "No rollout JSONL files found. Expected one of: "
        "<input>/*.jsonl, <input>/*/*.jsonl, or <input>/*/*/*.jsonl"
    )


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} line {line_no}: {exc}") from exc


def parse_step(path: Path) -> int:
    try:
        return int(path.stem)
    except ValueError as exc:
        raise ValueError(f"Rollout file name must be an integer step: {path}") from exc


def parse_run_name(run_name: str) -> tuple[str, str]:
    # Examples:
    # qwen3-1.7B-GRPO_balance -> (qwen3-1.7B, GRPO_balance)
    # qwen3-1.7B-GRPO_balance-neg -> (qwen3-1.7B, GRPO_balance-neg)
    match = re.match(r"^(?P<model>.+?)-(?:GRPO|TTRL)_(?P<family>.+)$", run_name)
    if match:
        marker = "GRPO" if "-GRPO_" in run_name else "TTRL"
        return match.group("model"), f"{marker}_{match.group('family')}"
    return run_name, ""


def method_label(row: dict[str, Any], run_method: str) -> str:
    method = str(row.get("select_method") or run_method or "unknown")
    ablation = str(row.get("selector_ablation_mode") or "").strip()
    if ablation:
        return f"{method}:{ablation}"
    return method


def adv_value(row: dict[str, Any]) -> float | None:
    value = safe_float(row.get("traj_adv_before_select"))
    if value is not None:
        return value
    return safe_float(row.get("traj_adv"))


def score_value(row: dict[str, Any]) -> float | None:
    value = safe_float(row.get("traj_pseudo_score"))
    if value is not None:
        return value
    value = safe_float(row.get("score"))
    if value is not None:
        return value
    return safe_float(row.get("acc"))


def sign_bucket(value: float | None, eps: float) -> str:
    if value is None or abs(value) <= eps:
        return "zero"
    return "pos" if value > 0 else "neg"


def add_sign_count(counts: dict[str, int], prefix: str, value: float | None, eps: float) -> None:
    counts[f"{prefix}_{sign_bucket(value, eps)}"] += 1


def first_nonempty(rows: list[dict[str, Any]], key: str) -> Any:
    for row in rows:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return None


def normalized_basis(value: str | None) -> str:
    text = str(value or "").strip().lower()
    if text in {"sum", "sumlogprob"}:
        return "sumlogprob"
    return "avglogprob"


def selected_entropy_push_value(row: dict[str, Any], entropy_basis: str) -> float | None:
    """Entropy push of one selected trajectory, frozen to the original group center.

    Positive means this selected trajectory pushes entropy upward under the same
    proxy used by group_cov_before_select. Fallbacks support older logs.
    """
    suffix = normalized_basis(entropy_basis)
    for key in (
        f"traj_entropy_push_selected_frozen_{suffix}",
        f"traj_entropy_push_before_{suffix}",
        "traj_entropy_push_proxy",
    ):
        value = safe_float(row.get(key))
        if value is not None:
            return value
    return None


def summarize_file(path: Path, rollout_root: Path, eps: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    step = parse_step(path)
    dataset_dir = path.parent
    run_dir = dataset_dir.parent
    run_name = run_dir.name
    dataset = dataset_dir.name
    model, run_method = parse_run_name(run_name)
    relpath = str(path.relative_to(rollout_root)).replace("\\", "/")

    rows = list(iter_jsonl(path))
    if not rows:
        return {}, []

    first = rows[0]
    method = method_label(first, run_method)
    entropy_basis = str(first.get("entropy_logprob_basis") or "")
    select_basis = str(first.get("select_logprob_basis") or "")
    advantage_key = str(first.get("advantage_key") or "")

    counts = defaultdict(int)
    total_entropy_sum = 0.0
    total_tokens = 0.0
    traj_entropy_means: list[float] = []
    score_values: list[float] = []
    selected_entropy_push_values: list[float] = []
    selected_entropy_push_pos_values: list[float] = []
    selected_entropy_push_neg_values: list[float] = []

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("group_key", ""))].append(row)

        selected = safe_bool(row.get("selected_for_update"))
        select_prefix = "selected_adv" if selected else "deleted_adv"
        add_sign_count(counts, select_prefix, adv_value(row), eps)

        score_prefix = "selected_score" if selected else "deleted_score"
        add_sign_count(counts, score_prefix, score_value(row), eps)

        entropy_sum = safe_float(row.get("traj_entropy_sum"))
        token_count = safe_float(row.get("traj_num_tokens"))
        if entropy_sum is not None and token_count is not None and token_count > 0:
            total_entropy_sum += entropy_sum
            total_tokens += token_count

        traj_entropy = safe_float(row.get("traj_entropy_mean"))
        if traj_entropy is not None:
            traj_entropy_means.append(traj_entropy)

        score = score_value(row)
        if score is not None:
            score_values.append(score)

        if selected:
            selected_push = selected_entropy_push_value(row, entropy_basis)
            if selected_push is not None:
                selected_entropy_push_values.append(selected_push)
                if selected_push > eps:
                    selected_entropy_push_pos_values.append(selected_push)
                elif selected_push < -eps:
                    selected_entropy_push_neg_values.append(selected_push)

    group_rows: list[dict[str, Any]] = []
    cov_before_values: list[float] = []
    cov_after_values: list[float] = []
    cov_before_avg_values: list[float] = []
    cov_before_sum_values: list[float] = []
    trend_before_values: list[float] = []
    trend_after_values: list[float] = []
    trend_before_sum_values: list[float] = []
    trend_after_sum_values: list[float] = []
    selected_entropy_push_sum_by_group: list[float] = []
    selected_entropy_push_mean_by_group: list[float] = []
    selected_entropy_increase_group_proxy_values: list[float] = []
    selected_counts: list[float] = []
    group_sizes: list[float] = []
    all_zero_flags: list[float] = []
    fallback_flags: list[float] = []
    single_sign_flags: list[float] = []

    for group_key, group in groups.items():
        rep = group[0]
        selected_rows = [row for row in group if safe_bool(row.get("selected_for_update"))]
        selected_pushes = [
            value
            for value in (selected_entropy_push_value(row, entropy_basis) for row in selected_rows)
            if value is not None
        ]
        selected_push_sum = sum(selected_pushes)
        selected_push_mean = mean(selected_pushes)
        if selected_pushes:
            selected_entropy_push_sum_by_group.append(selected_push_sum)
            selected_entropy_push_mean_by_group.append(float(selected_push_mean))
            selected_entropy_increase_group_proxy_values.append(selected_push_sum / max(len(group), 1))

        cov_before = safe_float(rep.get("group_cov_before_select"))
        cov_after = safe_float(rep.get("group_cov_after_select"))
        cov_before_avg = safe_float(rep.get("group_cov_before_avglogprob"))
        cov_before_sum = safe_float(rep.get("group_cov_before_sumlogprob"))
        trend_before = safe_float(rep.get("group_entropy_grad_proxy_before_select"))
        trend_after = safe_float(rep.get("group_entropy_grad_proxy_after_select"))
        trend_before_sum = safe_float(rep.get("group_entropy_grad_proxy_before_sumlogprob"))
        trend_after_sum = safe_float(rep.get("group_entropy_grad_proxy_after_sumlogprob"))

        if trend_before is None and cov_before is not None:
            trend_before = -cov_before
        if trend_after is None and cov_after is not None:
            trend_after = -cov_after
        if trend_before_sum is None and cov_before_sum is not None:
            trend_before_sum = -cov_before_sum

        for value, sink in [
            (cov_before, cov_before_values),
            (cov_after, cov_after_values),
            (cov_before_avg, cov_before_avg_values),
            (cov_before_sum, cov_before_sum_values),
            (trend_before, trend_before_values),
            (trend_after, trend_after_values),
            (trend_before_sum, trend_before_sum_values),
            (trend_after_sum, trend_after_sum_values),
        ]:
            if value is not None:
                sink.append(value)

        selected_count = safe_float(rep.get("group_selected_count"))
        if selected_count is None:
            selected_count = float(len(selected_rows))
        selected_counts.append(selected_count)

        group_size = safe_float(rep.get("group_size"))
        if group_size is None:
            group_size = float(len(group))
        group_sizes.append(group_size)
        all_zero_flags.append(1.0 if safe_bool(rep.get("group_adv_all_zero")) else 0.0)
        fallback_flags.append(1.0 if safe_bool(rep.get("selector_random_fallback")) else 0.0)
        single_sign_flags.append(1.0 if safe_bool(rep.get("selector_single_sign_only")) else 0.0)

        group_rows.append(
            {
                "model": model,
                "run_name": run_name,
                "run_method": run_method,
                "dataset": dataset,
                "method": method,
                "step": step,
                "group_key": group_key,
                "group_size": int(group_size),
                "selected_count": int(selected_count),
                "cov_before_select": cov_before,
                "cov_after_select": cov_after,
                "entropy_trend_before_select": trend_before,
                "entropy_trend_after_select": trend_after,
                "cov_before_avglogprob": cov_before_avg,
                "cov_before_sumlogprob": cov_before_sum,
                "entropy_trend_before_sumlogprob": trend_before_sum,
                "selected_entropy_push_sum_frozen": selected_push_sum,
                "selected_entropy_push_mean_frozen": selected_push_mean,
                "selected_entropy_increase_group_proxy": selected_push_sum / max(len(group), 1),
                "entropy_basis": entropy_basis,
                "select_basis": select_basis,
                "advantage_key": advantage_key,
                "selector_random_fallback": safe_bool(rep.get("selector_random_fallback")),
                "selector_single_sign_only": safe_bool(rep.get("selector_single_sign_only")),
                "group_adv_all_zero": safe_bool(rep.get("group_adv_all_zero")),
            }
        )

    summary = {
        "model": model,
        "run_name": run_name,
        "run_method": run_method,
        "dataset": dataset,
        "method": method,
        "step": step,
        "step_bin_start": ((step - 1) // 10) * 10 + 1,
        "rollout_relpath": relpath,
        "file_name": path.name,
        "entropy_basis": entropy_basis,
        "select_basis": select_basis,
        "advantage_key": advantage_key,
        "select_positive_ratio": safe_float(first_nonempty(rows, "select_positive_ratio")),
        "total_rows": len(rows),
        "group_count": len(groups),
        "actor_entropy_token_mean": (total_entropy_sum / total_tokens) if total_tokens > 0 else None,
        "actor_entropy_seq_mean": mean(traj_entropy_means),
        "score_mean": mean(score_values),
        "mean_group_size": mean(group_sizes),
        "mean_group_selected_count": mean(selected_counts),
        "group_adv_all_zero_ratio": mean(all_zero_flags),
        "selector_random_fallback_ratio": mean(fallback_flags),
        "selector_single_sign_only_ratio": mean(single_sign_flags),
        "cov_before_select_mean": mean(cov_before_values),
        "cov_after_select_mean": mean(cov_after_values),
        "cov_delta_select_mean": (
            None
            if mean(cov_before_values) is None or mean(cov_after_values) is None
            else mean(cov_after_values) - mean(cov_before_values)
        ),
        "entropy_trend_before_select_mean": mean(trend_before_values),
        "entropy_trend_after_select_mean": mean(trend_after_values),
        "entropy_trend_delta_select_mean": (
            None
            if mean(trend_before_values) is None or mean(trend_after_values) is None
            else mean(trend_after_values) - mean(trend_before_values)
        ),
        "cov_before_avglogprob_mean": mean(cov_before_avg_values),
        "cov_before_sumlogprob_mean": mean(cov_before_sum_values),
        "entropy_trend_before_sumlogprob_mean": mean(trend_before_sum_values),
        "entropy_trend_after_sumlogprob_mean": mean(trend_after_sum_values),
        "selected_entropy_push_sum_frozen_total": sum(selected_entropy_push_values),
        "selected_entropy_push_mean_frozen_per_selected_traj": mean(selected_entropy_push_values),
        "selected_entropy_push_pos_sum_frozen_total": sum(selected_entropy_push_pos_values),
        "selected_entropy_push_neg_sum_frozen_total": sum(selected_entropy_push_neg_values),
        "selected_entropy_push_sum_frozen_mean_per_group": mean(selected_entropy_push_sum_by_group),
        "selected_entropy_push_mean_frozen_mean_per_group": mean(selected_entropy_push_mean_by_group),
        "selected_entropy_increase_group_proxy_mean": mean(selected_entropy_increase_group_proxy_values),
    }
    for key in [
        "selected_adv_pos",
        "selected_adv_neg",
        "selected_adv_zero",
        "deleted_adv_pos",
        "deleted_adv_neg",
        "deleted_adv_zero",
        "selected_score_pos",
        "selected_score_neg",
        "selected_score_zero",
        "deleted_score_pos",
        "deleted_score_neg",
        "deleted_score_zero",
    ]:
        summary[f"{key}_count"] = counts[key]

    return summary, group_rows


def aggregate_ten_step(step_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in step_rows:
        key = (row["model"], row["run_name"], row["dataset"], row["method"], row["step_bin_start"])
        grouped[key].append(row)

    count_cols = [
        "selected_adv_pos_count",
        "selected_adv_neg_count",
        "selected_adv_zero_count",
        "deleted_adv_pos_count",
        "deleted_adv_neg_count",
        "deleted_adv_zero_count",
        "selected_score_pos_count",
        "selected_score_neg_count",
        "deleted_score_pos_count",
        "deleted_score_neg_count",
    ]
    metric_cols = [
        "actor_entropy_token_mean",
        "cov_before_select_mean",
        "entropy_trend_before_select_mean",
        "cov_after_select_mean",
        "entropy_trend_after_select_mean",
        "selected_entropy_push_sum_frozen_total",
        "selected_entropy_push_mean_frozen_per_selected_traj",
        "selected_entropy_push_sum_frozen_mean_per_group",
        "selected_entropy_increase_group_proxy_mean",
    ]

    output: list[dict[str, Any]] = []
    for (model, run_name, dataset, method, step_bin_start), rows in sorted(grouped.items()):
        result = {
            "model": model,
            "run_name": run_name,
            "dataset": dataset,
            "method": method,
            "step_bin_start": step_bin_start,
            "step_bin_end": step_bin_start + 9,
            "step_count": len(rows),
            "min_step": min(row["step"] for row in rows),
            "max_step": max(row["step"] for row in rows),
        }
        for col in count_cols + metric_cols:
            vals = [safe_float(row.get(col)) for row in rows]
            vals = [value for value in vals if value is not None]
            result[f"{col}_mean"] = mean(vals)
        output.append(result)
    return output


def aggregate_overall_summary(step_rows: list[dict[str, Any]], group_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cov_group_values = [
        value
        for value in (safe_float(row.get("cov_before_select")) for row in group_rows)
        if value is not None
    ]
    cov_step_values = [
        value
        for value in (safe_float(row.get("cov_before_select_mean")) for row in step_rows)
        if value is not None
    ]
    selected_push_values = [
        value
        for value in (safe_float(row.get("selected_entropy_push_sum_frozen")) for row in group_rows)
        if value is not None
    ]
    selected_proxy_values = [
        value
        for value in (safe_float(row.get("selected_entropy_increase_group_proxy")) for row in group_rows)
        if value is not None
    ]

    overall = {
        "scope": "all_input",
        "step_count": len(step_rows),
        "group_count": len(group_rows),
        "overall_cov_before_select_mean": mean(cov_group_values),
        "overall_cov_before_select_mean_over_groups": mean(cov_group_values),
        "overall_cov_before_select_mean_over_steps": mean(cov_step_values),
        "selected_entropy_increase_total": sum(selected_push_values),
        "selected_entropy_increase_mean_per_group": mean(selected_push_values),
        "selected_entropy_push_sum_frozen_total": sum(selected_push_values),
        "selected_entropy_push_sum_frozen_mean_per_group": mean(selected_push_values),
        "selected_entropy_increase_group_proxy_mean": mean(selected_proxy_values),
    }
    return [overall]


def aggregate_cov_summary(step_rows: list[dict[str, Any]], group_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []

    def add_scope(scope: str, key_fields: tuple[str, ...]) -> None:
        step_grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in step_rows:
            step_grouped[tuple(row[field] for field in key_fields)].append(row)

        group_grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in group_rows:
            group_grouped[tuple(row[field] for field in key_fields)].append(row)

        for key, rows in sorted(step_grouped.items()):
            group_vals = [
                safe_float(row.get("cov_before_select"))
                for row in group_grouped.get(key, [])
                if safe_float(row.get("cov_before_select")) is not None
            ]
            step_cov_vals = [
                safe_float(row.get("cov_before_select_mean"))
                for row in rows
                if safe_float(row.get("cov_before_select_mean")) is not None
            ]
            step_trend_vals = [
                safe_float(row.get("entropy_trend_before_select_mean"))
                for row in rows
                if safe_float(row.get("entropy_trend_before_select_mean")) is not None
            ]
            result = {"scope": scope}
            for field, value in zip(key_fields, key, strict=False):
                result[field] = value
            result.update(
                {
                    "step_count": len(rows),
                    "group_count": len(group_vals),
                    "cov_before_select_mean_over_steps": mean(step_cov_vals),
                    "cov_before_select_mean_over_groups": mean(group_vals),
                    "entropy_trend_before_select_mean_over_steps": mean(step_trend_vals),
                }
            )
            output.append(result)

    add_scope("model_dataset_method", ("model", "dataset", "method"))
    add_scope("model_method", ("model", "method"))
    add_scope("run_dataset_method", ("run_name", "dataset", "method"))
    return output


def write_plots(output_dir: Path, step_rows: list[dict[str, Any]], ten_step_rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; CSV files were written, plots were skipped.")
        return

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    combined_dir = plot_dir / "combined"
    by_method_dir = plot_dir / "by_method"
    combined_dir.mkdir(parents=True, exist_ok=True)
    by_method_dir.mkdir(parents=True, exist_ok=True)

    def series_label(row: dict[str, Any]) -> str:
        return f"{row['run_name']} | {row['dataset']} | {row['method']}"

    def safe_path_name(text: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "unknown"

    def line_plot(
        rows: list[dict[str, Any]],
        y_key: str,
        title: str,
        ylabel: str,
        file_name: str,
        target_dir: Path,
    ) -> None:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[series_label(row)].append(row)
        if not grouped:
            return
        plt.figure(figsize=(13, 7))
        for label, items in sorted(grouped.items()):
            items = sorted(items, key=lambda item: item["step"])
            xs = [item["step"] for item in items]
            ys = [safe_float(item.get(y_key)) for item in items]
            plt.plot(xs, ys, label=label, linewidth=1.6)
        plt.title(title)
        plt.xlabel("step")
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.25)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(target_dir / file_name, dpi=180)
        plt.close()

    plot_specs = [
        (
            "actor_entropy_token_mean",
            "Actor Entropy By Step",
            "actor/entropy",
            "actor_entropy_by_step.png",
        ),
        (
            "cov_before_select_mean",
            "Proof Covariance By Step: group_cov_before_select",
            "Cov(A, logp proxy)",
            "cov_before_select_by_step.png",
        ),
        (
            "entropy_trend_before_select_mean",
            "Entropy Trend By Step: -group_cov_before_select",
            "-Cov(A, logp proxy)",
            "entropy_trend_by_step.png",
        ),
        (
            "selected_entropy_push_sum_frozen_total",
            "Selected Samples Entropy Increase By Step",
            "sum selected A * (surprisal - H_full)",
            "selected_entropy_increase_by_step.png",
        ),
        (
            "selected_entropy_increase_group_proxy_mean",
            "Selected Samples Entropy Increase Proxy By Step",
            "mean_group selected push sum / group_size",
            "selected_entropy_increase_group_proxy_by_step.png",
        ),
    ]

    for y_key, title, ylabel, file_name in plot_specs:
        line_plot(step_rows, y_key, title, ylabel, file_name, combined_dir)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ten_step_rows:
        grouped[series_label(row)].append(row)

    count_cols = [
        ("selected_adv_pos_count_mean", "selected adv > 0"),
        ("selected_adv_neg_count_mean", "selected adv < 0"),
        ("deleted_adv_pos_count_mean", "deleted adv > 0"),
        ("deleted_adv_neg_count_mean", "deleted adv < 0"),
    ]
    for label, items in sorted(grouped.items()):
        items = sorted(items, key=lambda item: item["step_bin_start"])
        plt.figure(figsize=(11, 6))
        for col, col_label in count_cols:
            plt.plot(
                [item["step_bin_start"] for item in items],
                [safe_float(item.get(col)) for item in items],
                label=col_label,
                linewidth=1.8,
            )
        plt.title(f"10-Step Mean Adv Selection Counts\n{label}")
        plt.xlabel("step bin start")
        plt.ylabel("count mean")
        plt.grid(True, alpha=0.25)
        plt.legend(fontsize=9)
        plt.tight_layout()
        safe_name = safe_path_name(label)
        method_dir = by_method_dir / safe_name
        method_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(method_dir / "adv_selection_counts_10step.png", dpi=180)
        plt.close()

    step_by_series: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in step_rows:
        step_by_series[series_label(row)].append(row)
    for label, rows in sorted(step_by_series.items()):
        method_dir = by_method_dir / safe_path_name(label)
        method_dir.mkdir(parents=True, exist_ok=True)
        for y_key, title, ylabel, file_name in plot_specs:
            line_plot(rows, y_key, f"{title}\n{label}", ylabel, file_name, method_dir)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze rollout entropy/covariance statistics by model, dataset, and method.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        type=Path,
        default=None,
        help=(
            "Input rollout directory. Accepts the rollout_logs root, one run directory, "
            "or one dataset directory containing step JSONL files."
        ),
    )
    parser.add_argument(
        "--rollout-root",
        type=Path,
        default=None,
        help="Backward-compatible alias for input_dir. Ignored if positional input_dir is provided.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for CSV summaries and plots.",
    )
    parser.add_argument("--eps", type=float, default=EPS_DEFAULT, help="Epsilon for sign counts.")
    parser.add_argument("--no-plots", action="store_true", help="Only write CSV files.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    input_dir = args.input_dir or args.rollout_root or DEFAULT_ROLLOUT_ROOT
    rollout_root, files = discover_rollout_files(input_dir)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    step_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    inventory: dict[tuple[Any, ...], dict[str, Any]] = {}

    print(f"Input directory: {Path(input_dir).resolve()}")
    print(f"Logical rollout root: {rollout_root}")
    print(f"Found {len(files)} JSONL files.")

    for idx, path in enumerate(files, start=1):
        summary, file_group_rows = summarize_file(path=path, rollout_root=rollout_root, eps=args.eps)
        if not summary:
            continue
        step_rows.append(summary)
        group_rows.extend(file_group_rows)
        inv_key = (
            summary["model"],
            summary["run_name"],
            summary["dataset"],
            summary["method"],
            summary["entropy_basis"],
            summary["select_basis"],
            summary["advantage_key"],
        )
        item = inventory.setdefault(
            inv_key,
            {
                "model": summary["model"],
                "run_name": summary["run_name"],
                "run_method": summary["run_method"],
                "dataset": summary["dataset"],
                "method": summary["method"],
                "entropy_basis": summary["entropy_basis"],
                "select_basis": summary["select_basis"],
                "advantage_key": summary["advantage_key"],
                "file_count": 0,
                "min_step": summary["step"],
                "max_step": summary["step"],
            },
        )
        item["file_count"] += 1
        item["min_step"] = min(item["min_step"], summary["step"])
        item["max_step"] = max(item["max_step"], summary["step"])

        if idx % 50 == 0:
            print(f"Processed {idx}/{len(files)} files...")

    ten_step_rows = aggregate_ten_step(step_rows)
    cov_summary_rows = aggregate_cov_summary(step_rows, group_rows)
    overall_summary_rows = aggregate_overall_summary(step_rows, group_rows)
    inventory_rows = sorted(inventory.values(), key=lambda row: (row["model"], row["dataset"], row["method"]))

    write_csv(output_dir / "step_summary.csv", step_rows)
    write_csv(output_dir / "group_summary.csv", group_rows)
    write_csv(output_dir / "ten_step_adv_counts.csv", ten_step_rows)
    write_csv(output_dir / "cov_summary.csv", cov_summary_rows)
    write_csv(output_dir / "overall_summary.csv", overall_summary_rows)
    write_csv(output_dir / "method_inventory.csv", inventory_rows)

    if not args.no_plots:
        write_plots(output_dir=output_dir, step_rows=step_rows, ten_step_rows=ten_step_rows)

    print(f"Wrote analysis outputs to {output_dir}")
    print("Main covariance column: group_cov_before_select -> step_summary.cov_before_select_mean")
    print("Overall covariance value: overall_summary.overall_cov_before_select_mean")
    print("Selected entropy increase: step_summary.selected_entropy_push_sum_frozen_total")


if __name__ == "__main__":
    main()
