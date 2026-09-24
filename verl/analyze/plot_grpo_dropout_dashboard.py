from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_ANALYSIS_DIR = BASE_DIR / "outputs" / "grpo_dropout_experiments"
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs" / "grpo_dropout_dashboard"

# Edit these directly before running.
EXPERIMENT_FILTERS: list[str] | None = None
ROLLOUT_FILTERS: list[str] | None = None
QUESTION_IDS_TO_PLOT: list[str] = []

_MATPLOTLIB_IMPORT_ERROR: Exception | None = None
_MATPLOTLIB_ERROR_REPORTED = False


def load_matplotlib_pyplot():
    global _MATPLOTLIB_IMPORT_ERROR
    global _MATPLOTLIB_ERROR_REPORTED

    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        return plt
    except Exception as exc:
        _MATPLOTLIB_IMPORT_ERROR = exc
        if not _MATPLOTLIB_ERROR_REPORTED:
            print(f"[plot_grpo_dropout_dashboard] matplotlib unavailable: {exc}", flush=True)
            _MATPLOTLIB_ERROR_REPORTED = True
        return None


def get_matplotlib_import_error() -> Exception | None:
    return _MATPLOTLIB_IMPORT_ERROR


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return None if math.isnan(number) else number
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return None if math.isnan(number) else number


def read_csv_rows(file_path: Path) -> list[dict[str, str]]:
    with file_path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def include_row(
    row: dict[str, str],
    experiment_filters: list[str] | None,
    rollout_filters: list[str] | None,
) -> bool:
    experiment_id = str(row.get("experiment_id", ""))
    rollout_relpath = str(row.get("rollout_relpath", ""))

    if experiment_filters:
        lowered = experiment_id.lower()
        if not any(token.lower() in lowered for token in experiment_filters):
            return False
    if rollout_filters:
        lowered = rollout_relpath.lower()
        if not any(token.lower() in lowered for token in rollout_filters):
            return False
    return True


def build_label(row: dict[str, str]) -> str:
    experiment_id = str(row.get("experiment_id", "")).strip()
    rollout_relpath = str(row.get("rollout_relpath", "")).strip()
    return experiment_id or rollout_relpath or "unknown_experiment"


def sort_rows_by_step(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    def _key(row: dict[str, str]) -> tuple[int, str]:
        step = safe_float(row.get("step"))
        return (int(step) if step is not None else 10**18, build_label(row))

    return sorted(rows, key=_key)


def write_csv(file_path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def metric_has_data(rows: list[dict[str, str]], metric_key: str) -> bool:
    return any(safe_float(row.get(metric_key)) is not None for row in rows)


def split_by_experiment(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(build_label(row), []).append(row)
    for label in grouped:
        grouped[label] = sort_rows_by_step(grouped[label])
    return grouped


MODEL_GROUP_RE = re.compile(r"(qwen\d+(?:\.\d+)?-[0-9.]+b)", re.IGNORECASE)


def infer_model_group(row: dict[str, str]) -> str:
    candidates = [str(row.get("experiment_id", "")), str(row.get("rollout_relpath", ""))]
    for candidate in candidates:
        match = MODEL_GROUP_RE.search(candidate)
        if match:
            return match.group(1).lower()
    return "unknown_model"


def infer_dataset_group(row: dict[str, str]) -> str:
    def _normalize_dataset_name(name: str) -> str:
        normalized = name.strip()
        changed = True
        while changed:
            changed = False
            for suffix in ("-sum", "_sum", "-none", "_none"):
                if normalized.lower().endswith(suffix):
                    trimmed = normalized[: -len(suffix)] or normalized
                    if trimmed != normalized:
                        normalized = trimmed
                        changed = True
                    break
        return normalized

    rollout_relpath = str(row.get("rollout_relpath", "")).strip().replace("\\", "/")
    if rollout_relpath:
        parts = [part for part in rollout_relpath.split("/") if part]
        if len(parts) >= 2:
            return _normalize_dataset_name(parts[-1])

    for key in ("train_file", "val_file"):
        file_path = str(row.get(key, "")).strip().replace("\\", "/")
        if not file_path:
            continue
        stem = Path(file_path).stem
        for suffix in ("_verl_train", "_verl_val", "_train", "_val"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        if stem:
            return _normalize_dataset_name(stem)

    if rollout_relpath:
        parts = [part for part in rollout_relpath.split("/") if part]
        if parts:
            return _normalize_dataset_name(parts[-1])
    return "unknown_dataset"


def is_sum_variant(row: dict[str, str]) -> bool:
    candidates = [str(row.get("experiment_id", "")), str(row.get("rollout_relpath", ""))]
    return any("sum" in candidate.lower() for candidate in candidates)


def is_base_variant(row: dict[str, str]) -> bool:
    candidates = [str(row.get("experiment_id", "")), str(row.get("rollout_relpath", ""))]
    lowered = " ".join(candidate.lower() for candidate in candidates)
    return "grpo_base" in lowered or "_base" in lowered or "ttrl_base" in lowered


def rows_for_sum_vs_base(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if is_sum_variant(row) or is_base_variant(row)]


def rows_without_sum(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if not is_sum_variant(row)]


def discover_validation_metrics(rows: list[dict[str, str]]) -> list[tuple[str, str, str]]:
    metric_keys: set[str] = set()
    for row in rows:
        for key in row:
            if key.startswith("log/validation/"):
                metric_keys.add(key)

    def _sort_key(metric_key: str) -> tuple[str, str]:
        tail = metric_key.removeprefix("log/validation/")
        if "_best_at_" in tail:
            kind = "2"
        elif "_maj_at_" in tail:
            kind = "1"
        elif "_mean_at_" in tail:
            kind = "0"
        else:
            kind = "9"
        return (tail.split("_", 1)[0], f"{kind}:{tail}")

    discovered: list[tuple[str, str, str]] = []
    for metric_key in sorted(metric_keys, key=_sort_key):
        tail = metric_key.removeprefix("log/validation/")
        label = tail.replace("_", " ")
        title = f"Validation {label}"
        discovered.append((metric_key, title, label))
    return discovered


def plot_dual_axis(
    rows: list[dict[str, str]],
    output_path: Path,
    left_metric: str,
    right_metric: str,
    title: str,
    left_label: str,
    right_label: str,
) -> Path | None:
    plt = load_matplotlib_pyplot()
    if plt is None:
        return None

    grouped = split_by_experiment(rows)
    if not grouped:
        return None

    fig, ax_left = plt.subplots(figsize=(13, 6))
    ax_right = ax_left.twinx()

    cmap = plt.get_cmap("tab10")
    legend_handles = []
    legend_labels = []
    plotted = False

    for index, (label, series) in enumerate(grouped.items()):
        color = cmap(index % 10)

        xs_left: list[int] = []
        ys_left: list[float] = []
        xs_right: list[int] = []
        ys_right: list[float] = []

        for row in series:
            step = safe_float(row.get("step"))
            left_value = safe_float(row.get(left_metric))
            right_value = safe_float(row.get(right_metric))
            if step is not None and left_value is not None:
                xs_left.append(int(step))
                ys_left.append(left_value)
            if step is not None and right_value is not None:
                xs_right.append(int(step))
                ys_right.append(right_value)

        if xs_left:
            plotted = True
            (line_left,) = ax_left.plot(xs_left, ys_left, color=color, linewidth=1.8, label=f"{label} | {left_label}")
            legend_handles.append(line_left)
            legend_labels.append(f"{label} | {left_label}")
        if xs_right:
            plotted = True
            (line_right,) = ax_right.plot(
                xs_right, ys_right, color=color, linewidth=1.8, linestyle="--", label=f"{label} | {right_label}"
            )
            legend_handles.append(line_right)
            legend_labels.append(f"{label} | {right_label}")

    ax_left.set_title(title)
    ax_left.set_xlabel("Step")
    ax_left.set_ylabel(left_label)
    ax_right.set_ylabel(right_label)
    ax_left.grid(True, linestyle="--", alpha=0.35)
    if plotted:
        ax_left.legend(legend_handles, legend_labels, fontsize=8, loc="best")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_metric_panel(
    rows: list[dict[str, str]],
    output_path: Path,
    title: str,
    panels: list[tuple[str, list[tuple[str, str]]]],
) -> Path | None:
    plt = load_matplotlib_pyplot()
    if plt is None:
        return None

    grouped = split_by_experiment(rows)
    if not grouped:
        return None

    fig, axes = plt.subplots(len(panels), 1, figsize=(13, 4.2 * len(panels)), sharex=True)
    if len(panels) == 1:
        axes = [axes]

    for ax, (panel_title, metrics) in zip(axes, panels, strict=False):
        plotted = False
        for label, series in grouped.items():
            for metric_key, metric_label in metrics:
                xs: list[int] = []
                ys: list[float] = []
                for row in series:
                    step = safe_float(row.get("step"))
                    value = safe_float(row.get(metric_key))
                    if step is None or value is None:
                        continue
                    xs.append(int(step))
                    ys.append(value)
                if not xs:
                    continue
                plotted = True
                ax.plot(xs, ys, linewidth=1.6, label=f"{label} | {metric_label}")
        ax.set_title(panel_title)
        ax.grid(True, linestyle="--", alpha=0.35)
        if plotted:
            ax.legend(fontsize=8, loc="best")
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)

    axes[-1].set_xlabel("Step")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_single_metric(
    rows: list[dict[str, str]],
    output_path: Path,
    metric_key: str,
    title: str,
    y_label: str,
) -> Path | None:
    plt = load_matplotlib_pyplot()
    if plt is None:
        return None

    grouped = split_by_experiment(rows)
    if not grouped:
        return None

    fig, ax = plt.subplots(figsize=(13, 5.5))
    plotted = False

    for label, series in grouped.items():
        xs: list[int] = []
        ys: list[float] = []
        for row in series:
            step = safe_float(row.get("step"))
            value = safe_float(row.get(metric_key))
            if step is None or value is None:
                continue
            xs.append(int(step))
            ys.append(value)
        if not xs:
            continue
        plotted = True
        ax.plot(xs, ys, linewidth=1.8, label=label)

    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel(y_label)
    ax.grid(True, linestyle="--", alpha=0.35)
    if plotted:
        ax.legend(fontsize=8, loc="best")
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def maybe_plot_single_metric(
    rows: list[dict[str, str]],
    output_path: Path,
    metric_key: str,
    title: str,
    y_label: str,
) -> Path | None:
    if not metric_has_data(rows, metric_key):
        return None
    return plot_single_metric(
        rows=rows,
        output_path=output_path,
        metric_key=metric_key,
        title=title,
        y_label=y_label,
    )


def plot_question_trace(
    rows: list[dict[str, str]],
    output_path: Path,
    question_id: str,
) -> Path | None:
    plt = load_matplotlib_pyplot()
    if plt is None:
        return None

    filtered_all = [row for row in rows if str(row.get("question_id")) == str(question_id)]
    filtered_no_sum = rows_without_sum(filtered_all)
    if not filtered_all:
        return None

    fig, axes = plt.subplots(8, 1, figsize=(13, 28), sharex=True)
    panel_specs = [
        ("group_entropy_proxy", "Entropy Proxy"),
        ("group_entropy_grad_proxy_before_select", "Entropy Grad Before"),
        ("group_entropy_grad_proxy_after_select", "Entropy Grad After"),
        ("true_answer_ratio", "True Answer Ratio"),
        ("label_acc", "Label Accuracy"),
        ("vote_ratio", "Vote Ratio"),
        ("unique_pred_count", "Unique Pred Count"),
        ("selected_ratio", "Selected Ratio"),
    ]

    for ax, (metric_key, metric_label) in zip(axes, panel_specs, strict=False):
        plotted = False
        values_for_limits: list[float] = []
        metric_rows = filtered_no_sum if metric_key in {
            "group_entropy_proxy",
            "group_entropy_grad_proxy_before_select",
            "group_entropy_grad_proxy_after_select",
        } else filtered_all
        grouped = split_by_experiment(metric_rows)
        for label, series in grouped.items():
            xs: list[int] = []
            ys: list[float] = []
            for row in series:
                step = safe_float(row.get("step"))
                value = safe_float(row.get(metric_key))
                if step is None or value is None:
                    continue
                xs.append(int(step))
                ys.append(value)
                values_for_limits.append(value)
            if not xs:
                continue
            plotted = True
            ax.plot(xs, ys, linewidth=1.6, label=label)
        ax.set_title(metric_label)
        ax.grid(True, linestyle="--", alpha=0.35)
        if plotted:
            ax.legend(fontsize=8, loc="best")

        if values_for_limits:
            min_value = min(values_for_limits)
            max_value = max(values_for_limits)
            if max_value - min_value < 0.05 and 0.0 <= min_value <= 1.0 and 0.0 <= max_value <= 1.0:
                pad = max(0.01, (max_value - min_value) * 0.25)
                ax.set_ylim(min_value - pad, max_value + pad)

    axes[-1].set_xlabel("Step")
    fig.suptitle(f"Question Trace - qid={question_id}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_question_entropy_compare(
    rows: list[dict[str, str]],
    output_path: Path,
    question_id: str,
) -> Path | None:
    plt = load_matplotlib_pyplot()
    if plt is None:
        return None

    filtered = [row for row in rows if str(row.get("question_id")) == str(question_id)]
    grouped = split_by_experiment(filtered)
    if not grouped:
        return None

    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
    panel_specs = [
        ("group_entropy_proxy", "Entropy Proxy"),
        ("group_entropy_grad_proxy_before_select", "Entropy Grad Before"),
        ("group_entropy_grad_proxy_after_select", "Entropy Grad After"),
    ]

    for ax, (metric_key, metric_label) in zip(axes, panel_specs, strict=False):
        plotted = False
        for label, series in grouped.items():
            xs: list[int] = []
            ys: list[float] = []
            for row in series:
                step = safe_float(row.get("step"))
                value = safe_float(row.get(metric_key))
                if step is None or value is None:
                    continue
                xs.append(int(step))
                ys.append(value)
            if not xs:
                continue
            plotted = True
            ax.plot(xs, ys, linewidth=1.7, label=label)
        ax.set_title(metric_label)
        ax.grid(True, linestyle="--", alpha=0.35)
        if plotted:
            ax.legend(fontsize=8, loc="best")
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)

    axes[-1].set_xlabel("Step")
    fig.suptitle(f"Question Entropy Compare - qid={question_id} (base vs sum)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def run_dashboard(
    analysis_dir: Path,
    output_dir: Path,
    experiment_filters: list[str] | None = None,
    rollout_filters: list[str] | None = None,
    question_ids_to_plot: list[str] | None = None,
) -> list[Path]:
    step_summary_path = analysis_dir / "step_summary.csv"
    question_summary_path = analysis_dir / "question_step_summary.csv"
    if not step_summary_path.exists():
        raise FileNotFoundError(f"step_summary.csv was not found at: {step_summary_path}")
    if not question_summary_path.exists():
        raise FileNotFoundError(f"question_step_summary.csv was not found at: {question_summary_path}")

    step_rows_all = read_csv_rows(step_summary_path)
    question_rows_all = read_csv_rows(question_summary_path)
    step_rows = [
        row for row in step_rows_all if include_row(row, experiment_filters=experiment_filters, rollout_filters=rollout_filters)
    ]
    question_rows = [
        row for row in question_rows_all
        if include_row(row, experiment_filters=experiment_filters, rollout_filters=rollout_filters)
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "batch_step_summary.csv", step_rows)
    write_csv(output_dir / "question_step_summary.csv", question_rows)

    step_rows_by_model_dataset: dict[str, dict[str, list[dict[str, str]]]] = {}
    question_rows_by_model_dataset: dict[str, dict[str, list[dict[str, str]]]] = {}
    for row in step_rows:
        model_group = infer_model_group(row)
        dataset_group = infer_dataset_group(row)
        step_rows_by_model_dataset.setdefault(model_group, {}).setdefault(dataset_group, []).append(row)
    for row in question_rows:
        model_group = infer_model_group(row)
        dataset_group = infer_dataset_group(row)
        question_rows_by_model_dataset.setdefault(model_group, {}).setdefault(dataset_group, []).append(row)

    saved_paths: list[Path] = []

    for model_group in sorted(step_rows_by_model_dataset):
        model_output_dir = output_dir / model_group
        model_output_dir.mkdir(parents=True, exist_ok=True)

        for dataset_group in sorted(step_rows_by_model_dataset[model_group]):
            model_step_rows = step_rows_by_model_dataset[model_group][dataset_group]
            model_question_rows = question_rows_by_model_dataset.get(model_group, {}).get(dataset_group, [])
            non_sum_step_rows = rows_without_sum(model_step_rows)
            model_sum_compare_step_rows = rows_for_sum_vs_base(model_step_rows)
            model_sum_compare_question_rows = rows_for_sum_vs_base(model_question_rows)
            dataset_output_dir = model_output_dir / dataset_group
            dataset_output_dir.mkdir(parents=True, exist_ok=True)
            write_csv(dataset_output_dir / "batch_step_summary.csv", model_step_rows)
            write_csv(dataset_output_dir / "question_step_summary.csv", model_question_rows)

            for maybe_path in [
                maybe_plot_single_metric(
                    rows=model_step_rows,
                    output_path=dataset_output_dir / "actor_entropy.png",
                    metric_key="log/actor/entropy",
                    title=f"{model_group} - {dataset_group} - Actor Entropy",
                    y_label="Actor Entropy",
                ),
                maybe_plot_single_metric(
                    rows=non_sum_step_rows,
                    output_path=dataset_output_dir / "batch_group_entropy_proxy_mean.png",
                    metric_key="batch_group_entropy_proxy_mean",
                    title=f"{model_group} - {dataset_group} - Batch Group Entropy Proxy Mean",
                    y_label="Batch Group Entropy Proxy Mean",
                ),
                maybe_plot_single_metric(
                    rows=non_sum_step_rows,
                    output_path=dataset_output_dir / "batch_group_entropy_grad_proxy_before_avglogprob_mean.png",
                    metric_key="batch_group_entropy_grad_proxy_before_avglogprob_mean",
                    title=f"{model_group} - {dataset_group} - Batch Group Entropy Grad Proxy Before Mean (avglogprob)",
                    y_label="Before Mean (avglogprob)",
                ),
                maybe_plot_single_metric(
                    rows=non_sum_step_rows,
                    output_path=dataset_output_dir / "batch_group_entropy_grad_proxy_after_avglogprob_mean.png",
                    metric_key="batch_group_entropy_grad_proxy_after_avglogprob_mean",
                    title=f"{model_group} - {dataset_group} - Batch Group Entropy Grad Proxy After Mean (avglogprob)",
                    y_label="After Mean (avglogprob)",
                ),
                maybe_plot_single_metric(
                    rows=non_sum_step_rows,
                    output_path=dataset_output_dir / "batch_group_entropy_grad_proxy_before_sumlogprob_mean.png",
                    metric_key="batch_group_entropy_grad_proxy_before_sumlogprob_mean",
                    title=f"{model_group} - {dataset_group} - Batch Group Entropy Grad Proxy Before Mean (sumlogprob)",
                    y_label="Before Mean (sumlogprob)",
                ),
                maybe_plot_single_metric(
                    rows=non_sum_step_rows,
                    output_path=dataset_output_dir / "batch_group_entropy_grad_proxy_after_sumlogprob_mean.png",
                    metric_key="batch_group_entropy_grad_proxy_after_sumlogprob_mean",
                    title=f"{model_group} - {dataset_group} - Batch Group Entropy Grad Proxy After Mean (sumlogprob)",
                    y_label="After Mean (sumlogprob)",
                ),
                maybe_plot_single_metric(
                    rows=model_step_rows,
                    output_path=dataset_output_dir / "batch_vote_ratio_mean.png",
                    metric_key="batch_vote_ratio_mean",
                    title=f"{model_group} - {dataset_group} - Batch Vote Ratio Mean",
                    y_label="Vote Ratio Mean",
                ),
                maybe_plot_single_metric(
                    rows=model_step_rows,
                    output_path=dataset_output_dir / "batch_vote_tie_ratio.png",
                    metric_key="batch_vote_tie_ratio",
                    title=f"{model_group} - {dataset_group} - Batch Vote Tie Ratio",
                    y_label="Vote Tie Ratio",
                ),
                maybe_plot_single_metric(
                    rows=model_step_rows,
                    output_path=dataset_output_dir / "batch_unique_pred_count_mean.png",
                    metric_key="batch_unique_pred_count_mean",
                    title=f"{model_group} - {dataset_group} - Batch Unique Pred Count Mean",
                    y_label="Unique Pred Count Mean",
                ),
                maybe_plot_single_metric(
                    rows=model_step_rows,
                    output_path=dataset_output_dir / "batch_oracle_pos_ratio.png",
                    metric_key="batch_oracle_pos_ratio",
                    title=f"{model_group} - {dataset_group} - Batch Oracle Positive Ratio",
                    y_label="Oracle Positive Ratio",
                ),
                maybe_plot_single_metric(
                    rows=model_step_rows,
                    output_path=dataset_output_dir / "batch_label_acc_mean.png",
                    metric_key="batch_label_acc_mean",
                    title=f"{model_group} - {dataset_group} - Batch Label Accuracy Mean",
                    y_label="Label Accuracy Mean",
                ),
                maybe_plot_single_metric(
                    rows=model_step_rows,
                    output_path=dataset_output_dir / "batch_pseudo_pos_ratio.png",
                    metric_key="batch_pseudo_pos_ratio",
                    title=f"{model_group} - {dataset_group} - Batch Pseudo Positive Ratio",
                    y_label="Pseudo Positive Ratio",
                ),
                maybe_plot_single_metric(
                    rows=model_step_rows,
                    output_path=dataset_output_dir / "batch_response_length_mean.png",
                    metric_key="batch_response_length_mean",
                    title=f"{model_group} - {dataset_group} - Batch Response Length Mean",
                    y_label="Response Length Mean",
                ),
                maybe_plot_single_metric(
                    rows=model_step_rows,
                    output_path=dataset_output_dir / "batch_selected_response_length_mean.png",
                    metric_key="batch_selected_response_length_mean",
                    title=f"{model_group} - {dataset_group} - Batch Selected Response Length Mean",
                    y_label="Selected Response Length Mean",
                ),
                maybe_plot_single_metric(
                    rows=model_step_rows,
                    output_path=dataset_output_dir / "selected_count.png",
                    metric_key="log/selection/selected_count",
                    title=f"{model_group} - {dataset_group} - Selected Count",
                    y_label="Selected Count",
                ),
                maybe_plot_single_metric(
                    rows=model_step_rows,
                    output_path=dataset_output_dir / "selected_ratio.png",
                    metric_key="log/selection/selected_ratio",
                    title=f"{model_group} - {dataset_group} - Selected Ratio",
                    y_label="Selected Ratio",
                ),
                maybe_plot_single_metric(
                    rows=model_step_rows,
                    output_path=dataset_output_dir / "selected_count_mean_per_question.png",
                    metric_key="batch_selected_count_mean_per_question",
                    title=f"{model_group} - {dataset_group} - Selected Count Mean Per Question",
                    y_label="Selected Count Mean / Question",
                ),
                maybe_plot_single_metric(
                    rows=model_step_rows,
                    output_path=dataset_output_dir / "heldout_0_actor_entropy.png",
                    metric_key="log/probe/heldout_0/actor_entropy",
                    title=f"{model_group} - {dataset_group} - heldout_0 Actor Entropy",
                    y_label="Actor Entropy",
                ),
                maybe_plot_single_metric(
                    rows=model_step_rows,
                    output_path=dataset_output_dir / "heldout_0_entropy_proxy_avglogprob.png",
                    metric_key="log/probe/heldout_0/group_entropy_proxy_avglogprob",
                    title=f"{model_group} - {dataset_group} - heldout_0 Entropy Proxy (avglogprob)",
                    y_label="Entropy Proxy (avglogprob)",
                ),
                maybe_plot_single_metric(
                    rows=model_step_rows,
                    output_path=dataset_output_dir / "heldout_0_entropy_proxy_sumlogprob.png",
                    metric_key="log/probe/heldout_0/group_entropy_proxy_sumlogprob",
                    title=f"{model_group} - {dataset_group} - heldout_0 Entropy Proxy (sumlogprob)",
                    y_label="Entropy Proxy (sumlogprob)",
                ),
                maybe_plot_single_metric(
                    rows=model_step_rows,
                    output_path=dataset_output_dir / "heldout_1_actor_entropy.png",
                    metric_key="log/probe/heldout_1/actor_entropy",
                    title=f"{model_group} - {dataset_group} - heldout_1 Actor Entropy",
                    y_label="Actor Entropy",
                ),
                maybe_plot_single_metric(
                    rows=model_step_rows,
                    output_path=dataset_output_dir / "heldout_1_entropy_proxy_avglogprob.png",
                    metric_key="log/probe/heldout_1/group_entropy_proxy_avglogprob",
                    title=f"{model_group} - {dataset_group} - heldout_1 Entropy Proxy (avglogprob)",
                    y_label="Entropy Proxy (avglogprob)",
                ),
                maybe_plot_single_metric(
                    rows=model_step_rows,
                    output_path=dataset_output_dir / "heldout_1_entropy_proxy_sumlogprob.png",
                    metric_key="log/probe/heldout_1/group_entropy_proxy_sumlogprob",
                    title=f"{model_group} - {dataset_group} - heldout_1 Entropy Proxy (sumlogprob)",
                    y_label="Entropy Proxy (sumlogprob)",
                ),
                maybe_plot_single_metric(
                    rows=model_step_rows,
                    output_path=dataset_output_dir / "heldout_2_actor_entropy.png",
                    metric_key="log/probe/heldout_2/actor_entropy",
                    title=f"{model_group} - {dataset_group} - heldout_2 Actor Entropy",
                    y_label="Actor Entropy",
                ),
                maybe_plot_single_metric(
                    rows=model_step_rows,
                    output_path=dataset_output_dir / "heldout_2_entropy_proxy_avglogprob.png",
                    metric_key="log/probe/heldout_2/group_entropy_proxy_avglogprob",
                    title=f"{model_group} - {dataset_group} - heldout_2 Entropy Proxy (avglogprob)",
                    y_label="Entropy Proxy (avglogprob)",
                ),
                maybe_plot_single_metric(
                    rows=model_step_rows,
                    output_path=dataset_output_dir / "heldout_2_entropy_proxy_sumlogprob.png",
                    metric_key="log/probe/heldout_2/group_entropy_proxy_sumlogprob",
                    title=f"{model_group} - {dataset_group} - heldout_2 Entropy Proxy (sumlogprob)",
                    y_label="Entropy Proxy (sumlogprob)",
                ),
                maybe_plot_single_metric(
                    rows=model_step_rows,
                    output_path=dataset_output_dir / "heldout_mean_actor_entropy.png",
                    metric_key="log/probe/mean/actor_entropy",
                    title=f"{model_group} - {dataset_group} - Held-Out Mean Actor Entropy",
                    y_label="Actor Entropy Mean",
                ),
                maybe_plot_single_metric(
                    rows=model_step_rows,
                    output_path=dataset_output_dir / "heldout_mean_entropy_proxy_avglogprob.png",
                    metric_key="log/probe/mean/group_entropy_proxy_avglogprob",
                    title=f"{model_group} - {dataset_group} - Held-Out Mean Entropy Proxy (avglogprob)",
                    y_label="Entropy Proxy Mean (avglogprob)",
                ),
                maybe_plot_single_metric(
                    rows=model_step_rows,
                    output_path=dataset_output_dir / "heldout_mean_entropy_proxy_sumlogprob.png",
                    metric_key="log/probe/mean/group_entropy_proxy_sumlogprob",
                    title=f"{model_group} - {dataset_group} - Held-Out Mean Entropy Proxy (sumlogprob)",
                    y_label="Entropy Proxy Mean (sumlogprob)",
                ),
                maybe_plot_single_metric(
                    rows=model_step_rows,
                    output_path=dataset_output_dir / "heldout_mean_response_length.png",
                    metric_key="log/probe/mean/response_length_mean",
                    title=f"{model_group} - {dataset_group} - Held-Out Mean Response Length",
                    y_label="Response Length Mean",
                ),
            ]:
                if maybe_path is not None:
                    saved_paths.append(maybe_path)

            for metric_key, title_suffix, y_label in discover_validation_metrics(model_step_rows):
                safe_name = metric_key.removeprefix("log/").replace("/", "__")
                maybe_path = maybe_plot_single_metric(
                    rows=model_step_rows,
                    output_path=dataset_output_dir / f"{safe_name}.png",
                    metric_key=metric_key,
                    title=f"{model_group} - {dataset_group} - {title_suffix}",
                    y_label=y_label,
                )
                if maybe_path is not None:
                    saved_paths.append(maybe_path)

            if model_sum_compare_step_rows:
                for maybe_path in [
                    maybe_plot_single_metric(
                        rows=model_sum_compare_step_rows,
                        output_path=dataset_output_dir / "batch_group_entropy_proxy_mean_sum_vs_base.png",
                        metric_key="batch_group_entropy_proxy_mean",
                        title=f"{model_group} - {dataset_group} - Batch Group Entropy Proxy Mean (base vs sum)",
                        y_label="Batch Group Entropy Proxy Mean",
                    ),
                    maybe_plot_single_metric(
                        rows=model_sum_compare_step_rows,
                        output_path=dataset_output_dir / "batch_group_entropy_grad_proxy_before_avglogprob_mean_sum_vs_base.png",
                        metric_key="batch_group_entropy_grad_proxy_before_avglogprob_mean",
                        title=f"{model_group} - {dataset_group} - Batch Group Entropy Grad Proxy Before Mean (avglogprob, base vs sum)",
                        y_label="Before Mean (avglogprob)",
                    ),
                    maybe_plot_single_metric(
                        rows=model_sum_compare_step_rows,
                        output_path=dataset_output_dir / "batch_group_entropy_grad_proxy_after_avglogprob_mean_sum_vs_base.png",
                        metric_key="batch_group_entropy_grad_proxy_after_avglogprob_mean",
                        title=f"{model_group} - {dataset_group} - Batch Group Entropy Grad Proxy After Mean (avglogprob, base vs sum)",
                        y_label="After Mean (avglogprob)",
                    ),
                    maybe_plot_single_metric(
                        rows=model_sum_compare_step_rows,
                        output_path=dataset_output_dir / "batch_group_entropy_grad_proxy_before_sumlogprob_mean_sum_vs_base.png",
                        metric_key="batch_group_entropy_grad_proxy_before_sumlogprob_mean",
                        title=f"{model_group} - {dataset_group} - Batch Group Entropy Grad Proxy Before Mean (sumlogprob, base vs sum)",
                        y_label="Before Mean (sumlogprob)",
                    ),
                    maybe_plot_single_metric(
                        rows=model_sum_compare_step_rows,
                        output_path=dataset_output_dir / "batch_group_entropy_grad_proxy_after_sumlogprob_mean_sum_vs_base.png",
                        metric_key="batch_group_entropy_grad_proxy_after_sumlogprob_mean",
                        title=f"{model_group} - {dataset_group} - Batch Group Entropy Grad Proxy After Mean (sumlogprob, base vs sum)",
                        y_label="After Mean (sumlogprob)",
                    ),
                ]:
                    if maybe_path is not None:
                        saved_paths.append(maybe_path)

            for question_id in question_ids_to_plot or []:
                maybe_path = plot_question_trace(
                    rows=model_question_rows,
                    output_path=dataset_output_dir / "question_traces" / f"qid_{question_id}.png",
                    question_id=str(question_id),
                )
                if maybe_path is not None:
                    saved_paths.append(maybe_path)
                if model_sum_compare_question_rows:
                    maybe_path = plot_question_entropy_compare(
                        rows=model_sum_compare_question_rows,
                        output_path=dataset_output_dir / "question_traces" / f"qid_{question_id}_entropy_sum_vs_base.png",
                        question_id=str(question_id),
                    )
                    if maybe_path is not None:
                        saved_paths.append(maybe_path)

    return saved_paths


if __name__ == "__main__":
    analysis_dir = DEFAULT_ANALYSIS_DIR
    output_dir = DEFAULT_OUTPUT_DIR

    saved_paths = run_dashboard(
        analysis_dir=analysis_dir,
        output_dir=output_dir,
        experiment_filters=EXPERIMENT_FILTERS,
        rollout_filters=ROLLOUT_FILTERS,
        question_ids_to_plot=QUESTION_IDS_TO_PLOT,
    )

    print(f"Saved {len(saved_paths)} plot files.")
    for path in saved_paths:
        print(f"Saved: {path}")
