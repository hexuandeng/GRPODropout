from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_ANALYSIS_DIR = BASE_DIR / "outputs" / "grpo_dropout_experiments"
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs" / "grpo_dropout_plots"


DEFAULT_METRIC_GROUPS: list[tuple[str, list[tuple[str, str]]]] = [
    (
        "entropy_overview",
        [
            ("log/actor/entropy", "Actor Entropy"),
            ("question_entropy_proxy_mean", "Question Entropy Proxy Mean"),
            ("question_entropy_proxy_std", "Question Entropy Proxy Std"),
            ("question_entropy_proxy_max", "Question Entropy Proxy Max"),
        ],
    ),
    (
        "selection_overview",
        [
            ("selected_oracle_pos_ratio", "Selected Oracle Positive Ratio"),
            ("selected_pseudo_pos_ratio", "Selected Pseudo Positive Ratio"),
            ("selected_push_pos_ratio", "Selected Push Positive Ratio"),
            ("selected_push_neg_ratio", "Selected Push Negative Ratio"),
        ],
    ),
    (
        "quality_overview",
        [
            ("traj_oracle_acc_mean", "Trajectory Oracle Accuracy"),
            ("question_label_acc_mean", "Question Label Accuracy"),
            ("question_true_answer_ratio_mean", "Question True Answer Ratio"),
            ("question_vote_ratio_mean", "Question Vote Ratio"),
        ],
    ),
    (
        "length_overview",
        [
            ("mean_response_length", "Mean Response Length"),
            ("selected_mean_response_length", "Selected Mean Response Length"),
            ("question_unique_pred_count_mean", "Question Unique Pred Count Mean"),
            ("question_vote_tie_ratio", "Question Vote Tie Ratio"),
        ],
    ),
]


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if math.isnan(number):
        return None
    return number


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
        experiment_text = experiment_id.lower()
        if not any(token.lower() in experiment_text for token in experiment_filters):
            return False
    if rollout_filters:
        rollout_text = rollout_relpath.lower()
        if not any(token.lower() in rollout_text for token in rollout_filters):
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


def plot_metric_groups(
    step_rows: list[dict[str, str]],
    output_dir: Path,
    metric_groups: list[tuple[str, list[tuple[str, str]]]],
    figure_title_prefix: str = "",
) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("matplotlib is required to generate plots.") from exc

    if not step_rows:
        raise ValueError("No rows were available for plotting after filtering.")

    by_experiment: dict[str, list[dict[str, str]]] = {}
    for row in step_rows:
        label = build_label(row)
        by_experiment.setdefault(label, []).append(row)

    for label in by_experiment:
        by_experiment[label] = sort_rows_by_step(by_experiment[label])

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    for group_name, metrics in metric_groups:
        fig, axes = plt.subplots(len(metrics), 1, figsize=(13, 3.6 * len(metrics)), sharex=True)
        if len(metrics) == 1:
            axes = [axes]

        for ax, (metric_key, title) in zip(axes, metrics, strict=False):
            plotted = False
            for label, rows in by_experiment.items():
                xs: list[int] = []
                ys: list[float] = []
                for row in rows:
                    step = safe_float(row.get("step"))
                    value = safe_float(row.get(metric_key))
                    if step is None or value is None:
                        continue
                    xs.append(int(step))
                    ys.append(value)
                if not xs:
                    continue
                plotted = True
                ax.plot(xs, ys, marker="o", linewidth=1.5, markersize=3, label=label)

            ax.set_title(title)
            ax.grid(True, linestyle="--", alpha=0.35)
            if plotted:
                ax.legend(fontsize=8)
            else:
                ax.text(0.5, 0.5, f"No data for {metric_key}", ha="center", va="center", transform=ax.transAxes)

        axes[-1].set_xlabel("Step")
        title_prefix = f"{figure_title_prefix} - " if figure_title_prefix else ""
        fig.suptitle(f"{title_prefix}{group_name.replace('_', ' ').title()}", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.97))

        file_path = output_dir / f"{group_name}.png"
        fig.savefig(file_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(file_path)

    return saved_paths


def run_plotting(
    analysis_dir: Path,
    output_dir: Path,
    experiment_filters: list[str] | None = None,
    rollout_filters: list[str] | None = None,
    metric_groups: list[tuple[str, list[tuple[str, str]]]] | None = None,
) -> list[Path]:
    step_summary_path = analysis_dir / "step_summary.csv"
    if not step_summary_path.exists():
        raise FileNotFoundError(f"step_summary.csv was not found at: {step_summary_path}")

    rows = read_csv_rows(step_summary_path)
    filtered_rows = [
        row for row in rows if include_row(row, experiment_filters=experiment_filters, rollout_filters=rollout_filters)
    ]

    title_parts: list[str] = []
    if experiment_filters:
        title_parts.append("exp=" + ",".join(experiment_filters))
    if rollout_filters:
        title_parts.append("rollout=" + ",".join(rollout_filters))

    return plot_metric_groups(
        step_rows=filtered_rows,
        output_dir=output_dir,
        metric_groups=metric_groups or DEFAULT_METRIC_GROUPS,
        figure_title_prefix=" | ".join(title_parts),
    )


if __name__ == "__main__":
    analysis_dir = DEFAULT_ANALYSIS_DIR
    output_dir = DEFAULT_OUTPUT_DIR

    # Set to None to keep all experiments.
    experiment_filters = None

    # Example: ["AMC23"] or ["AIME24"] to only keep part of rollout_relpath.
    rollout_filters = None

    # Replace with your own metric groups if needed.
    metric_groups = DEFAULT_METRIC_GROUPS

    saved_paths = run_plotting(
        analysis_dir=analysis_dir,
        output_dir=output_dir,
        experiment_filters=experiment_filters,
        rollout_filters=rollout_filters,
        metric_groups=metric_groups,
    )

    print(f"Saved {len(saved_paths)} plot files.")
    for path in saved_paths:
        print(f"Saved: {path}")
