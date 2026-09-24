from __future__ import annotations

import csv
import math
import os
import sys
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from grpo_dropout_analysis_common import (
    discover_experiments,
    load_dataset_view_map,
    mean,
    read_jsonl,
    safe_bool,
    safe_float,
    sort_step_files,
)


DEFAULT_LOG_DIR = _REPO_ROOT / "analyze" / "logs"
DEFAULT_ROLLOUT_ROOT = _REPO_ROOT / "rollout_logs"
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs" / "grpo_dropout_heldout"

_DATA_ROOT = Path(os.environ.get("DATA_ROOT", _REPO_ROOT.parent / "data"))
DEFAULT_DATASET_VIEWS = {
    "aime24": _DATA_ROOT / "AIME24/AIME24_verl_val_view.json",
    "amc23": _DATA_ROOT / "AMC23/AMC23_verl_train_view.json",
}


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


def maybe_plot(file_path: Path, rows: list[dict[str, Any]], experiment_ids: list[str]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    if not rows:
        return

    by_experiment: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_experiment.setdefault(str(row["experiment_id"]), []).append(row)

    fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)
    metrics = [
        ("mean_avg_token_prob", "Mean Avg Token Prob"),
        ("group_entropy_proxy", "Group Entropy Proxy"),
        ("oracle_pos_ratio", "Oracle Positive Ratio"),
        ("mean_response_length", "Mean Response Length"),
    ]
    for ax, (metric_key, title) in zip(axes, metrics, strict=False):
        for experiment_id in experiment_ids:
            series = sorted(by_experiment.get(experiment_id, []), key=lambda item: int(item["step"]))
            if not series:
                continue
            xs = [int(item["step"]) for item in series]
            ys = [safe_float(item.get(metric_key)) for item in series]
            ax.plot(xs, ys, marker="o", linewidth=1.5, markersize=3, label=experiment_id)
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("Step")
    fig.tight_layout()
    file_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(file_path, dpi=200, bbox_inches="tight")


def run_tracking(
    log_dir: Path,
    rollout_root: Path,
    output_dir: Path,
    question_id: str,
    dataset: str,
    dataset_view: Path | None = None,
    experiment_filter: str | None = None,
    max_files_per_experiment: int | None = None,
) -> None:
    dataset_key = dataset.strip().lower()
    dataset_view_path = dataset_view or DEFAULT_DATASET_VIEWS.get(dataset_key)
    if dataset_view_path is None:
        raise FileNotFoundError(f"No default dataset view is configured for dataset={dataset!r}.")
    dataset_map = load_dataset_view_map(dataset_view_path=dataset_view_path)
    dataset_item = dataset_map.get(str(question_id))

    experiments = discover_experiments(log_dir=log_dir, rollout_root=rollout_root)
    if experiment_filter:
        experiments = [exp for exp in experiments if experiment_filter.lower() in exp.experiment_id.lower()]

    trace_rows: list[dict[str, Any]] = []
    experiment_ids: list[str] = []

    for spec in experiments:
        if dataset.lower() not in spec.rollout_relpath.lower():
            continue
        experiment_ids.append(spec.experiment_id)
        rollout_files = [path for path in sort_step_files(list(spec.rollout_dir.glob("*.jsonl"))) if path.stat().st_size > 0]
        if max_files_per_experiment is not None:
            rollout_files = rollout_files[:max_files_per_experiment]

        for file_path in rollout_files:
            rows = read_jsonl(file_path=file_path)
            if not rows:
                continue
            matching_rows = [row for row in rows if str(row.get("question_id")) == str(question_id)]
            if not matching_rows:
                continue

            selected_rows = [row for row in matching_rows if safe_bool(row.get("selected_for_update"))]
            avg_logprobs = [safe_float(row.get("traj_avg_logprob", row.get("traj_logp_avglogprob"))) for row in matching_rows]
            selected_logprobs = [
                safe_float(row.get("traj_avg_logprob", row.get("traj_logp_avglogprob"))) for row in selected_rows
            ]
            response_lengths = [safe_float(row.get("traj_num_tokens")) for row in matching_rows]
            selected_response_lengths = [safe_float(row.get("traj_num_tokens")) for row in selected_rows]
            oracle_values = [safe_float(row.get("oracle_acc")) for row in matching_rows]
            pseudo_values = [safe_float(row.get("score")) for row in matching_rows]
            group_entropy_proxy = safe_float(matching_rows[0].get("group_entropy_proxy"))
            group_entropy_grad_proxy_before = safe_float(matching_rows[0].get("group_entropy_grad_proxy_before_select"))
            group_entropy_grad_proxy_after = safe_float(matching_rows[0].get("group_entropy_grad_proxy_after_select"))
            group_cov_before = safe_float(matching_rows[0].get("group_cov_before_select"))
            group_cov_after = safe_float(matching_rows[0].get("group_cov_after_select"))

            trace_rows.append(
                {
                    "experiment_id": spec.experiment_id,
                    "rollout_relpath": spec.rollout_relpath,
                    "step": int(matching_rows[0].get("step", file_path.stem)),
                    "file_name": file_path.name,
                    "question_id": question_id,
                    "prompt_text": (
                        dataset_item.get("extra_info", {}).get("raw_problem")
                        if dataset_item is not None
                        else None
                    ),
                    "ground_truth": (
                        dataset_item.get("reward_model", {}).get("ground_truth")
                        if dataset_item is not None
                        else None
                    ),
                    "group_size": len(matching_rows),
                    "selected_count": len(selected_rows),
                    "vote_answer": matching_rows[0].get("vote_answer"),
                    "vote_ratio": safe_float(matching_rows[0].get("vote_ratio")),
                    "vote_tie": safe_bool(matching_rows[0].get("vote_tie")),
                    "unique_pred_count": len(
                        {
                            str(row.get("pred", "")).strip()
                            for row in matching_rows
                            if str(row.get("pred", "")).strip()
                        }
                    ),
                    "label_acc": safe_float(matching_rows[0].get("label_acc")),
                    "true_answer_ratio": safe_float(matching_rows[0].get("true_answer_ratio")),
                    "group_reward_acc": safe_float(matching_rows[0].get("group_reward_acc")),
                    "group_reward_mean": safe_float(matching_rows[0].get("group_reward_mean")),
                    "group_entropy_proxy": group_entropy_proxy,
                    "group_entropy_grad_proxy_before_select": group_entropy_grad_proxy_before,
                    "group_entropy_grad_proxy_after_select": group_entropy_grad_proxy_after,
                    "group_entropy_grad_proxy_delta": (
                        None
                        if group_entropy_grad_proxy_before is None or group_entropy_grad_proxy_after is None
                        else group_entropy_grad_proxy_after - group_entropy_grad_proxy_before
                    ),
                    "group_cov_before_select": group_cov_before,
                    "group_cov_after_select": group_cov_after,
                    "group_cov_delta": (
                        None if group_cov_before is None or group_cov_after is None else group_cov_after - group_cov_before
                    ),
                    "mean_traj_avg_logprob": mean([v for v in avg_logprobs if v is not None]),
                    "selected_mean_traj_avg_logprob": mean([v for v in selected_logprobs if v is not None]),
                    "mean_avg_token_prob": mean([math.exp(v) for v in avg_logprobs if v is not None]),
                    "selected_mean_avg_token_prob": mean(
                        [math.exp(v) for v in selected_logprobs if v is not None]
                    ),
                    "mean_response_length": mean([v for v in response_lengths if v is not None]),
                    "selected_mean_response_length": mean([v for v in selected_response_lengths if v is not None]),
                    "oracle_pos_ratio": mean([v for v in oracle_values if v is not None]),
                    "pseudo_pos_ratio": mean([1.0 if v is not None and v > 0 else 0.0 for v in pseudo_values if v is not None]),
                    "selected_oracle_pos_ratio": mean(
                        [safe_float(row.get("oracle_acc")) for row in selected_rows if safe_float(row.get("oracle_acc")) is not None]
                    ),
                    "selected_pseudo_pos_ratio": mean(
                        [
                            1.0 if safe_float(row.get("score")) is not None and safe_float(row.get("score")) > 0 else 0.0
                            for row in selected_rows
                            if safe_float(row.get("score")) is not None
                        ]
                    ),
                }
            )

    trace_rows = sorted(trace_rows, key=lambda row: (row["experiment_id"], int(row["step"])))
    save_dir = output_dir / f"{dataset_key}_qid_{question_id}"
    write_csv(save_dir / "heldout_trace.csv", trace_rows)
    maybe_plot(save_dir / "heldout_trace.png", trace_rows, experiment_ids=sorted(set(experiment_ids)))

    print(f"Tracked rows: {len(trace_rows)}")
    print(f"Saved: {save_dir / 'heldout_trace.csv'}")
    if dataset_item is not None:
        print(f"Held-out prompt source: {dataset_view_path}")
    else:
        print("Held-out prompt text was not found in the dataset view file.")


if __name__ == "__main__":
    log_dir = DEFAULT_LOG_DIR
    rollout_root = DEFAULT_ROLLOUT_ROOT
    output_dir = DEFAULT_OUTPUT_DIR

    # Edit these directly before running.
    question_id = "1"
    dataset = "amc23"
    dataset_view = None
    experiment_filter = None
    max_files_per_experiment = None

    run_tracking(
        log_dir=log_dir,
        rollout_root=rollout_root,
        output_dir=output_dir,
        question_id=question_id,
        dataset=dataset,
        dataset_view=dataset_view,
        experiment_filter=experiment_filter,
        max_files_per_experiment=max_files_per_experiment,
    )