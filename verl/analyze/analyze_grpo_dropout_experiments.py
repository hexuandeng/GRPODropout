from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from grpo_dropout_analysis_common import (
    ExperimentSpec,
    discover_experiments,
    read_jsonl,
    sort_step_files,
    summarize_group_rows,
    summarize_step_rows,
)


DEFAULT_LOG_DIR = _REPO_ROOT / "analyze" / "logs"
DEFAULT_ROLLOUT_ROOT = _REPO_ROOT / "rollout_logs"
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs" / "grpo_dropout_experiments"


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


def build_experiment_row(spec: ExperimentSpec, nonempty_files: int) -> dict[str, Any]:
    config = spec.config
    return {
        "experiment_id": spec.experiment_id,
        "rollout_relpath": spec.rollout_relpath,
        "rollout_dir": str(spec.rollout_dir),
        "log_path": str(spec.log_path) if spec.log_path is not None else None,
        "nonempty_rollout_files": nonempty_files,
        "model_path": config.get("MODEL_PATH"),
        "save_dir": config.get("SAVE_DIR"),
        "train_file": config.get("TRAIN_FILE"),
        "val_file": config.get("VAL_FILE"),
        "select_method": config.get("SELECT_METHOD"),
        "n_votes_per_prompt": config.get("N_VOTES_PER_PROMPT"),
        "n_update_samples_per_prompt": config.get("N_UPDATE_SAMPLES_PER_PROMPT"),
        "select_random_seed": config.get("SELECT_RANDOM_SEED"),
        "entropy_logprob_basis": config.get("ENTROPY_LOGPROB_BASIS"),
        "select_logprob_basis": config.get("SELECT_LOGPROB_BASIS"),
        "select_positive_ratio": config.get("SELECT_POSITIVE_RATIO"),
        "rollout_temperature": config.get("ROLLOUT_TEMPERATURE"),
        "rollout_n": config.get("ROLLOUT_N"),
        "actor_lr": config.get("ACTOR_LR"),
        "kl_loss_coef": config.get("KL_LOSS_COEF"),
        "total_epochs": config.get("TOTAL_EPOCHS"),
    }


def build_log_only_step_row(
    spec: ExperimentSpec,
    step: int,
) -> dict[str, Any]:
    return {
        "experiment_id": spec.experiment_id,
        "rollout_relpath": spec.rollout_relpath,
        "file_name": f"log_only_step_{step}",
        "step": step,
    }


def run_analysis(
    log_dir: Path,
    rollout_root: Path,
    output_dir: Path,
    target_step: int | None = None,
    max_experiments: int | None = None,
    max_files_per_experiment: int | None = None,
    sign_eps: float = 1e-12,
) -> None:
    print("Discovering experiments...", flush=True)
    experiments = discover_experiments(log_dir=log_dir, rollout_root=rollout_root)
    if max_experiments is not None:
        experiments = experiments[:max_experiments]

    if not experiments:
        raise FileNotFoundError("No experiments could be matched between log files and rollout directories.")

    print(f"Matched experiments: {len(experiments)}", flush=True)

    experiment_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    question_rows: list[dict[str, Any]] = []
    skipped_experiments: list[tuple[str, str]] = []

    for index, spec in enumerate(experiments, start=1):
        print(f"[{index}/{len(experiments)}] Analyzing {spec.experiment_id} -> {spec.rollout_relpath}", flush=True)
        rollout_files_all = [
            path for path in sort_step_files(list(spec.rollout_dir.glob("*.jsonl"))) if path.stat().st_size > 0
        ]

        if target_step is not None:
            step_to_file: dict[int, Path] = {}
            for file_path in rollout_files_all:
                try:
                    step_to_file[int(file_path.stem)] = file_path
                except ValueError:
                    continue

            eligible_steps = [step_value for step_value in step_to_file if step_value >= target_step]
            if not eligible_steps:
                skipped_experiments.append((spec.experiment_id, f"missing any step >= {target_step}"))
                print(f"  skipped: missing any step >= {target_step}", flush=True)
                continue

            rollout_files = [path for path in rollout_files_all if path.stem.isdigit() and int(path.stem) >= target_step]
        else:
            rollout_files = rollout_files_all

        if max_files_per_experiment is not None:
            rollout_files = rollout_files[:max_files_per_experiment]

        print(f"  rollout files to analyze: {len(rollout_files)}", flush=True)

        experiment_rows.append(build_experiment_row(spec=spec, nonempty_files=len(rollout_files)))
        seen_rollout_steps: set[int] = set()

        for file_path in rollout_files:
            rows = read_jsonl(file_path=file_path)
            if not rows:
                continue

            step = int(rows[0].get("step", file_path.stem))
            seen_rollout_steps.add(step)
            step_question_rows = summarize_group_rows(
                rows=rows,
                experiment_id=spec.experiment_id,
                rollout_relpath=spec.rollout_relpath,
                file_name=file_path.name,
                step=step,
                eps=sign_eps,
            )
            question_rows.extend(step_question_rows)

            step_row = summarize_step_rows(
                rows=rows,
                question_rows=step_question_rows,
                experiment_id=spec.experiment_id,
                rollout_relpath=spec.rollout_relpath,
                file_name=file_path.name,
                step=step,
                eps=sign_eps,
            )
            log_metrics = spec.log_metrics_by_step.get(step, {})
            for key, value in log_metrics.items():
                if key == "step":
                    continue
                step_row[f"log/{key}"] = value
            step_rows.append(step_row)

        missing_log_steps = sorted(
            step
            for step in spec.log_metrics_by_step
            if step not in seen_rollout_steps and (target_step is None or step >= target_step)
        )
        for step in missing_log_steps:
            step_row = build_log_only_step_row(spec=spec, step=step)
            log_metrics = spec.log_metrics_by_step.get(step, {})
            for key, value in log_metrics.items():
                if key == "step":
                    continue
                step_row[f"log/{key}"] = value
            step_rows.append(step_row)

    write_csv(output_dir / "experiments.csv", experiment_rows)
    write_csv(output_dir / "step_summary.csv", step_rows)
    write_csv(output_dir / "question_step_summary.csv", question_rows)

    print(f"Experiments analyzed: {len(experiment_rows)}")
    print(f"Saved: {output_dir / 'experiments.csv'}")
    print(f"Saved: {output_dir / 'step_summary.csv'}")
    print(f"Saved: {output_dir / 'question_step_summary.csv'}")
    if skipped_experiments:
        print(f"Skipped experiments: {len(skipped_experiments)}")
        for experiment_id, reason in skipped_experiments:
            print(f"Skipped: {experiment_id} ({reason})")


if __name__ == "__main__":
    log_dir = DEFAULT_LOG_DIR
    rollout_root = DEFAULT_ROLLOUT_ROOT
    output_dir = DEFAULT_OUTPUT_DIR

    # Optional limits for quick debugging. Set to None for full analysis.
    target_step = None
    max_experiments = None
    max_files_per_experiment = None
    sign_eps = 1e-12

    run_analysis(
        log_dir=log_dir,
        rollout_root=rollout_root,
        output_dir=output_dir,
        target_step=target_step,
        max_experiments=max_experiments,
        max_files_per_experiment=max_files_per_experiment,
        sign_eps=sign_eps,
    )
