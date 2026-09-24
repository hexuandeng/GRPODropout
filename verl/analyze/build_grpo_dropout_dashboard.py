from __future__ import annotations

from pathlib import Path

from analyze_grpo_dropout_experiments import run_analysis
from plot_grpo_dropout_dashboard import get_matplotlib_import_error, run_dashboard


BASE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[1]

# Edit these constants directly before running.
LOG_DIR = BASE_DIR / "logs"
ROLLOUT_ROOT = _REPO_ROOT / "rollout_logs"
TARGET_STEP: int | None = 0
ANALYSIS_OUTPUT_DIR = BASE_DIR / "outputs" / (f"grpo_dropout_experiments_step_{TARGET_STEP}" if TARGET_STEP is not None else "grpo_dropout_experiments")
DASHBOARD_OUTPUT_DIR = BASE_DIR / "outputs" / (f"grpo_dropout_dashboard_step_{TARGET_STEP}" if TARGET_STEP is not None else "grpo_dropout_dashboard")

EXPERIMENT_FILTERS: list[str] | None = None
ROLLOUT_FILTERS: list[str] | None = None
QUESTION_IDS_TO_PLOT: list[str] = ["0","4","47"]

MAX_EXPERIMENTS: int | None = None
MAX_FILES_PER_EXPERIMENT: int | None = None
SIGN_EPS = 1e-12


def _write_readme(analysis_dir: Path, dashboard_dir: Path, target_step: int | None) -> Path:
    model_dirs = sorted(path.name for path in dashboard_dir.iterdir() if path.is_dir()) if dashboard_dir.exists() else []
    question_trace_note = (
        "Question trace figures are generated only when `QUESTION_IDS_TO_PLOT` is non-empty in this file."
    )

    lines = [
        "# GRPO-Dropout Dashboard Output",
        "",
        f"- Target step: `{target_step}`" if target_step is not None else "- Target step: `None` (all available steps)",
        f"- Analysis CSV directory: `{analysis_dir}`",
        f"- Plot directory: `{dashboard_dir}`",
        "",
        "## Main CSV files",
        "",
        "- `batch_step_summary.csv`: batch-level time series summary. Each row is one experiment at one step.",
        "- `question_step_summary.csv`: question-level time series summary. Each row is one question at one step.",
        "- `experiments.csv` (in analysis dir): experiment metadata and matched rollout/log paths.",
        "",
        "## Plot organization",
        "",
        "Plots are grouped first by model, then by dataset. Each dataset subdirectory compares different settings of the same model on the same training dataset.",
        "",
    ]

    if model_dirs:
        lines.append("Detected model groups:")
        for model_dir in model_dirs:
            lines.append(f"- `{model_dir}`")
        lines.append("")

    lines.extend(
        [
            "## Common plot meanings",
            "",
            "- `actor_entropy.png`: token-level actor entropy across steps.",
            "- `batch_group_entropy_proxy_mean.png`: trajectory-level entropy proxy across steps.",
            "- `batch_group_entropy_grad_proxy_before_avglogprob_mean.png`: pre-selection entropy gradient proxy across steps under avg-logprob basis.",
            "- `batch_group_entropy_grad_proxy_after_avglogprob_mean.png`: post-selection entropy gradient proxy across steps under avg-logprob basis.",
            "- `batch_group_entropy_grad_proxy_before_sumlogprob_mean.png`: pre-selection entropy gradient proxy across steps under sum-logprob basis.",
            "- `batch_group_entropy_grad_proxy_after_sumlogprob_mean.png`: post-selection entropy gradient proxy across steps under sum-logprob basis.",
            "- `batch_vote_ratio_mean.png`: average answer consistency across the 8 questions in each step.",
            "- `batch_vote_tie_ratio.png`: fraction of questions with vote ties in each step.",
            "- `batch_unique_pred_count_mean.png`: average number of distinct predictions per question in each step.",
            "- `batch_oracle_pos_ratio.png`: oracle-correct rollout ratio in each step.",
            "- `batch_label_acc_mean.png`: average pseudo-label correctness at the question level.",
            "- `batch_pseudo_pos_ratio.png`: pseudo-positive rollout ratio in each step.",
            "- `batch_response_length_mean.png`: average response length across all rollouts in each step.",
            "- `batch_selected_response_length_mean.png`: average response length among selected rollouts.",
            "- `selected_count.png`: number of selected rollouts used for actor update in each step.",
            "- `selected_ratio.png`: selected rollout ratio in each step.",
            "- `selected_count_mean_per_question.png`: average selected rollout count per question.",
            "- `validation_*.png`: validation curves parsed from training logs.",
            "- `heldout_0/1/2_actor_entropy.png`: held-out actor entropy for the three tracked held-out questions.",
            "- `heldout_0/1/2_entropy_proxy_avglogprob.png`: held-out avg-logprob entropy proxy for the three tracked held-out questions.",
            "- `heldout_0/1/2_entropy_proxy_sumlogprob.png`: held-out sum-logprob entropy proxy for the three tracked held-out questions.",
            "- `heldout_mean_actor_entropy.png`: mean actor entropy across all held-out questions.",
            "- `heldout_mean_entropy_proxy_avglogprob.png`: mean avg-logprob entropy proxy across all held-out questions.",
            "- `heldout_mean_entropy_proxy_sumlogprob.png`: mean sum-logprob entropy proxy across all held-out questions.",
            "- `heldout_mean_response_length.png`: mean response length across all held-out questions.",
            "",
            "## Question trace figures",
            "",
            f"- {question_trace_note}",
            "- When enabled, figures are written to `question_traces/qid_<QUESTION_ID>.png` under each model directory.",
            "- Each question trace contains one subplot per metric, so each subplot compares only different experiment settings.",
            "",
            "## Typical reading order",
            "",
            "- Start with `validation_*.png` to see which setting improves external performance.",
            "- Then compare `actor_entropy.png` and `batch_group_entropy_proxy_mean.png` to understand entropy trends.",
            "- Use the `batch_group_entropy_grad_proxy_*_avglogprob_mean.png` and `batch_group_entropy_grad_proxy_*_sumlogprob_mean.png` figures for proof-related comparisons under both bases.",
            "- Check `batch_vote_ratio_mean.png`, `batch_label_acc_mean.png`, and `batch_oracle_pos_ratio.png` for consistency and correctness.",
            "- Finally inspect held-out plots and any question trace plots for per-question behavior.",
            "",
        ]
    )

    readme_path = dashboard_dir / "README.md"
    readme_path.parent.mkdir(parents=True, exist_ok=True)
    readme_path.write_text("\n".join(lines), encoding="utf-8")
    return readme_path


def main() -> None:
    print("Starting GRPO-dropout dashboard build...", flush=True)
    print(f"LOG_DIR={LOG_DIR}", flush=True)
    print(f"ROLLOUT_ROOT={ROLLOUT_ROOT}", flush=True)
    print(f"TARGET_STEP={TARGET_STEP}", flush=True)
    print(f"ANALYSIS_OUTPUT_DIR={ANALYSIS_OUTPUT_DIR}", flush=True)
    print(f"DASHBOARD_OUTPUT_DIR={DASHBOARD_OUTPUT_DIR}", flush=True)
    print(f"LOG_DIR exists={LOG_DIR.exists()}", flush=True)
    print(f"ROLLOUT_ROOT exists={ROLLOUT_ROOT.exists()}", flush=True)

    run_analysis(
        log_dir=LOG_DIR,
        rollout_root=ROLLOUT_ROOT,
        output_dir=ANALYSIS_OUTPUT_DIR,
        target_step=TARGET_STEP,
        max_experiments=MAX_EXPERIMENTS,
        max_files_per_experiment=MAX_FILES_PER_EXPERIMENT,
        sign_eps=SIGN_EPS,
    )
    print("Finished building CSV summaries. Starting plots...", flush=True)

    saved_paths = run_dashboard(
        analysis_dir=ANALYSIS_OUTPUT_DIR,
        output_dir=DASHBOARD_OUTPUT_DIR,
        experiment_filters=EXPERIMENT_FILTERS,
        rollout_filters=ROLLOUT_FILTERS,
        question_ids_to_plot=QUESTION_IDS_TO_PLOT,
    )

    print(f"Dashboard plots saved: {len(saved_paths)}", flush=True)
    if not saved_paths:
        matplotlib_error = get_matplotlib_import_error()
        if matplotlib_error is not None:
            print(
                f"No plots were saved because matplotlib could not be loaded: {matplotlib_error}",
                flush=True,
            )
        else:
            print(
                "No plots were saved. CSV summaries exist, so please check metric keys and filters.",
                flush=True,
            )
    for path in saved_paths:
        print(f"Saved: {path}", flush=True)

    readme_path = _write_readme(
        analysis_dir=ANALYSIS_OUTPUT_DIR,
        dashboard_dir=DASHBOARD_OUTPUT_DIR,
        target_step=TARGET_STEP,
    )
    print(f"Saved: {readme_path}", flush=True)


if __name__ == "__main__":
    main()