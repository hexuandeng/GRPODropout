"""Main entry point: chain the cosine + LLM-judge metrics, compute path diversity for
every (model, dataset, question) under root, and output per-question detail, per
(model, dataset) averages, and cross-model comparison.

Usage examples:
    python run_analysis.py                       # full run, both think and full
    python run_analysis.py --metrics cosine      # cosine only (no API needed)
    python run_analysis.py --max_questions 3     # small smoke test
    python run_analysis.py --models qwen3-4B-GRPO_base,qwen3-4B-GRPO_clip-cov
    python run_analysis.py --judge_path_text think   # judge on think only to save cost

When the judge endpoint is not configured, judge metrics are skipped automatically and only cosine is output.
"""
import argparse
import csv
import json
import os
from collections import defaultdict
from typing import Dict, List, Optional

import config
import cosine_diversity as cos
import answer_diversity as ans
from length_stats import length_stats
from data_loader import discover_groups, load_units


def _parse_list(s: Optional[str]) -> Optional[List[str]]:
    if not s or s.strip().lower() == "all":
        return None
    return [x.strip() for x in s.split(",") if x.strip()]


def _mean(xs: List[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def analyze(args):
    config.ensure_output_dirs()

    models = _parse_list(args.models)
    datasets = _parse_list(args.datasets)
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    cos_modes = [m.strip() for m in args.path_text.split(",") if m.strip()]
    judge_modes = [m.strip() for m in args.judge_path_text.split(",") if m.strip()]
    # rollout subset criterion: all / correct (correct only) / both (compute both)
    subset_arg = args.rollout_subset.strip().lower()
    if subset_arg == "both":
        subsets = ["all", "correct"]
    else:
        subsets = [subset_arg]

    want_cosine = "cosine" in metrics
    want_answer = "answer" in metrics
    want_length = "length" in metrics
    want_cluster = "judge_cluster" in metrics
    want_pairwise = "judge_pairwise" in metrics
    judge_needed = want_cluster or want_pairwise

    judge = None
    if judge_needed:
        if config.judge_configured():
            import llm_judge_diversity as judge  # noqa: N813
            print(f"[judge] using endpoint {config.JUDGE_BASE_URL} model={config.JUDGE_MODEL}")
        else:
            print("[judge] JUDGE_BASE_URL/JUDGE_API_KEY/JUDGE_MODEL not configured "
                  "-> skipping judge metrics, computing the rest only. Fill in the endpoint and rerun to add them.")
            want_cluster = want_pairwise = False
            judge_needed = False

    groups = discover_groups(config.DATA_ROOT if args.root is None else args.root,
                             models=models, datasets=datasets,
                             exclude_models=_parse_list(args.exclude_models),
                             exclude_datasets=_parse_list(args.exclude_datasets))
    if not groups:
        print(f"No <model>/<dataset>/test_*.jsonl found under {args.root or config.DATA_ROOT}")
        return

    print(f"Discovered {len(groups)} (model, dataset) groups:")
    for g in groups:
        print(f"  - {g.model} / {g.dataset}  <- {os.path.basename(g.jsonl_path)}")

    summary_rows: List[Dict] = []

    for g in groups:
        units = load_units(g, max_questions=args.max_questions)
        if not units:
            print(f"[skip] {g.model}/{g.dataset}: no valid questions")
            continue
        print(f"\n=== {g.model} / {g.dataset}: {len(units)} questions ===")

        per_q_records: List[Dict] = []
        # aggregation accumulators
        acc: Dict[str, List[float]] = defaultdict(list)

        for ui, u in enumerate(units):
            rec: Dict = {
                "id": u.qid,
                "question": u.question[:200],
                "n_rollouts": u.n,
                "n_correct": (sum(u.correctness) if u.correctness else None),
                "accuracy": u.accuracy,
            }

            for subset in subsets:
                # when the correct subset has fewer than 2 rollouts, pairwise metrics are meaningless; skip that subset (answer/length still recorded)
                n_sub = u.n_subset(subset)
                sub_tag = subset  # column-name suffix: __all / __correct

                if want_cosine:
                    for mode in cos_modes:
                        texts = u.path_texts(mode, subset)
                        stats = cos.cosine_diversity_for_texts(texts)
                        rec[f"cosine__{mode}__{sub_tag}"] = stats
                        if n_sub >= 2:
                            acc[f"cosine_div__{mode}__{sub_tag}"].append(stats["diversity"])

                if want_answer:
                    stats = ans.answer_diversity(
                        u.answers_subset(subset),
                        u.correctness_subset(subset),
                        u.gold_answer)
                    rec[f"answer__{sub_tag}"] = stats
                    if n_sub >= 1:
                        acc[f"answer_norm_entropy__{sub_tag}"].append(stats["normalized_entropy"])
                        acc[f"answer_n_distinct__{sub_tag}"].append(stats["n_distinct"])
                        acc[f"answer_distinct_ratio__{sub_tag}"].append(stats["distinct_ratio"])
                        acc[f"answer_top1_share__{sub_tag}"].append(stats["top1_share"])
                        acc[f"answer_empty_ratio__{sub_tag}"].append(stats["empty_ratio"])

                if want_length:
                    stats = length_stats(u.responses_subset(subset), u.answers_subset(subset))
                    rec[f"length__{sub_tag}"] = stats
                    if n_sub >= 1:
                        acc[f"char_len_mean__{sub_tag}"].append(stats.get("char_len_mean"))
                        acc[f"char_len_median__{sub_tag}"].append(stats.get("char_len_median"))
                        acc[f"word_len_mean__{sub_tag}"].append(stats.get("word_len_mean"))
                        acc[f"think_unclosed_ratio__{sub_tag}"].append(stats.get("think_unclosed_ratio"))
                        if stats.get("char_len_empty_mean") is not None:
                            acc[f"char_len_empty_mean__{sub_tag}"].append(stats["char_len_empty_mean"])
                        if stats.get("char_len_nonempty_mean") is not None:
                            acc[f"char_len_nonempty_mean__{sub_tag}"].append(stats["char_len_nonempty_mean"])

                if want_cluster and judge is not None and n_sub >= 2:
                    for mode in judge_modes:
                        stats = judge.cluster_diversity(u.question, u.path_texts(mode, subset))
                        rec[f"judge_cluster__{mode}__{sub_tag}"] = stats
                        acc[f"cluster_norm_entropy__{mode}__{sub_tag}"].append(stats["normalized_entropy"])
                        acc[f"cluster_n_distinct__{mode}__{sub_tag}"].append(stats["n_distinct"])

                if want_pairwise and judge is not None and n_sub >= 2:
                    for mode in judge_modes:
                        stats = judge.pairwise_diversity(u.question, u.path_texts(mode, subset),
                                                         seed=config.RANDOM_SEED)
                        rec[f"judge_pairwise__{mode}__{sub_tag}"] = stats
                        acc[f"pairwise_div__{mode}__{sub_tag}"].append(stats["pairwise_diversity"])
                        acc[f"distinct_path_ratio__{mode}__{sub_tag}"].append(stats["distinct_path_ratio"])

            per_q_records.append(rec)
            print(f"  [{ui + 1}/{len(units)}] id={u.qid} done", end="\r")

        if judge is not None:
            judge.flush()
        print()

        # write per-question detail
        pq_path = os.path.join(config.PER_QUESTION_DIR, f"{g.model}__{g.dataset}.json")
        with open(pq_path, "w", encoding="utf-8") as f:
            json.dump(per_q_records, f, ensure_ascii=False, indent=2)
        print(f"  [saved] {pq_path}")

        # (model, dataset) average row
        row = {"model": g.model, "dataset": g.dataset, "n_questions": len(per_q_records)}
        for key, vals in acc.items():
            row[key + "_mean"] = _mean(vals)
        summary_rows.append(row)

    _write_summaries(summary_rows)
    print("\nDone. See outputs/summary/ for the aggregates.")


def _write_summaries(summary_rows: List[Dict]):
    if not summary_rows:
        return
    # collect all metric columns
    metric_keys = sorted({k for r in summary_rows for k in r
                          if k not in ("model", "dataset", "n_questions")})

    # 1) per_model_dataset.csv
    pmd = os.path.join(config.SUMMARY_DIR, "per_model_dataset.csv")
    cols = ["model", "dataset", "n_questions"] + metric_keys
    with open(pmd, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in summary_rows:
            w.writerow({c: r.get(c, "") for c in cols})
    print(f"[saved] {pmd}")

    # 2) cross_model_compare.csv: re-average each model over its datasets (model-level score) + expand per dataset
    by_model: Dict[str, List[Dict]] = defaultdict(list)
    for r in summary_rows:
        by_model[r["model"]].append(r)
    cmc = os.path.join(config.SUMMARY_DIR, "cross_model_compare.csv")
    cols2 = ["model", "n_datasets"] + [f"{k}_overall" for k in metric_keys]
    with open(cmc, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols2)
        w.writeheader()
        for model, rows in sorted(by_model.items()):
            out = {"model": model, "n_datasets": len(rows)}
            for k in metric_keys:
                out[f"{k}_overall"] = _mean([r.get(k) for r in rows])
            w.writerow(out)
    print(f"[saved] {cmc}")

    # 3) also store a json copy for programmatic reading
    with open(os.path.join(config.SUMMARY_DIR, "per_model_dataset.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary_rows, f, ensure_ascii=False, indent=2)


def build_argparser():
    p = argparse.ArgumentParser(description="Rollout path diversity analysis")
    p.add_argument("--root", default=None,
                   help=f"root of the sampling outputs, default config.DATA_ROOT ({config.DATA_ROOT})")
    p.add_argument("--models", default="all", help="comma-separated model directory names (exact match), or all")
    p.add_argument("--exclude_models", default=None,
                   help="comma-separated model directory names to exclude (exact match, takes priority over --models)")
    p.add_argument("--datasets", default="all", help="comma-separated dataset names, or all")
    p.add_argument("--exclude_datasets", default=None,
                   help="comma-separated dataset names to exclude")
    p.add_argument("--metrics", default="cosine,answer,length,judge_cluster,judge_pairwise",
                   help="metrics to run: cosine,answer,length,judge_cluster,judge_pairwise")
    p.add_argument("--path_text", default=",".join(config.PATH_TEXT_MODES),
                   help="path-text views for cosine, comma-separated: think,full")
    p.add_argument("--judge_path_text", default=",".join(config.JUDGE_PATH_TEXT_MODES),
                   help="path-text views for the judge (think only to save cost)")
    p.add_argument("--rollout_subset", default="both",
                   help="which rollouts to evaluate: all / correct (correct only) / both (default)")
    p.add_argument("--max_questions", type=int, default=None,
                   help="max questions to analyze per group (for smoke tests)")
    return p


if __name__ == "__main__":
    analyze(build_argparser().parse_args())
