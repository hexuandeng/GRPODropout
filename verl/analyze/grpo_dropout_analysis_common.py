from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
SHELL_ASSIGN_RE = re.compile(r"^\+\s*([A-Z][A-Z0-9_]+)=(.*)$")
STEP_LINE_RE = re.compile(r"step:(\d+)\s+-\s+")
VALIDATION_FULL_LINE_RE = re.compile(
    r"\[validation\]\[step=(\d+)\]\s+([^:]+):\s+mean@(\d+)=([^\s|]+)\s+\|\s+maj@\3/mean=([^\s|]+)\s+\|\s+best@\3/mean=([^\s|]+)"
)
VALIDATION_MEAN_ONLY_LINE_RE = re.compile(
    r"\[validation\]\[step=(\d+)\]\s+([^:]+):\s+mean@(\d+)=([^\s|]+)"
)
HELDOUT_LINE_RE = re.compile(r"\[heldout_probe\]\[step=(\d+)\]\[repeat=(\d+)\]\s+(.*)")


@dataclass
class ExperimentSpec:
    experiment_id: str
    rollout_dir: Path
    rollout_relpath: str
    log_path: Path | None
    config: dict[str, Any]
    log_metrics_by_step: dict[int, dict[str, Any]]


def strip_ansi(text: str) -> str:
    return ANSI_ESCAPE_RE.sub("", text)


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    wrappers = ("np.float64(", "np.float32(", "tensor(", "array(")
    for prefix in wrappers:
        if text.startswith(prefix) and text.endswith(")"):
            text = text[len(prefix) : -1].strip()
            break
    if text.lower() in {"nan", "none", "null"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def safe_int(value: Any) -> int | None:
    number = safe_float(value)
    if number is None:
        return None
    return int(number)


def safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    left = int(math.floor(pos))
    right = int(math.ceil(pos))
    if left == right:
        return ordered[left]
    alpha = pos - left
    return ordered[left] * (1.0 - alpha) + ordered[right] * alpha


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def std(values: list[float]) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    avg = sum(values) / len(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / len(values))


def canonical_rollout_relpath(value: str | Path | None) -> str | None:
    if value is None:
        return None
    text = str(value).replace("\\", "/").strip()
    marker = "/rollout_logs/"
    lower = text.lower()
    if marker in lower:
        start = lower.index(marker) + len(marker)
        text = text[start:]
    return text.strip("/")


def discover_rollout_dirs(rollout_root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    candidate_dirs: dict[str, Path] = {}

    for path in sorted(rollout_root.glob("*")):
        if not path.is_dir():
            continue
        has_nonempty = any(file_path.stat().st_size > 0 for file_path in path.glob("*.jsonl"))
        if has_nonempty:
            relpath = path.relative_to(rollout_root).as_posix()
            candidate_dirs[relpath] = path

    for path in sorted(rollout_root.glob("*/*")):
        if not path.is_dir():
            continue
        has_nonempty = any(file_path.stat().st_size > 0 for file_path in path.glob("*.jsonl"))
        if has_nonempty:
            relpath = path.relative_to(rollout_root).as_posix()
            candidate_dirs[relpath] = path

    result.update(candidate_dirs)
    return result


def parse_log_file(log_path: Path) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    config: dict[str, Any] = {}
    step_metrics: dict[int, dict[str, Any]] = {}

    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = strip_ansi(raw_line).strip()
            if not line:
                continue

            assign_match = SHELL_ASSIGN_RE.match(line)
            if assign_match:
                config[assign_match.group(1)] = assign_match.group(2).strip()
                continue

            validation_match = VALIDATION_FULL_LINE_RE.search(line)
            if validation_match:
                step = int(validation_match.group(1))
                dataset = validation_match.group(2).strip().lower().replace("/", "_")
                at_k = validation_match.group(3).strip()
                metrics = step_metrics.setdefault(step, {"step": step})
                metrics[f"validation/{dataset}_mean_at_{at_k}"] = safe_float(validation_match.group(4))
                metrics[f"validation/{dataset}_maj_at_{at_k}_mean"] = safe_float(validation_match.group(5))
                metrics[f"validation/{dataset}_best_at_{at_k}_mean"] = safe_float(validation_match.group(6))
                continue

            validation_mean_only_match = VALIDATION_MEAN_ONLY_LINE_RE.search(line)
            if validation_mean_only_match:
                step = int(validation_mean_only_match.group(1))
                dataset = validation_mean_only_match.group(2).strip().lower().replace("/", "_")
                at_k = validation_mean_only_match.group(3).strip()
                metrics = step_metrics.setdefault(step, {"step": step})
                metrics[f"validation/{dataset}_mean_at_{at_k}"] = safe_float(validation_mean_only_match.group(4))
                continue

            heldout_match = HELDOUT_LINE_RE.search(line)
            if heldout_match:
                step = int(heldout_match.group(1))
                metrics = step_metrics.setdefault(step, {"step": step})
                payload = heldout_match.group(3).strip()
                actor_entropy_values: list[float] = []
                avg_logprob_values: list[float] = []
                avg_prob_values: list[float] = []
                entropy_proxy_avg_values: list[float] = []
                entropy_proxy_sum_values: list[float] = []
                response_length_values: list[float] = []

                for chunk in [part.strip() for part in payload.split(" ; ") if part.strip()]:
                    if ":" not in chunk:
                        continue
                    question_id, body = chunk.split(":", 1)
                    question_id = question_id.strip()
                    metric_pairs = {}
                    for token in body.strip().split():
                        if "=" not in token:
                            continue
                        key, value = token.split("=", 1)
                        metric_pairs[key.strip()] = safe_float(value.strip())

                    actor_entropy = metric_pairs.get("actor_H")
                    avg_logprob = metric_pairs.get("avg_logp")
                    avg_prob = metric_pairs.get("avg_prob")
                    entropy_proxy_avg = metric_pairs.get("H_avg")
                    entropy_proxy_sum = metric_pairs.get("H_sum")
                    response_length = metric_pairs.get("resp_len")

                    metrics[f"probe/{question_id}/actor_entropy"] = actor_entropy
                    metrics[f"probe/{question_id}/avg_token_logprob"] = avg_logprob
                    metrics[f"probe/{question_id}/avg_token_prob"] = avg_prob
                    metrics[f"probe/{question_id}/group_entropy_proxy_avglogprob"] = entropy_proxy_avg
                    metrics[f"probe/{question_id}/group_entropy_proxy_sumlogprob"] = entropy_proxy_sum
                    metrics[f"probe/{question_id}/response_length_mean"] = response_length

                    if actor_entropy is not None:
                        actor_entropy_values.append(actor_entropy)
                    if avg_logprob is not None:
                        avg_logprob_values.append(avg_logprob)
                    if avg_prob is not None:
                        avg_prob_values.append(avg_prob)
                    if entropy_proxy_avg is not None:
                        entropy_proxy_avg_values.append(entropy_proxy_avg)
                    if entropy_proxy_sum is not None:
                        entropy_proxy_sum_values.append(entropy_proxy_sum)
                    if response_length is not None:
                        response_length_values.append(response_length)

                metrics["probe/mean/actor_entropy"] = mean(actor_entropy_values)
                metrics["probe/mean/avg_token_logprob"] = mean(avg_logprob_values)
                metrics["probe/mean/avg_token_prob"] = mean(avg_prob_values)
                metrics["probe/mean/group_entropy_proxy_avglogprob"] = mean(entropy_proxy_avg_values)
                metrics["probe/mean/group_entropy_proxy_sumlogprob"] = mean(entropy_proxy_sum_values)
                metrics["probe/mean/response_length_mean"] = mean(response_length_values)
                continue

            step_match = STEP_LINE_RE.search(line)
            if not step_match:
                continue

            step = int(step_match.group(1))
            metrics = step_metrics.setdefault(step, {"step": step})
            parts = line[step_match.end() :].split(" - ")
            for part in parts:
                if ":" not in part:
                    continue
                key, value = part.split(":", 1)
                key = key.strip()
                value = value.strip()
                parsed = safe_float(value)
                metrics[key] = parsed if parsed is not None else value

    return config, step_metrics


def discover_experiments(log_dir: Path, rollout_root: Path) -> list[ExperimentSpec]:
    rollout_dirs = discover_rollout_dirs(rollout_root=rollout_root)
    experiments: list[ExperimentSpec] = []
    used_rollout_keys: set[str] = set()

    for log_path in sorted(log_dir.rglob("*.log")):
        config, log_metrics = parse_log_file(log_path)
        rollout_relpath = canonical_rollout_relpath(config.get("ROLLOUT_LOG_DIR"))
        if rollout_relpath is None or rollout_relpath not in rollout_dirs:
            continue
        used_rollout_keys.add(rollout_relpath)
        relative_log = log_path.relative_to(log_dir).with_suffix("").as_posix()
        experiments.append(
            ExperimentSpec(
                experiment_id=relative_log.replace("/", "__"),
                rollout_dir=rollout_dirs[rollout_relpath],
                rollout_relpath=rollout_relpath,
                log_path=log_path,
                config=config,
                log_metrics_by_step=log_metrics,
            )
        )

    for rollout_relpath, rollout_dir in sorted(rollout_dirs.items()):
        if rollout_relpath in used_rollout_keys:
            continue
        experiments.append(
            ExperimentSpec(
                experiment_id=rollout_relpath.replace("/", "__"),
                rollout_dir=rollout_dir,
                rollout_relpath=rollout_relpath,
                log_path=None,
                config={},
                log_metrics_by_step={},
            )
        )

    return experiments


def read_jsonl(file_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with file_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def sort_step_files(paths: list[Path]) -> list[Path]:
    def _key(path: Path) -> tuple[int, str]:
        try:
            return int(path.stem), path.name
        except ValueError:
            return 10**18, path.name

    return sorted(paths, key=_key)


def count_signs(values: list[float | None], eps: float = 1e-12) -> tuple[int, int, int]:
    positive = 0
    negative = 0
    zero = 0
    for value in values:
        if value is None or abs(value) <= eps:
            zero += 1
        elif value > 0:
            positive += 1
        else:
            negative += 1
    return positive, negative, zero


def summarize_group_rows(
    rows: list[dict[str, Any]],
    experiment_id: str,
    rollout_relpath: str,
    file_name: str,
    step: int,
    eps: float = 1e-12,
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get("group_key", "")), []).append(row)

    summaries: list[dict[str, Any]] = []
    for group_key, group_rows in groups.items():
        rep = group_rows[0]
        selected_rows = [row for row in group_rows if safe_bool(row.get("selected_for_update"))]

        pseudo_values = [safe_float(row.get("score")) for row in group_rows]
        oracle_values = [safe_float(row.get("oracle_acc")) for row in group_rows]
        push_values = [safe_float(row.get("traj_entropy_push_proxy")) for row in group_rows]
        lengths = [safe_float(row.get("traj_num_tokens")) for row in group_rows]
        avg_logprobs = [safe_float(row.get("traj_avg_logprob", row.get("traj_logp_avglogprob"))) for row in group_rows]
        selected_lengths = [safe_float(row.get("traj_num_tokens")) for row in selected_rows]
        selected_logprobs = [
            safe_float(row.get("traj_avg_logprob", row.get("traj_logp_avglogprob"))) for row in selected_rows
        ]

        pseudo_pos = sum(1 for value in pseudo_values if value is not None and value > 0.0)
        oracle_pos = sum(1 for value in oracle_values if value is not None and value > 0.0)
        push_pos, push_neg, push_zero = count_signs(push_values, eps=eps)

        selected_pseudo_values = [safe_float(row.get("score")) for row in selected_rows]
        selected_oracle_values = [safe_float(row.get("oracle_acc")) for row in selected_rows]
        selected_push_values = [safe_float(row.get("traj_entropy_push_proxy")) for row in selected_rows]
        selected_pseudo_pos = sum(1 for value in selected_pseudo_values if value is not None and value > 0.0)
        selected_oracle_pos = sum(1 for value in selected_oracle_values if value is not None and value > 0.0)
        selected_push_pos, selected_push_neg, selected_push_zero = count_signs(selected_push_values, eps=eps)

        non_empty_preds = [str(row.get("pred", "")).strip() for row in group_rows if str(row.get("pred", "")).strip()]
        unique_pred_count = len(Counter(non_empty_preds))
        avg_token_probs = [math.exp(value) for value in avg_logprobs if value is not None]
        selected_avg_token_probs = [math.exp(value) for value in selected_logprobs if value is not None]
        group_entropy_proxy = safe_float(rep.get("group_entropy_proxy"))
        group_entropy_grad_proxy_before = safe_float(rep.get("group_entropy_grad_proxy_before_select"))
        group_entropy_grad_proxy_after = safe_float(rep.get("group_entropy_grad_proxy_after_select"))
        group_entropy_grad_proxy_before_avglogprob = safe_float(rep.get("group_entropy_grad_proxy_before_avglogprob"))
        group_entropy_grad_proxy_after_avglogprob = safe_float(rep.get("group_entropy_grad_proxy_after_avglogprob"))
        group_entropy_grad_proxy_before_sumlogprob = safe_float(rep.get("group_entropy_grad_proxy_before_sumlogprob"))
        group_entropy_grad_proxy_after_sumlogprob = safe_float(rep.get("group_entropy_grad_proxy_after_sumlogprob"))
        group_cov_before = safe_float(rep.get("group_cov_before_select"))
        group_cov_after = safe_float(rep.get("group_cov_after_select"))

        summaries.append(
            {
                "experiment_id": experiment_id,
                "rollout_relpath": rollout_relpath,
                "file_name": file_name,
                "step": step,
                "group_key": group_key,
                "question_id": rep.get("question_id"),
                "group_size": len(group_rows),
                "selected_count": len(selected_rows),
                "selected_ratio": len(selected_rows) / max(len(group_rows), 1),
                "pseudo_pos_count": pseudo_pos,
                "pseudo_neg_count": len(group_rows) - pseudo_pos,
                "pseudo_pos_ratio": pseudo_pos / max(len(group_rows), 1),
                "selected_pseudo_pos_count": selected_pseudo_pos,
                "selected_pseudo_neg_count": len(selected_rows) - selected_pseudo_pos,
                "selected_pseudo_pos_ratio": selected_pseudo_pos / max(len(selected_rows), 1),
                "oracle_pos_count": oracle_pos,
                "oracle_neg_count": len(group_rows) - oracle_pos,
                "oracle_pos_ratio": oracle_pos / max(len(group_rows), 1),
                "selected_oracle_pos_count": selected_oracle_pos,
                "selected_oracle_neg_count": len(selected_rows) - selected_oracle_pos,
                "selected_oracle_pos_ratio": selected_oracle_pos / max(len(selected_rows), 1),
                "push_pos_count": push_pos,
                "push_neg_count": push_neg,
                "push_zero_count": push_zero,
                "selected_push_pos_count": selected_push_pos,
                "selected_push_neg_count": selected_push_neg,
                "selected_push_zero_count": selected_push_zero,
                "mean_response_length": mean([value for value in lengths if value is not None]),
                "p50_response_length": quantile([value for value in lengths if value is not None], 0.5),
                "p90_response_length": quantile([value for value in lengths if value is not None], 0.9),
                "selected_mean_response_length": mean([value for value in selected_lengths if value is not None]),
                "mean_traj_avg_logprob": mean([value for value in avg_logprobs if value is not None]),
                "selected_mean_traj_avg_logprob": mean([value for value in selected_logprobs if value is not None]),
                "mean_avg_token_prob": mean(avg_token_probs),
                "selected_mean_avg_token_prob": mean(selected_avg_token_probs),
                "non_empty_pred_count": len(non_empty_preds),
                "non_empty_pred_ratio": len(non_empty_preds) / max(len(group_rows), 1),
                "unique_pred_count": unique_pred_count,
                "vote_answer": rep.get("vote_answer"),
                "vote_count": safe_int(rep.get("vote_count")),
                "vote_ratio": safe_float(rep.get("vote_ratio")),
                "vote_tie": safe_bool(rep.get("vote_tie")),
                "group_entropy_proxy": group_entropy_proxy,
                "group_entropy_grad_proxy_before_select": group_entropy_grad_proxy_before,
                "group_entropy_grad_proxy_after_select": group_entropy_grad_proxy_after,
                "group_entropy_grad_proxy_before_avglogprob": group_entropy_grad_proxy_before_avglogprob,
                "group_entropy_grad_proxy_after_avglogprob": group_entropy_grad_proxy_after_avglogprob,
                "group_entropy_grad_proxy_before_sumlogprob": group_entropy_grad_proxy_before_sumlogprob,
                "group_entropy_grad_proxy_after_sumlogprob": group_entropy_grad_proxy_after_sumlogprob,
                "group_entropy_grad_proxy_delta": (
                    None
                    if group_entropy_grad_proxy_before is None or group_entropy_grad_proxy_after is None
                    else group_entropy_grad_proxy_after - group_entropy_grad_proxy_before
                ),
                "group_entropy_grad_proxy_delta_avglogprob": (
                    None
                    if group_entropy_grad_proxy_before_avglogprob is None or group_entropy_grad_proxy_after_avglogprob is None
                    else group_entropy_grad_proxy_after_avglogprob - group_entropy_grad_proxy_before_avglogprob
                ),
                "group_entropy_grad_proxy_delta_sumlogprob": (
                    None
                    if group_entropy_grad_proxy_before_sumlogprob is None or group_entropy_grad_proxy_after_sumlogprob is None
                    else group_entropy_grad_proxy_after_sumlogprob - group_entropy_grad_proxy_before_sumlogprob
                ),
                "group_cov_before_select": group_cov_before,
                "group_cov_after_select": group_cov_after,
                "group_cov_delta": (
                    None if group_cov_before is None or group_cov_after is None else group_cov_after - group_cov_before
                ),
                "label_acc": safe_float(rep.get("label_acc")),
                "true_answer_ratio": safe_float(rep.get("true_answer_ratio")),
                "group_reward_acc": safe_float(rep.get("group_reward_acc")),
                "group_reward_mean": safe_float(rep.get("group_reward_mean")),
            }
        )

    return summaries


def summarize_step_rows(
    rows: list[dict[str, Any]],
    question_rows: list[dict[str, Any]],
    experiment_id: str,
    rollout_relpath: str,
    file_name: str,
    step: int,
    eps: float = 1e-12,
) -> dict[str, Any]:
    selected_rows = [row for row in rows if safe_bool(row.get("selected_for_update"))]

    pseudo_values = [safe_float(row.get("score")) for row in rows]
    oracle_values = [safe_float(row.get("oracle_acc")) for row in rows]
    push_values = [safe_float(row.get("traj_entropy_push_proxy")) for row in rows]
    lengths = [safe_float(row.get("traj_num_tokens")) for row in rows]
    avg_logprobs = [safe_float(row.get("traj_avg_logprob", row.get("traj_logp_avglogprob"))) for row in rows]

    selected_pseudo_values = [safe_float(row.get("score")) for row in selected_rows]
    selected_oracle_values = [safe_float(row.get("oracle_acc")) for row in selected_rows]
    selected_push_values = [safe_float(row.get("traj_entropy_push_proxy")) for row in selected_rows]
    selected_lengths = [safe_float(row.get("traj_num_tokens")) for row in selected_rows]
    selected_logprobs = [
        safe_float(row.get("traj_avg_logprob", row.get("traj_logp_avglogprob"))) for row in selected_rows
    ]

    pseudo_pos = sum(1 for value in pseudo_values if value is not None and value > 0.0)
    oracle_pos = sum(1 for value in oracle_values if value is not None and value > 0.0)
    push_pos, push_neg, push_zero = count_signs(push_values, eps=eps)
    selected_pseudo_pos = sum(1 for value in selected_pseudo_values if value is not None and value > 0.0)
    selected_oracle_pos = sum(1 for value in selected_oracle_values if value is not None and value > 0.0)
    selected_push_pos, selected_push_neg, selected_push_zero = count_signs(selected_push_values, eps=eps)

    avg_token_probs = [math.exp(value) for value in avg_logprobs if value is not None]
    selected_avg_token_probs = [math.exp(value) for value in selected_logprobs if value is not None]
    question_entropy_values = [
        value for value in [safe_float(row.get("group_entropy_proxy")) for row in question_rows] if value is not None
    ]
    question_entropy_grad_before_values = [
        value
        for value in [safe_float(row.get("group_entropy_grad_proxy_before_select")) for row in question_rows]
        if value is not None
    ]
    question_entropy_grad_after_values = [
        value
        for value in [safe_float(row.get("group_entropy_grad_proxy_after_select")) for row in question_rows]
        if value is not None
    ]
    question_entropy_grad_delta_values = [
        value for value in [safe_float(row.get("group_entropy_grad_proxy_delta")) for row in question_rows] if value is not None
    ]
    question_entropy_grad_before_avglogprob_values = [
        value
        for value in [safe_float(row.get("group_entropy_grad_proxy_before_avglogprob")) for row in question_rows]
        if value is not None
    ]
    question_entropy_grad_after_avglogprob_values = [
        value
        for value in [safe_float(row.get("group_entropy_grad_proxy_after_avglogprob")) for row in question_rows]
        if value is not None
    ]
    question_entropy_grad_delta_avglogprob_values = [
        value
        for value in [safe_float(row.get("group_entropy_grad_proxy_delta_avglogprob")) for row in question_rows]
        if value is not None
    ]
    question_entropy_grad_before_sumlogprob_values = [
        value
        for value in [safe_float(row.get("group_entropy_grad_proxy_before_sumlogprob")) for row in question_rows]
        if value is not None
    ]
    question_entropy_grad_after_sumlogprob_values = [
        value
        for value in [safe_float(row.get("group_entropy_grad_proxy_after_sumlogprob")) for row in question_rows]
        if value is not None
    ]
    question_entropy_grad_delta_sumlogprob_values = [
        value
        for value in [safe_float(row.get("group_entropy_grad_proxy_delta_sumlogprob")) for row in question_rows]
        if value is not None
    ]
    question_cov_before_values = [
        value for value in [safe_float(row.get("group_cov_before_select")) for row in question_rows] if value is not None
    ]
    question_cov_after_values = [
        value for value in [safe_float(row.get("group_cov_after_select")) for row in question_rows] if value is not None
    ]
    question_cov_delta_values = [
        value for value in [safe_float(row.get("group_cov_delta")) for row in question_rows] if value is not None
    ]
    total_rows = len(rows)
    selected_total_rows = len(selected_rows)
    question_vote_tie_values = [1.0 if safe_bool(row.get("vote_tie")) else 0.0 for row in question_rows]

    return {
        "experiment_id": experiment_id,
        "rollout_relpath": rollout_relpath,
        "file_name": file_name,
        "step": step,
        "group_count": len(question_rows),
        "total_rows": total_rows,
        "selected_rows": selected_total_rows,
        "pseudo_pos_count": pseudo_pos,
        "pseudo_neg_count": total_rows - pseudo_pos,
        "selected_pseudo_pos_count": selected_pseudo_pos,
        "selected_pseudo_neg_count": selected_total_rows - selected_pseudo_pos,
        "oracle_pos_count": oracle_pos,
        "oracle_neg_count": total_rows - oracle_pos,
        "selected_oracle_pos_count": selected_oracle_pos,
        "selected_oracle_neg_count": selected_total_rows - selected_oracle_pos,
        "push_pos_count": push_pos,
        "push_neg_count": push_neg,
        "push_zero_count": push_zero,
        "selected_push_pos_count": selected_push_pos,
        "selected_push_neg_count": selected_push_neg,
        "selected_push_zero_count": selected_push_zero,
        "pseudo_pos_ratio": pseudo_pos / max(total_rows, 1),
        "oracle_pos_ratio": oracle_pos / max(total_rows, 1),
        "push_pos_ratio": push_pos / max(total_rows, 1),
        "push_neg_ratio": push_neg / max(total_rows, 1),
        "selected_pseudo_pos_ratio": selected_pseudo_pos / max(selected_total_rows, 1),
        "selected_oracle_pos_ratio": selected_oracle_pos / max(selected_total_rows, 1),
        "selected_push_pos_ratio": selected_push_pos / max(selected_total_rows, 1),
        "selected_push_neg_ratio": selected_push_neg / max(selected_total_rows, 1),
        "mean_response_length": mean([value for value in lengths if value is not None]),
        "p50_response_length": quantile([value for value in lengths if value is not None], 0.5),
        "p90_response_length": quantile([value for value in lengths if value is not None], 0.9),
        "selected_mean_response_length": mean([value for value in selected_lengths if value is not None]),
        "mean_traj_avg_logprob": mean([value for value in avg_logprobs if value is not None]),
        "selected_mean_traj_avg_logprob": mean([value for value in selected_logprobs if value is not None]),
        "mean_avg_token_prob": mean(avg_token_probs),
        "selected_mean_avg_token_prob": mean(selected_avg_token_probs),
        "traj_oracle_acc_mean": mean([value for value in oracle_values if value is not None]),
        "selected_traj_oracle_acc_mean": mean([value for value in selected_oracle_values if value is not None]),
        "question_entropy_proxy_mean": mean(question_entropy_values),
        "question_entropy_proxy_std": std(question_entropy_values),
        "question_entropy_proxy_min": min(question_entropy_values) if question_entropy_values else None,
        "question_entropy_proxy_max": max(question_entropy_values) if question_entropy_values else None,
        "question_entropy_proxy_range": (
            None
            if not question_entropy_values
            else max(question_entropy_values) - min(question_entropy_values)
        ),
        "question_entropy_grad_proxy_before_mean": mean(question_entropy_grad_before_values),
        "question_entropy_grad_proxy_after_mean": mean(question_entropy_grad_after_values),
        "question_entropy_grad_proxy_delta_mean": mean(question_entropy_grad_delta_values),
        "question_entropy_grad_proxy_before_avglogprob_mean": mean(question_entropy_grad_before_avglogprob_values),
        "question_entropy_grad_proxy_after_avglogprob_mean": mean(question_entropy_grad_after_avglogprob_values),
        "question_entropy_grad_proxy_delta_avglogprob_mean": mean(question_entropy_grad_delta_avglogprob_values),
        "question_entropy_grad_proxy_before_sumlogprob_mean": mean(question_entropy_grad_before_sumlogprob_values),
        "question_entropy_grad_proxy_after_sumlogprob_mean": mean(question_entropy_grad_after_sumlogprob_values),
        "question_entropy_grad_proxy_delta_sumlogprob_mean": mean(question_entropy_grad_delta_sumlogprob_values),
        "question_cov_before_mean": mean(question_cov_before_values),
        "question_cov_after_mean": mean(question_cov_after_values),
        "question_cov_delta_mean": mean(question_cov_delta_values),
        "question_vote_ratio_mean": mean(
            [value for value in [safe_float(row.get("vote_ratio")) for row in question_rows] if value is not None]
        ),
        "question_vote_tie_ratio": mean(question_vote_tie_values),
        "question_label_acc_mean": mean(
            [value for value in [safe_float(row.get("label_acc")) for row in question_rows] if value is not None]
        ),
        "question_true_answer_ratio_mean": mean(
            [value for value in [safe_float(row.get("true_answer_ratio")) for row in question_rows] if value is not None]
        ),
        "question_reward_acc_mean": mean(
            [value for value in [safe_float(row.get("group_reward_acc")) for row in question_rows] if value is not None]
        ),
        "question_reward_mean": mean(
            [value for value in [safe_float(row.get("group_reward_mean")) for row in question_rows] if value is not None]
        ),
        "question_non_empty_pred_ratio_mean": mean(
            [value for value in [safe_float(row.get("non_empty_pred_ratio")) for row in question_rows] if value is not None]
        ),
        "question_unique_pred_count_mean": mean(
            [float(row["unique_pred_count"]) for row in question_rows if row.get("unique_pred_count") is not None]
        ),
        "batch_group_entropy_proxy_mean": mean(question_entropy_values),
        "batch_group_entropy_grad_proxy_before_mean": mean(question_entropy_grad_before_values),
        "batch_group_entropy_grad_proxy_after_mean": mean(question_entropy_grad_after_values),
        "batch_group_entropy_grad_proxy_before_avglogprob_mean": mean(question_entropy_grad_before_avglogprob_values),
        "batch_group_entropy_grad_proxy_after_avglogprob_mean": mean(question_entropy_grad_after_avglogprob_values),
        "batch_group_entropy_grad_proxy_delta_avglogprob_mean": mean(question_entropy_grad_delta_avglogprob_values),
        "batch_group_entropy_grad_proxy_before_sumlogprob_mean": mean(question_entropy_grad_before_sumlogprob_values),
        "batch_group_entropy_grad_proxy_after_sumlogprob_mean": mean(question_entropy_grad_after_sumlogprob_values),
        "batch_group_entropy_grad_proxy_delta_sumlogprob_mean": mean(question_entropy_grad_delta_sumlogprob_values),
        "batch_vote_ratio_mean": mean(
            [value for value in [safe_float(row.get("vote_ratio")) for row in question_rows] if value is not None]
        ),
        "batch_vote_tie_ratio": mean(question_vote_tie_values),
        "batch_unique_pred_count_mean": mean(
            [float(row["unique_pred_count"]) for row in question_rows if row.get("unique_pred_count") is not None]
        ),
        "batch_label_acc_mean": mean(
            [value for value in [safe_float(row.get("label_acc")) for row in question_rows] if value is not None]
        ),
        "batch_true_answer_ratio_mean": mean(
            [value for value in [safe_float(row.get("true_answer_ratio")) for row in question_rows] if value is not None]
        ),
        "batch_oracle_pos_ratio": oracle_pos / max(total_rows, 1),
        "batch_pseudo_pos_ratio": pseudo_pos / max(total_rows, 1),
        "batch_response_length_mean": mean([value for value in lengths if value is not None]),
        "batch_selected_response_length_mean": mean([value for value in selected_lengths if value is not None]),
        "batch_selected_count_total": selected_total_rows,
        "batch_selected_count_mean_per_question": selected_total_rows / max(len(question_rows), 1),
    }


def load_dataset_view_map(dataset_view_path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(dataset_view_path.read_text(encoding="utf-8"))
    result: dict[str, dict[str, Any]] = {}
    for item in data:
        extra = item.get("extra_info", {}) or {}
        question_id = extra.get("question_id") or extra.get("index")
        if question_id is None:
            continue
        result[str(question_id)] = item
    return result
