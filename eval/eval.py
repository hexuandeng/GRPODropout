import json
import argparse
import os
import importlib.util
import random
import time
import pickle
import sys
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm
import numpy as np

# VLLM & Utils Imports
from vllm import LLM, SamplingParams
import vllm.envs as envs
from transformers import AutoTokenizer
from utils.utils import set_seed, load_jsonl, save_jsonl

# These utility functions are expected to be available under your utils directory.
from utils.utils import set_seed, load_jsonl, save_jsonl
from utils.parser import CHOICE_DATASETS, SHORT_NUMERIC_DATASETS, parse_question, parse_ground_truth, extract_answer, normalize_short_numeric_answer
from utils.data_loader import load_data
from utils.grader import check_is_correct

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ================= Configuration =================

# Dataset configuration mapping
# key: data_name (used to load prompt and parser)
# value: filename (actual uploaded file name, used by data_loader)
DATASET_CONFIG = {
    "aime24": "AIME24.parquet",
    "aime2025": "aime2025.jsonl",
    "amc23": "AMC23.parquet",
    "gsm8k": "GSM8K.parquet",
    "hmmt25": "HMMT25.jsonl",
    "math_500": "MATH-500.jsonl",
    "olympiadbench": "olympiadbench.jsonl",
    "arc_challenge": "ARC-Challenge.jsonl",
    "mmlu_pro": "MMLU-Pro.jsonl",
    "gpqa_diamond": "gpqa_diamond.jsonl",
    "supergpqa": "SuperGPQA.jsonl",
}

VERL_DATASET_CONFIG = {
    "aime24": os.path.join("AIME24", "AIME24_verl_val.parquet"),
    "amc23": os.path.join("AMC23", "AMC23_verl_val.parquet"),
    "olympiadbench": os.path.join("Olympiad", "Olympiad_verl_val.parquet"),
    "dapo17k": os.path.join("dapo17k", "dapo17k_verl_val.parquet"),
}

MARIO_MATH_DATASETS = {
    "aime24",
    "aime2025",
    "amc23",
    "gsm8k",
    "hmmt25",
    "math_500",
    "olympiadbench",
    "dapo17k",
}

# ================= Core logic =================

def parse_list(arg):
    return arg.split(',')

def save_completions(completions, filepath):
    with open(filepath, 'wb') as file:
        pickle.dump(completions, file)

def load_module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def load_verl_reward_fn(reward_path):
    if not os.path.isabs(reward_path):
        reward_path = os.path.abspath(os.path.join(BASE_DIR, reward_path))
    if not os.path.exists(reward_path):
        raise FileNotFoundError(f"Verl reward file not found: {reward_path}")

    repo_root = os.path.abspath(os.path.join(BASE_DIR, ".."))
    verl_root = os.path.join(repo_root, "verl")
    for path in [repo_root, verl_root]:
        if path not in sys.path:
            sys.path.insert(0, path)

    reward_score_dir = os.path.join(verl_root, "verl", "utils", "reward_score")
    if os.path.isdir(reward_score_dir):
        package_names = [
            "verl",
            "verl.utils",
            "verl.utils.reward_score",
        ]
        for package_name in package_names:
            package = sys.modules.get(package_name)
            if package is None:
                package = importlib.util.module_from_spec(
                    importlib.machinery.ModuleSpec(package_name, loader=None, is_package=True)
                )
                package.__path__ = []
                sys.modules[package_name] = package

        load_module_from_file("verl.utils.reward_score.math", os.path.join(reward_score_dir, "math.py"))
        load_module_from_file("verl.utils.reward_score.math_dapo", os.path.join(reward_score_dir, "math_dapo.py"))

    module = load_module_from_file("verl_grpo_dropout_reward", reward_path)
    return module.compute_score_gt_math

def load_mario_is_equiv(mario_eval_path=None, required=True):
    if mario_eval_path:
        mario_eval_path = os.path.abspath(os.path.expanduser(mario_eval_path))
        if not os.path.exists(mario_eval_path):
            raise FileNotFoundError(f"MARIO_EVAL path not found: {mario_eval_path}")
        if mario_eval_path not in sys.path:
            sys.path.insert(0, mario_eval_path)

    try:
        from math_evaluation import is_equiv
    except Exception as e:
        message = (
            "Failed to import MARIO_EVAL. Install it or pass --mario_eval_path "
            "pointing to the MARIO_EVAL repo root."
        )
        if required:
            raise ImportError(message) from e
        print(f"Warning: {message} MARIO fallback will be disabled.")
        return None

    return is_equiv

def grade_with_optional_mario(pred, gt, data_name, mario_is_equiv=None, strict_enabled=True):
    info = {
        "pred": "" if pred is None else str(pred),
        "strict_acc": 0.0,
        "mario_acc": 0.0,
        "acc": 0.0,
        "score": 0.0,
        "grade_source": "strict",
    }

    if strict_enabled and data_name in SHORT_NUMERIC_DATASETS:
        pred_num = normalize_short_numeric_answer(pred)
        gt_values = gt if isinstance(gt, list) else [gt]
        strict_correct = pred_num != "" and any(
            pred_num == normalize_short_numeric_answer(candidate)
            for candidate in gt_values
        )
        info["strict_acc"] = float(strict_correct)
    elif strict_enabled:
        try:
            strict_correct = bool(check_is_correct(pred, gt, data_name=data_name))
            info["strict_acc"] = float(strict_correct)
        except Exception as e:
            strict_correct = False
            info["strict_error"] = f"{type(e).__name__}: {e}"
    else:
        strict_correct = False

    if strict_correct:
        info["acc"] = 1.0
        info["score"] = 1.0
        return info

    if mario_is_equiv is not None and data_name in MARIO_MATH_DATASETS:
        gt_candidates = gt if isinstance(gt, list) else [gt]
        for candidate in gt_candidates:
            try:
                if mario_is_equiv(str(candidate), str(pred), fast=False):
                    info["mario_acc"] = 1.0
                    info["acc"] = 1.0
                    info["score"] = 1.0
                    info["grade_source"] = "mario_fallback"
                    break
            except Exception as e:
                info["mario_error"] = f"{type(e).__name__}: {e}"

    return info

def is_mario_math_effective(args):
    return args.grade_mode != "verl" and not args.disable_mario_math and (
        args.grade_mode == "paper" or args.use_mario_math
    )

def grade_response_for_paper(response, gt, data_name, data_source=None, extra_info=None, verl_reward_fn=None, mario_is_equiv=None):
    pred = extract_answer(response, data_name)
    info = {
        "pred": "" if pred is None else str(pred),
        "acc": 0.0,
        "score": 0.0,
        "grade_source": "incorrect",
        "choice_acc": 0.0,
        "verl_acc": 0.0,
        "strict_acc": 0.0,
        "mario_acc": 0.0,
    }

    if data_name in CHOICE_DATASETS:
        choice_correct = bool(check_is_correct(pred, gt, data_name=data_name))
        info["choice_acc"] = float(choice_correct)
        if choice_correct:
            info["acc"] = 1.0
            info["score"] = 1.0
            info["grade_source"] = "choice"
        return info

    verl_info = None
    if verl_reward_fn is not None:
        try:
            verl_info = verl_reward_fn(data_source or data_name, response, gt, extra_info)
            info["verl_info"] = verl_info
            verl_pred = str(verl_info.get("pred", ""))
            verl_correct = bool(verl_info.get("acc", verl_info.get("score", 0.0)))
            info["verl_acc"] = float(verl_correct)
            if verl_correct:
                if verl_pred:
                    info["pred"] = verl_pred
                info["acc"] = 1.0
                info["score"] = 1.0
                info["grade_source"] = "verl_reward"
                return info
        except Exception as e:
            info["verl_error"] = f"{type(e).__name__}: {e}"

    fallback_info = grade_with_optional_mario(pred, gt, data_name, mario_is_equiv=mario_is_equiv)
    info["strict_acc"] = fallback_info.get("strict_acc", 0.0)
    info["mario_acc"] = fallback_info.get("mario_acc", 0.0)
    info["pred"] = fallback_info.get("pred", info["pred"])
    for key in ("strict_error", "mario_error"):
        if key in fallback_info:
            info[key] = fallback_info[key]

    if fallback_info.get("acc", 0.0):
        info["acc"] = 1.0
        info["score"] = 1.0
        info["grade_source"] = fallback_info.get("grade_source", "strict")

    return info

def get_verl_ground_truth(example):
    reward_model = example.get("reward_model")
    if isinstance(reward_model, dict) and "ground_truth" in reward_model:
        return reward_model.get("ground_truth")
    _, gt_ans = parse_ground_truth(example)
    return gt_ans

def get_verl_extra_info(example):
    extra_info = example.get("extra_info")
    return extra_info if isinstance(extra_info, dict) else None

def get_data_source(example, data_name):
    return example.get("data_source") or example.get("source") or data_name

def get_prompt_user_content(messages):
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            content = str(message.get("content", "")).strip()
            if content:
                return content
    return ""

def get_extra_question_text(example):
    extra_info = example.get("extra_info")
    if not isinstance(extra_info, dict):
        return ""
    for key in ("raw_problem", "problem", "question", "raw_prompt", "input", "content"):
        value = extra_info.get(key)
        if value:
            return str(value).strip()
    return ""

def get_question_text(example, data_name):
    try:
        question = parse_question(example, data_name)
        if question:
            return question
    except Exception:
        pass

    question = get_prompt_user_content(example.get("prompt"))
    if question:
        return question

    question = get_extra_question_text(example)
    if question:
        return question

    if "raw_prompt" in example:
        return str(example["raw_prompt"])

    try:
        return parse_question(example, data_name)
    except Exception:
        raise

def read_parquet_records(path):
    try:
        import pandas as pd
        return pd.read_parquet(path).to_dict("records")
    except ImportError:
        import pyarrow.parquet as pq
        return pq.read_table(path).to_pylist()

def empirical_pass_at_k(correctness, k):
    """Return 1 if any of the first k rollouts is correct, otherwise 0."""
    if k < 1:
        raise ValueError("k must be >= 1")
    if len(correctness) < k:
        raise ValueError(f"pass@k requires at least k rollouts, but got n={len(correctness)}, k={k}")
    return 1.0 if any(correctness[:k]) else 0.0

def get_correctness_list(item):
    correctness = item.get("answers_correctness")
    if correctness is not None:
        return [bool(x) for x in correctness]
    if "is_correct" in item:
        return [bool(item["is_correct"])]
    return []

def bootstrap_metric(data, subset_size, reduce_fns, n_bootstrap=1000, seed=42):
    np.random.seed(seed)
    results = [[] for _ in reduce_fns]
    for _ in range(n_bootstrap):
        idxs = np.random.choice(len(data), size=subset_size, replace=True)
        sample = [data[i] for i in idxs]
        for i, reduce_fn in enumerate(reduce_fns):
            results[i].append(reduce_fn(sample))
    return [(float(np.mean(values)), float(np.std(values))) for values in results]

def calc_maj_val(data, vote_key="pred", val_key="val"):
    vote2vals = defaultdict(list)
    for item in data:
        vote2vals[item[vote_key]].append(item[val_key])
    if not vote2vals:
        return 0.0
    vote2cnt = {key: len(values) for key, values in vote2vals.items()}
    maj_vote = max(vote2cnt, key=vote2cnt.get)
    return float(vote2vals[maj_vote][0])

def deterministic_maj_val(preds, vals):
    vote_data = [{"pred": pred, "val": val} for pred, val in zip(preds, vals)]
    return calc_maj_val(vote_data)

def compute_metrics(file_outputs, k):
    total = len(file_outputs)
    if total == 0:
        return {
            "mean_score": 0.0,
            "pass_at_k": 0.0,
            "best_at_k_mean": 0.0,
            "best_at_k_std": 0.0,
            "maj_at_k_det": 0.0,
            "maj_at_k_mean": 0.0,
            "maj_at_k_std": 0.0,
            "correct": 0,
            "total": 0,
            "total_correct_completions": 0,
            "total_completions": 0,
        }

    correctness_lists = [get_correctness_list(item) for item in file_outputs]
    pred_lists = [item.get("generated_answers") or item.get("preds") or [] for item in file_outputs]
    correct = sum(1 for correctness in correctness_lists if any(correctness))
    total_correct_completions = sum(sum(correctness) for correctness in correctness_lists)
    total_completions = sum(len(correctness) for correctness in correctness_lists)

    pass_at_k_scores = []
    maj_det_scores = []
    best_boot_scores = []
    best_boot_stds = []
    maj_boot_scores = []
    maj_boot_stds = []
    for correctness in correctness_lists:
        if len(correctness) >= k:
            pass_at_k_scores.append(empirical_pass_at_k(correctness, k))
            [(best_mean, best_std)] = bootstrap_metric(
                data=[float(x) for x in correctness],
                subset_size=k,
                reduce_fns=[np.max],
            )
            best_boot_scores.append(best_mean)
            best_boot_stds.append(best_std)

    for correctness, preds in zip(correctness_lists, pred_lists):
        if len(correctness) >= k and len(preds) >= k:
            vals = [float(x) for x in correctness[:k]]
            cur_preds = [str(x) for x in preds[:k]]
            maj_det_scores.append(deterministic_maj_val(cur_preds, vals))
            vote_data = [{"val": val, "pred": pred} for val, pred in zip(vals, cur_preds)]
            [(maj_mean, maj_std)] = bootstrap_metric(
                data=vote_data,
                subset_size=k,
                reduce_fns=[calc_maj_val],
            )
            maj_boot_scores.append(maj_mean)
            maj_boot_stds.append(maj_std)

    return {
        "mean_score": total_correct_completions / total_completions if total_completions > 0 else 0.0,
        "pass_at_k": sum(pass_at_k_scores) / len(pass_at_k_scores) if pass_at_k_scores else 0.0,
        "best_at_k_mean": sum(best_boot_scores) / len(best_boot_scores) if best_boot_scores else 0.0,
        "best_at_k_std": sum(best_boot_stds) / len(best_boot_stds) if best_boot_stds else 0.0,
        "maj_at_k_det": sum(maj_det_scores) / len(maj_det_scores) if maj_det_scores else 0.0,
        "maj_at_k_mean": sum(maj_boot_scores) / len(maj_boot_scores) if maj_boot_scores else 0.0,
        "maj_at_k_std": sum(maj_boot_stds) / len(maj_boot_stds) if maj_boot_stds else 0.0,
        "correct": correct,
        "total": total,
        "total_correct_completions": total_correct_completions,
        "total_completions": total_completions,
    }

def build_mode_suffix(args):
    tags = []
    if args.grade_mode != "local":
        tags.append(args.grade_mode)
    if args.use_verl_prompt:
        tags.append("verlprompt")
    if args.use_verl_reward or args.grade_mode == "verl":
        tags.append("verlreward")
    if is_mario_math_effective(args):
        tags.append("mario")
    if args.enable_thinking:
        tags.append("think")
    if args.repeat_requests:
        tags.append("repeatreq")
    if args.top_k != -1:
        tags.append(f"topk{args.top_k}")
    return "_" + "_".join(tags) if tags else ""

def build_summary(args, model_short_name, data_name, processing_end_idx, metrics, resumed=False):
    return {
        "model": model_short_name,
        "dataset": data_name,
        "split": args.split,
        "prompt_type": args.prompt_type,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "n_sampling": args.n_sampling,
        "k": args.k,
        "grade_mode": args.grade_mode,
        "seed": args.seed,
        "start_idx": args.start_idx,
        "end_idx": processing_end_idx,
        "use_verl_prompt": args.use_verl_prompt,
        "use_verl_reward": args.use_verl_reward,
        "use_mario_math": args.use_mario_math,
        "disable_mario_math": args.disable_mario_math,
        "mario_math_effective": is_mario_math_effective(args),
        "mario_eval_path": args.mario_eval_path,
        "enable_thinking": args.enable_thinking,
        "repeat_requests": args.repeat_requests,
        "verl_reward_path": args.verl_reward_path,
        "mean_score": metrics["mean_score"],
        "pass_at_k": metrics["pass_at_k"],
        f"best@{args.k}/mean": metrics["best_at_k_mean"],
        f"best@{args.k}/std": metrics["best_at_k_std"],
        f"maj@{args.k}/mean": metrics["maj_at_k_mean"],
        f"maj@{args.k}/std": metrics["maj_at_k_std"],
        f"maj@{args.k}_det": metrics["maj_at_k_det"],
        "correct": metrics["correct"],
        "total": metrics["total"],
        "total_correct_completions": metrics["total_correct_completions"],
        "total_completions": metrics["total_completions"],
        "actor_entropy_token_mean": None,
        "total_response_tokens": None,
        "resumed": resumed,
    }

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default="./", help="model dir")
    parser.add_argument('--n_sampling', type=int, default=1, help="n for sampling")
    parser.add_argument("--k", type=int, default=None, help="Value of k for pass@k calculation. Defaults to n_sampling.")
    parser.add_argument("--data_dir", default="./data", type=str, help="Directory containing dataset files")
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=-1)
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--max_tokens", default=8192, type=int)
    parser.add_argument("--prompt_type", default="qwen-base", type=str)
    parser.add_argument("--use_few_shot", action="store_true", help="[Disabled] Keep for compatibility but forced to False in code")
    parser.add_argument("--surround_with_messages", action="store_true")
    parser.add_argument("--output_dir", default="./outputs", type=str)
    parser.add_argument('--stop', type=parse_list)
    parser.add_argument("--top_p", default=0.95, type=float)
    parser.add_argument("--top_k", default=-1, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--completions_save_dir", default='./completions', type=str)
    parser.add_argument(
        "--grade_mode",
        default="local",
        choices=["local", "verl", "paper"],
        help=(
            "local: original eval grader; verl: validation reward only; "
            "paper: unified paper-grade line with choice/verL/strict/MARIO fallback"
        ),
    )
    parser.add_argument("--use_verl_prompt", action="store_true", help="Use verl parquet prompt messages directly when available")
    parser.add_argument("--use_verl_reward", action="store_true", help="Use verl validation reward function for grading")
    parser.add_argument("--use_mario_math", action="store_true", help="Use MARIO_EVAL is_equiv as a fallback after strict math grading fails")
    parser.add_argument("--disable_mario_math", action="store_true", help="Disable MARIO fallback in paper grade mode")
    parser.add_argument("--mario_eval_path", default=None, type=str, help="Optional path to the MARIO_EVAL repo root")
    parser.add_argument("--enable_thinking", action="store_true", help="Pass enable_thinking=True to tokenizer.apply_chat_template")
    parser.add_argument("--repeat_requests", action="store_true", help="Generate n rollouts by repeating each prompt n times with SamplingParams.n=1, matching verl validation style")
    parser.add_argument("--verl_reward_path", default="../verl/examples/grpo_dropout/reward.py", type=str)

    # Register the GPU memory control parameter
    parser.add_argument("--gpu_memory_utilization", default=0.90, type=float, help="vLLM GPU memory utilization")

    # Add a parameter to specify which datasets to run; defaults to all
    parser.add_argument("--datasets", type=parse_list, default=list(DATASET_CONFIG.keys()),
                        help="List of datasets to evaluate. Defaults to all datasets.")

    # During evaluation, compute training-consistent actor/entropy (token-mean). On by default;
    # use --no-compute_entropy to disable and keep the old fast pure-vLLM path.
    parser.add_argument("--compute_entropy", dest="compute_entropy", action="store_true", default=True,
                        help="After generation, load HF model and compute training-consistent actor/entropy (default on)")
    parser.add_argument("--no-compute_entropy", dest="compute_entropy", action="store_false",
                        help="Disable actor/entropy computation")
    parser.add_argument("--force_entropy", action="store_true",
                        help="Recompute actor/entropy even if the summary already has actor_entropy_token_mean")
    parser.add_argument("--entropy_gpu_memory_utilization", default=None, type=float,
                        help="Optional GPU mem util for the HF entropy model (defaults to --gpu_memory_utilization)")

    args = parser.parse_args()

    # Force to False before returning, regardless of whether the bash script passes this flag, to guarantee zero-shot
    args.use_few_shot = False
    if args.grade_mode == "verl":
        args.use_verl_reward = True
    args.top_p = 1 if args.temperature == 0 else args.top_p 
    if args.n_sampling < 1:
        raise ValueError("--n_sampling must be >= 1")
    if args.k is not None and args.k != args.n_sampling:
        print(
            f"Warning: empirical pass@k uses k=n_sampling; "
            f"overriding --k {args.k} to {args.n_sampling}."
        )
    args.k = args.n_sampling
    if args.k < 1:
        raise ValueError("--k must be >= 1")
    return args

def get_conversation_prompt_by_messages(tokenizer, messages, enable_thinking=False):
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if enable_thinking:
        kwargs["enable_thinking"] = True

    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)

def get_three_prompt(prompt_type, data_name):
    qa_datasets = {
        "arc_challenge",
        "mmlu_pro",
        "supergpqa",
        "gpqa_diamond",
    }

    search_name = "qa" if data_name in qa_datasets else "math"
    
    file_path = os.path.join(BASE_DIR, "prompts", prompt_type, f"{search_name}.py")
    
    # ... the loading logic below is unchanged ...

    if not os.path.exists(file_path):
        # Fallback to default if specific prompt not found
        print(f"Warning: Prompt file for {search_name} not found at {file_path}. Checking default...")
        raise FileNotFoundError(f"File not found: {file_path}")
        # Adjust here to match your actual file layout, or just raise an exception
        raise FileNotFoundError(f"File not found: {file_path}")

    spec = importlib.util.spec_from_file_location("dynamic_module", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    return module.system_prompt, module.few_shot_prompt, module.question_format

def load_examples(args, data_name, file_name):
    try:
        return load_data(data_name, args.split, args.data_dir)
    except Exception as e:
        print(f"Standard load failed, trying direct file path load for {file_name}...")
        candidate_dirs = [
            args.data_dir,
            os.path.join(BASE_DIR, "data"),
            os.path.join(os.path.dirname(BASE_DIR), "data"),
        ]
        for data_dir in candidate_dirs:
            full_path = os.path.join(data_dir, file_name)
            if not os.path.exists(full_path):
                continue
            if file_name.endswith('.parquet'):
                return read_parquet_records(full_path)
            elif file_name.endswith('.jsonl'):
                return list(load_jsonl(full_path))
        raise e

def build_prompt_for_example(args, tokenizer, example, data_name, system_prompt, few_shot_prompt, question_format):
    messages = example.get("prompt")
    if args.use_verl_prompt and get_prompt_user_content(messages):
        return get_conversation_prompt_by_messages(
            tokenizer=tokenizer,
            messages=messages,
            enable_thinking=args.enable_thinking,
        )

    question = get_question_text(example, data_name)
    is_choice_dataset = data_name in CHOICE_DATASETS or data_name == "gaokao"

    if args.use_verl_prompt:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
        return get_conversation_prompt_by_messages(
            tokenizer=tokenizer,
            messages=messages,
            enable_thinking=args.enable_thinking,
        )

    if is_choice_dataset:
        user_content = question_format.format(question=question)
    else:
        user_content = question

    if args.use_few_shot:
        user_content = few_shot_prompt + user_content

    if args.surround_with_messages:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        return get_conversation_prompt_by_messages(
            tokenizer=tokenizer,
            messages=messages,
            enable_thinking=args.enable_thinking,
        )

    return user_content

def infer_single_dataset(args, llm, tokenizer, data_name, file_name, entropy_jobs=None):
    """
    Run inference-based evaluation for a single dataset
    """
    print(f"\n{'='*20} Evaluating: {data_name} ({file_name}) {'='*20}")

    # Temporarily override args.data_name so parser and loader use it
    original_data_name = getattr(args, 'data_name', '')
    args.data_name = data_name

    # Compute the sampling factor
    n_sampling = args.n_sampling
    if args.repeat_requests:
        factor = 1
        generation_epoch = n_sampling
    else:
        factor = 1
        for i in range(2, 65):
            if n_sampling % i == 0:
                factor = i
        generation_epoch = n_sampling // factor
    
    sampling_params = SamplingParams(
        temperature=args.temperature, 
        max_tokens=args.max_tokens, 
        n=factor,
        top_p=args.top_p,
        top_k=args.top_k,
        stop=args.stop
    )
    
    # Load data
    # Note: this assumes load_data can handle a specific file path, or that you join file_name to data_dir.
    # If load_data only accepts data_name, you may need to modify utils/data_loader.py to handle file names.
    try:
        # Try loading directly via load_data, assuming its logic already handles data_name
        examples = load_examples(args, data_name, file_name)
    except Exception as e:
        print(f"Standard load failed, trying direct file path load for {file_name}...")
        full_path = os.path.join(args.data_dir, file_name)
        # Fallback: for a simple jsonl/parquet, load directly
        full_path = os.path.join(args.data_dir, file_name)
        if file_name.endswith('.parquet'):
            examples = read_parquet_records(full_path)
        elif file_name.endswith('.jsonl'):
            examples = list(load_jsonl(full_path))
        else:
            raise e

    if args.end_idx == -1:
        processing_end_idx = len(examples)
    else:
        processing_end_idx = args.end_idx
    
    current_examples = examples[args.start_idx : processing_end_idx]
    
    model_short_name = args.model_name_or_path.rstrip('/').split("/")[-1]
    mode_suffix = build_mode_suffix(args)
    out_file_prefix = f'{args.split}_{args.prompt_type}_t{args.temperature}{mode_suffix}'
    save_dir = f'{args.output_dir}/{model_short_name}/{data_name}'
    completion_dir = f'{args.completions_save_dir}/{model_short_name}/{data_name}'
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(completion_dir, exist_ok=True)
    
    out_file = f'{save_dir}/{out_file_prefix}_k{args.n_sampling}_s{args.start_idx}_e{processing_end_idx}.jsonl'
    summary_path = os.path.join(
        save_dir,
        f"summary_t{args.temperature}{mode_suffix}_k{args.k}.json"
    )

    if os.path.exists(out_file):
        print(f"Skipping {data_name}: Output file {out_file} already exists.")
        # If the file exists, try to read it and compute accuracy to return for the final summary
        try:
            processed_data = list(load_jsonl(out_file))
            metrics = compute_metrics(processed_data, args.k)

            # ===== Added: write summary =====
            summary = build_summary(args, model_short_name, data_name, processing_end_idx, metrics, resumed=True)

            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

            print(f"[Saved summary (resumed)] {summary_path}")
            # =============================

            # Existing rollouts are also included in the phase-2 entropy stats
            if entropy_jobs is not None:
                entropy_jobs.append({
                    "data_name": data_name,
                    "out_file": out_file,
                    "summary_path": summary_path,
                })

            return metrics
        except Exception:
            return 0

    # Prepare prompts
    if args.use_verl_reward or args.grade_mode == "paper":
        reward_fn = load_verl_reward_fn(args.verl_reward_path)
    else:
        reward_fn = None
    if is_mario_math_effective(args):
        mario_is_equiv = load_mario_is_equiv(args.mario_eval_path, required=True)
    else:
        mario_is_equiv = None

    prompt_batch = []
    system_prompt, few_shot_prompt, question_format = get_three_prompt(args.prompt_type, data_name)
    
    for example in tqdm(current_examples, desc=f"Preparing prompts for {data_name}"):
        cur_prompt = build_prompt_for_example(
            args=args,
            tokenizer=tokenizer,
            example=example,
            data_name=data_name,
            system_prompt=system_prompt,
            few_shot_prompt=few_shot_prompt,
            question_format=question_format,
        )
        
        prompt_batch.append(cur_prompt)

    # Inference generation
    file_outputs = []

    # Initialize the file_outputs structure
    for i, d in enumerate(current_examples):
        question = get_question_text(d, data_name)
        item = {
            "question": question,
            "prompt_text": prompt_batch[i],
            "generated_responses": [],
        }
        if args.use_verl_prompt and get_prompt_user_content(d.get("prompt")):
            item["prompt_messages"] = d["prompt"]
        if "id" in d: item["id"] = d["id"]
        if "source" in d: item["source"] = d["source"]
        if "data_source" in d: item["data_source"] = d["data_source"]
        if "extra_info" in d: item["extra_info"] = d["extra_info"]
        file_outputs.append(item)

    for cur_gen_epoch in range(generation_epoch):
        pkl_path = f'{completion_dir}/{out_file_prefix}_gen_round{cur_gen_epoch}.pkl'
        
        print(f"Generating epoch {cur_gen_epoch + 1}/{generation_epoch}...")
        completions = llm.generate(prompt_batch, sampling_params)
        save_completions(completions, pkl_path)
        
        for i, completion in enumerate(completions):
            resps = [o.text for o in completion.outputs]
            file_outputs[i]['generated_responses'].extend(resps)

    # Grade accuracy
    k = args.k
    
    for i in tqdm(range(len(file_outputs)), desc=f"Grading {data_name}"):
        d = current_examples[i]
        generated_responses = file_outputs[i]['generated_responses']

        if args.grade_mode == "paper":
            gt_ans = get_verl_ground_truth(d)
            extra_info = get_verl_extra_info(d)
            data_source = get_data_source(d, data_name)
            grade_infos = [
                grade_response_for_paper(
                    response=response,
                    gt=gt_ans,
                    data_name=data_name,
                    data_source=data_source,
                    extra_info=extra_info,
                    verl_reward_fn=reward_fn,
                    mario_is_equiv=mario_is_equiv,
                )
                for response in generated_responses
            ]
            generated_answers = [str(info.get("pred", "")) for info in grade_infos]
            is_correct_list = [bool(info.get("acc", info.get("score", 0.0))) for info in grade_infos]
        elif args.use_verl_reward:
            gt_ans = get_verl_ground_truth(d)
            extra_info = get_verl_extra_info(d)
            data_source = get_data_source(d, data_name)
            grade_infos = [
                reward_fn(data_source, response, gt_ans, extra_info)
                for response in generated_responses
            ]
            generated_answers = [str(info.get("pred", "")) for info in grade_infos]
            is_correct_list = [bool(info.get("acc", info.get("score", 0.0))) for info in grade_infos]
        else:
            gt_cot, gt_ans = parse_ground_truth(d, data_name)
            generated_answers = [extract_answer(r, data_name) for r in generated_responses]
            grade_infos = [
                grade_with_optional_mario(ans, gt_ans, data_name, mario_is_equiv=mario_is_equiv)
                for ans in generated_answers
            ]
            is_correct_list = [bool(info.get("acc", info.get("score", 0.0))) for info in grade_infos]

        is_correct = any(is_correct_list)
        num_generated = len(is_correct_list)
        num_correct = sum(is_correct_list)
        pass_at_k = empirical_pass_at_k(is_correct_list, k)
        
        file_outputs[i].update({
            'generated_answers': generated_answers,
            'gold_answer': gt_ans,
            'is_correct': is_correct,
            'answers_correctness': is_correct_list,
            'grade_infos': grade_infos,
            'num_generated': num_generated,
            'num_correct': num_correct,
            f'pass@{k}': pass_at_k,
        })
    # Save results
    save_jsonl(file_outputs, out_file)
    
    metrics = compute_metrics(file_outputs, k)

    print(
        f"{data_name} Result: "
        f"Mean@{k}: {metrics['mean_score']:.4f} "
        f"Best@{k}: {metrics['best_at_k_mean']:.4f} "
        f"Maj@{k}: {metrics['maj_at_k_mean']:.4f} "
        f"Pass@{k}: {metrics['pass_at_k']:.4f}"
    )

    # ================= Added: save summary.json =================
    summary = build_summary(args, model_short_name, data_name, processing_end_idx, metrics, resumed=False)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[Saved summary] {summary_path}")
    # ==========================================================

    # Record this dataset's output so phase 2 (after releasing vLLM) can compute actor/entropy uniformly
    if entropy_jobs is not None:
        entropy_jobs.append({
            "data_name": data_name,
            "out_file": out_file,
            "summary_path": summary_path,
        })

    return metrics

def shutdown_vllm(llm):
    """Explicitly shut down a vLLM engine and its TP worker processes / distributed
    state. Does NOT free memory by itself — the caller must drop its reference to
    the engine (del / set to None) and then run gc + empty_cache, see free_vllm_memory.
    """
    import contextlib

    # explicit engine shutdown when available (vLLM v1)
    with contextlib.suppress(Exception):
        if hasattr(llm, "shutdown") and callable(llm.shutdown):
            llm.shutdown()
    with contextlib.suppress(Exception):
        engine = getattr(llm, "llm_engine", None)
        if engine is not None and hasattr(engine, "shutdown"):
            engine.shutdown()

    # tear down the distributed/model-parallel state used by TP>1 runs
    with contextlib.suppress(Exception):
        from vllm.distributed.parallel_state import (
            destroy_model_parallel,
            destroy_distributed_environment,
        )
        destroy_model_parallel()
        destroy_distributed_environment()

def free_vllm_memory():
    """Run garbage collection and empty the CUDA cache. Call AFTER every reference
    to the vLLM engine has been dropped, so the freed memory is returned to the GPU.
    """
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def run_entropy_phase(args, tokenizer, entropy_jobs):
    """Phase 2: load a HF model once and compute training-consistent actor/entropy
    for every dataset produced in phase 1.

    The vLLM engine MUST already be released by the caller (see release_vllm)
    before this runs, so the HF model has enough GPU memory.

    For each job (out_file + summary_path) it rewrites the rollout jsonl with
    per-response actor_entropy_mean/response_len and writes the dataset-level
    actor_entropy_token_mean/total_response_tokens back into the summary.json.

    Datasets whose summary already has a non-null actor_entropy_token_mean are
    skipped unless --force_entropy is set. Returns {data_name: actor_entropy_token_mean}
    (including values loaded from summaries for skipped datasets) so the caller can
    render them in the final report.
    """
    import gc
    import torch
    from transformers import AutoModelForCausalLM
    from utils.entropy import compute_entropy_for_file_outputs

    def summary_entropy(summary_path):
        """Return existing actor_entropy_token_mean from a summary, or None."""
        if not os.path.exists(summary_path):
            return None
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                return json.load(f).get("actor_entropy_token_mean")
        except Exception:
            return None

    entropy_by_dataset = {}

    # First determine which datasets still need computing (skip those whose summary already has entropy, unless --force_entropy)
    pending_jobs = []
    for job in entropy_jobs:
        existing = summary_entropy(job["summary_path"])
        if existing is not None and not args.force_entropy:
            entropy_by_dataset[job["data_name"]] = existing
            print(f"[entropy][skip] {job['data_name']}: already has actor_entropy_token_mean={existing:.6f}")
        else:
            pending_jobs.append(job)

    if not pending_jobs:
        print("[entropy] all datasets already have actor/entropy; nothing to compute.")
        return entropy_by_dataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[entropy] loading HF model for actor/entropy on {device}: {args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(device).eval()

    for job in pending_jobs:
        out_file = job["out_file"]
        summary_path = job["summary_path"]
        data_name = job["data_name"]
        if not os.path.exists(out_file):
            print(f"[entropy][skip] missing rollout file: {out_file}")
            continue

        file_outputs = list(load_jsonl(out_file))
        missing_prompt = sum(1 for item in file_outputs if not item.get("prompt_text"))
        if missing_prompt:
            print(
                f"[entropy][warn] {data_name}: {missing_prompt} items lack prompt_text; "
                f"their responses will report null entropy. Use recompute_entropy.py "
                f"with --data_dir to rebuild prompts for legacy results."
            )

        total_ent_sum, total_tok = compute_entropy_for_file_outputs(
            model=model,
            tokenizer=tokenizer,
            file_outputs=file_outputs,
            temperature=args.temperature,
            device=device,
            progress_desc=f"entropy {data_name}",
        )
        dataset_entropy = (total_ent_sum / total_tok) if total_tok > 0 else None
        entropy_by_dataset[data_name] = dataset_entropy

        save_jsonl(file_outputs, out_file)

        if os.path.exists(summary_path):
            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    summary = json.load(f)
            except Exception:
                summary = {}
            summary["actor_entropy_token_mean"] = dataset_entropy
            summary["total_response_tokens"] = total_tok
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

        entropy_str = f"{dataset_entropy:.6f}" if dataset_entropy is not None else "None"
        print(f"[entropy] {data_name}: actor_entropy_token_mean={entropy_str} over {total_tok} tokens")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return entropy_by_dataset

def main():
    args = parse_args()
    set_seed(args.seed)
    
    # 1. Initialize Tokenizer and LLM (only once)
    print(f"Initializing LLM: {args.model_name_or_path}")
    available_gpus = os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    # Initialize the LLM engine
    llm = LLM(
        model=args.model_name_or_path,
        tensor_parallel_size=len(available_gpus),
        trust_remote_code=True,
        # Use the command-line parameter instead of a hardcoded 0.4
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_tokens + 2048 # Ensure enough context
    )

    # 2. Iterate over the evaluation list
    results = {}
    entropy_jobs = [] if args.compute_entropy else None

    print(f"Starting evaluation on datasets: {args.datasets}")

    known_datasets = set(DATASET_CONFIG) | set(VERL_DATASET_CONFIG)
    for data_name in args.datasets:
        if data_name not in known_datasets:
            print(f"Skipping unknown dataset name: {data_name}")
            continue

        file_name = DATASET_CONFIG.get(data_name, f"{data_name}.jsonl")
        try:
            metrics = infer_single_dataset(args, llm, tokenizer, data_name, file_name, entropy_jobs=entropy_jobs)
            results[data_name] = metrics
        except Exception as e:
            print(f"Error evaluating {data_name}: {e}")
            import traceback
            traceback.print_exc()
            results[data_name] = "Error"

    # 2.5 Phase 2: fully release vLLM (including TP worker processes) before loading the HF model to compute actor/entropy
    entropy_by_dataset = {}
    if args.compute_entropy and entropy_jobs:
        # Key: shut down the engine first, then drop the reference to llm, and only then run gc + empty_cache;
        # otherwise the llm object is still referenced, vLLM memory is not freed, and the HF model will OOM.
        shutdown_vllm(llm)
        llm = None
        free_vllm_memory()
        entropy_by_dataset = run_entropy_phase(args, tokenizer, entropy_jobs)

    # 3. Print the final summary report (one column per dataset, including actor/entropy)
    print("\n" + "=" * 72)
    model_short_name = args.model_name_or_path.rstrip('/').split("/")[-1]
    print("FINAL EVALUATION REPORT")
    print(f"Model: {model_short_name}")

    dataset_names = list(results.keys())
    metric_rows = [
        (f"Mean@{args.k}", "mean_score", True),
        (f"Best@{args.k}", "best_at_k_mean", True),
        (f"Maj@{args.k}", "maj_at_k_mean", True),
        (f"Pass@{args.k}", "pass_at_k", True),
        ("actor/entropy", "actor_entropy_token_mean", False),
    ]

    metric_col_w = 14
    data_col_w = max(12, max((len(name) for name in dataset_names), default=12))

    header = f"{'Metric':<{metric_col_w}} | " + " | ".join(f"{name:<{data_col_w}}" for name in dataset_names)
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for label, key, is_pct in metric_rows:
        cells = []
        for name in dataset_names:
            metrics = results.get(name)
            if key == "actor_entropy_token_mean":
                value = entropy_by_dataset.get(name)
                cell = f"{value:.6f}" if isinstance(value, (int, float)) else "N/A"
            elif isinstance(metrics, dict):
                cell = f"{metrics[key]:.2%}" if is_pct else f"{metrics[key]}"
            else:
                cell = str(metrics)  # e.g. "Error"
            cells.append(f"{cell:<{data_col_w}}")
        print(f"{label:<{metric_col_w}} | " + " | ".join(cells))
    print("=" * len(header))

if __name__ == "__main__":
    main()
