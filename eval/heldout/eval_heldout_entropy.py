#!/usr/bin/env python3
"""Held-out OOD eval on MMLU-Pro (non-math) with training-consistent actor/entropy.

For ONE checkpoint:
  1. vLLM generates rollouts (n=1 per question by default, temp/top_p from args).
  2. ALL rollouts are saved verbatim (rollouts.jsonl).
  3. HF forward recomputes per-response token entropy using verl's
     entropy_from_logits + response-token token-mean  -> identical definition to
     the training-time `actor/entropy` metric, evaluated on held-out data.
  4. Multiple-choice grading (letter match), reusing eval/utils parsers.
  5. Summary (accuracy + mean actor/entropy) saved.

Kept separate from eval.py on purpose (eval.py uses vLLM only and cannot produce
the training-consistent entropy). Run via run_heldout_all_checks.sh for the 10
checkpoints.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
REPO_ROOT = os.path.abspath(os.path.join(EVAL_DIR, ".."))
for p in (EVAL_DIR, REPO_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from utils.utils import set_seed, load_jsonl  # noqa: E402
from utils.parser import parse_question, parse_ground_truth, extract_answer  # noqa: E402
from utils.grader import check_is_correct  # noqa: E402
from utils.entropy import entropy_from_logits, response_token_entropy  # noqa: E402

# Multiple-choice (QA) system prompt, same as eval/prompts/qwen/qa.py — used only
# for CHOICE datasets (e.g. mmlu_pro).
QA_SYSTEM_PROMPT = (
    'Answer the following multiple-choice question. Think step by step and then '
    'finish your answer with "The answer is (X)" where X is the correct letter choice.'
)
QA_QUESTION_FORMAT = "Question: {question}\n\nSolution:"


def _apply_template(tokenizer, messages, enable_thinking):
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if enable_thinking:
        kwargs["enable_thinking"] = True
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def build_prompt_from_choice(tokenizer, example, data_name, enable_thinking):
    """Build a chat prompt for a CHOICE (multiple-choice) jsonl example."""
    question = parse_question(example, data_name)
    user = QA_QUESTION_FORMAT.format(question=question)
    messages = [
        {"role": "system", "content": QA_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    return _apply_template(tokenizer, messages, enable_thinking)


def _to_plain(obj):
    """Normalize numpy arrays / records loaded from parquet into python types."""
    if isinstance(obj, np.ndarray):
        return [_to_plain(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(x) for x in obj]
    return obj


def load_examples(data_file, data_name, tokenizer, enable_thinking):
    """Return a list of normalized records, each with a pre-built prompt.

    Supports two formats:
      * .jsonl  -> the original CHOICE format (mmlu_pro), graded as a choice task.
      * .parquet -> verl format (prompt=messages, reward_model.ground_truth),
                    used for math datasets (amc23, dapo17k). The stored chat
                    messages (with the boxed math system prompt) are reused as-is,
                    keeping the prompt training-consistent.

    Each record: {id, category, data_source, question, ground_truth, prompt_text}
    """
    records = []
    if str(data_file).endswith(".parquet"):
        import pandas as pd
        df = pd.read_parquet(data_file)
        for i, row in df.iterrows():
            row = row.to_dict()
            messages = _to_plain(row.get("prompt"))
            if not isinstance(messages, list):
                raise ValueError(f"parquet row {i}: 'prompt' is not a message list")
            rm = _to_plain(row.get("reward_model")) or {}
            gt = rm.get("ground_truth")
            gt = None if gt is None else str(gt)
            extra = _to_plain(row.get("extra_info")) or {}
            question = extra.get("raw_problem")
            if not question:
                question = next(
                    (m.get("content", "") for m in reversed(messages)
                     if m.get("role") == "user"),
                    "",
                )
            records.append({
                "id": extra.get("index", i),
                "category": row.get("ability"),
                "data_source": row.get("data_source", data_name),
                "question": question,
                "ground_truth": gt,
                "prompt_text": _apply_template(tokenizer, messages, enable_thinking),
            })
        return records

    # .jsonl CHOICE format (mmlu_pro)
    for ex in load_jsonl(data_file):
        _, gt = parse_ground_truth(ex, data_name)
        records.append({
            "id": ex.get("id"),
            "category": ex.get("category"),
            "data_source": ex.get("source", data_name),
            "question": parse_question(ex, data_name),
            "ground_truth": gt,
            "prompt_text": build_prompt_from_choice(tokenizer, ex, data_name, enable_thinking),
        })
    return records


# response_entropy is kept as a thin wrapper so existing call sites and the
# training-consistent definition stay in the shared utils.entropy module.
response_entropy = response_token_entropy



def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name_or_path", required=True)
    ap.add_argument("--data_file", default=os.path.join(BASE_DIR, "data", "MMLU-Pro-100-nomath.jsonl"))
    ap.add_argument("--data_name", default="mmlu_pro",
                    help="dataset name driving the parser/grader "
                         "(mmlu_pro -> choice; amc23/dapo17k etc -> numeric/math)")
    ap.add_argument("--output_dir", required=True, help="per-checkpoint output dir")
    ap.add_argument("--n_sampling", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--top_k", type=int, default=-1)
    ap.add_argument("--max_tokens", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    ap.add_argument("--enable_thinking", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    data_name = args.data_name
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    examples = load_examples(args.data_file, data_name, tokenizer, args.enable_thinking)
    print(f"Loaded {len(examples)} held-out questions from {args.data_file} (data_name={data_name})")

    n_gpus = len(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(","))

    # ---- 1. vLLM rollout generation ----
    llm = LLM(
        model=args.model_name_or_path,
        tensor_parallel_size=n_gpus,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_tokens + 2048,
    )
    sp = SamplingParams(
        temperature=args.temperature,
        top_p=(1.0 if args.temperature == 0 else args.top_p),
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        n=args.n_sampling,
    )
    prompts = [ex["prompt_text"] for ex in examples]
    completions = llm.generate(prompts, sp)

    # collect responses per example
    per_ex_responses = []
    for comp in completions:
        per_ex_responses.append([o.text for o in comp.outputs])

    # free vLLM before loading HF model for entropy
    del llm
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # ---- 2/3. HF forward: training-consistent actor/entropy ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(device).eval()

    rollout_path = os.path.join(args.output_dir, "rollouts.jsonl")
    all_entropy_tokens = []   # for dataset-level actor/entropy (token-mean over ALL response tokens)
    total_ent_sum = 0.0
    total_tok = 0
    correct = 0
    total = 0

    with open(rollout_path, "w", encoding="utf-8") as fout:
        for i, ex in enumerate(tqdm(examples, desc="grading + entropy")):
            gt = ex["ground_truth"]
            prompt_text = prompts[i]
            responses = per_ex_responses[i]
            resp_records = []
            for r in responses:
                pred = extract_answer(r, data_name)
                is_corr = bool(check_is_correct(pred, gt, data_name=data_name))
                ent, rlen = response_entropy(
                    model, tokenizer, prompt_text, r, args.temperature, device
                )
                if ent is not None and rlen > 0:
                    resp_ent_mean = float(ent.mean().item())
                    total_ent_sum += float(ent.sum().item())
                    total_tok += rlen
                else:
                    resp_ent_mean = None
                resp_records.append({
                    "response": r,
                    "pred": pred,
                    "is_correct": is_corr,
                    "response_len": rlen,
                    "actor_entropy_mean": resp_ent_mean,
                })
                total += 1
                correct += int(is_corr)

            fout.write(json.dumps({
                "id": ex["id"],
                "category": ex["category"],
                "data_source": ex["data_source"],
                "question": ex["question"],
                "ground_truth": gt,
                "rollouts": resp_records,
            }, ensure_ascii=False) + "\n")

    # ---- 4. summary ----
    # dataset-level actor/entropy = token-mean over all response tokens (== training token-mean)
    dataset_actor_entropy = (total_ent_sum / total_tok) if total_tok > 0 else 0.0
    acc = (correct / total) if total > 0 else 0.0
    summary = {
        "model": os.path.basename(args.model_name_or_path.rstrip("/")),
        "data_name": data_name,
        "data_file": args.data_file,
        "n_questions": len(examples),
        "n_sampling": args.n_sampling,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "acc_mean": acc,                         # accuracy over all rollouts
        "actor_entropy_token_mean": dataset_actor_entropy,  # training-consistent actor/entropy
        "total_response_tokens": total_tok,
        "total_rollouts": total,
    }
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[saved] rollouts -> {rollout_path}")
    print(f"[saved] summary  -> {summary_path}")


if __name__ == "__main__":
    main()


