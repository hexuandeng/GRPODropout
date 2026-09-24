#!/usr/bin/env python3
"""Recompute training-consistent actor/entropy from already-saved rollouts.

Given eval results produced by eval.py (rollout jsonl + summary_*.json), reload
the checkpoint with HF and recompute per-response actor/entropy and the
dataset-level actor_entropy_token_mean, using the exact same definition as the
training-time metric and the held-out probe.

This does NOT re-generate anything — it only reads the saved responses.

Typical layout under a results root:
  <results_root>/<model>/<dataset>/<split>_..._k*_s*_e*.jsonl
  <results_root>/<model>/<dataset>/summary_t<temp>..._k*.json

Prompt reconstruction:
  - New results save "prompt_text" per item -> exact recompute.
  - Legacy results without "prompt_text" -> best-effort rebuild via --data_dir,
    reusing eval.py's build_prompt_for_example. This depends on the prompt-related
    flags (prompt_type / enable_thinking / surround_with_messages / use_verl_prompt)
    which are read from the sibling summary.json when available.

Usage:
  python recompute_entropy.py --model_name_or_path /path/ckpt \
      --results_root ./eval_results
  python recompute_entropy.py --model_name_or_path /path/ckpt \
      --result_file a.jsonl --result_file b.jsonl --temperature 1.0
"""

import argparse
import glob
import json
import os
import sys
import types

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from utils.utils import load_jsonl, save_jsonl  # noqa: E402
from utils.entropy import compute_entropy_for_file_outputs  # noqa: E402

# Reuse eval.py's prompt builders so the legacy-rebuild path cannot drift from
# the real eval definition.
import eval as eval_mod  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_name_or_path", required=True)
    ap.add_argument("--results_root", default=None,
                    help="Recurse this dir for rollout jsonl files (e.g. ./eval_results)")
    ap.add_argument("--result_file", action="append", default=[],
                    help="Explicit rollout jsonl file(s); can be repeated")
    ap.add_argument("--temperature", type=float, default=None,
                    help="Override sampling temperature; default reads each summary.json (fallback 1.0)")
    ap.add_argument("--data_dir", default=os.path.join(BASE_DIR, "data"),
                    help="Dataset dir for rebuilding prompts when prompt_text is missing (legacy results)")
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85,
                    help="Unused by HF forward; kept for CLI parity")
    ap.add_argument("--entropy_device", default=None, help="cuda / cpu; default auto")
    ap.add_argument("--dry_run", action="store_true", help="Compute and print but do not write files")
    return ap.parse_args()


def find_result_files(args):
    files = list(args.result_file)
    if args.results_root:
        pattern = os.path.join(args.results_root, "**", "*.jsonl")
        for path in glob.glob(pattern, recursive=True):
            # skip completion/pickle-style or clearly non-rollout files by name heuristic
            if os.path.basename(path).startswith("summary"):
                continue
            files.append(path)
    # de-dup, keep order
    seen = set()
    unique = []
    for path in files:
        ap = os.path.abspath(path)
        if ap not in seen and os.path.exists(ap):
            seen.add(ap)
            unique.append(ap)
    return unique


def find_summary_for(result_file):
    """Return the sibling summary_*.json path for a rollout jsonl, if present."""
    d = os.path.dirname(result_file)
    candidates = sorted(glob.glob(os.path.join(d, "summary_*.json")))
    return candidates[0] if candidates else None


def load_summary(summary_path):
    if summary_path and os.path.exists(summary_path):
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def build_rebuild_args(summary, data_dir):
    """Construct a minimal args namespace for eval_mod.build_prompt_for_example."""
    ns = types.SimpleNamespace(
        prompt_type=summary.get("prompt_type", "qwen"),
        enable_thinking=bool(summary.get("enable_thinking", False)),
        use_verl_prompt=bool(summary.get("use_verl_prompt", False)),
        use_few_shot=False,
        # eval.py sets surround_with_messages via CLI; summary does not record it.
        # Results are produced with --surround_with_messages in the run scripts,
        # so default to True for faithful reconstruction.
        surround_with_messages=bool(summary.get("surround_with_messages", True)),
        data_dir=data_dir,
    )
    return ns


def maybe_rebuild_prompts(file_outputs, data_name, summary, tokenizer, data_dir):
    """Fill missing prompt_text in-place using eval.py prompt builders.

    Returns the number of items whose prompt_text was rebuilt.
    """
    need = [item for item in file_outputs if not item.get("prompt_text")]
    if not need:
        return 0

    rebuild_args = build_rebuild_args(summary, data_dir)
    try:
        system_prompt, few_shot_prompt, question_format = eval_mod.get_three_prompt(
            rebuild_args.prompt_type, data_name
        )
    except Exception as e:
        print(f"[recompute][warn] cannot load prompts for {data_name}: {e}")
        return 0

    file_name = eval_mod.DATASET_CONFIG.get(data_name, f"{data_name}.jsonl")
    try:
        examples = eval_mod.load_examples(rebuild_args, data_name, file_name)
    except Exception as e:
        print(f"[recompute][warn] cannot load dataset {data_name} from {data_dir}: {e}")
        return 0

    # index examples by id when possible, else fall back to positional order
    by_id = {}
    for ex in examples:
        if isinstance(ex, dict) and "id" in ex:
            by_id[str(ex["id"])] = ex

    rebuilt = 0
    for pos, item in enumerate(file_outputs):
        if item.get("prompt_text"):
            continue
        ex = None
        if "id" in item and str(item["id"]) in by_id:
            ex = by_id[str(item["id"])]
        elif pos < len(examples):
            ex = examples[pos]
        if ex is None:
            continue
        try:
            item["prompt_text"] = eval_mod.build_prompt_for_example(
                args=rebuild_args,
                tokenizer=tokenizer,
                example=ex,
                data_name=data_name,
                system_prompt=system_prompt,
                few_shot_prompt=few_shot_prompt,
                question_format=question_format,
            )
            rebuilt += 1
        except Exception as e:
            print(f"[recompute][warn] rebuild prompt failed (pos={pos}): {e}")
    return rebuilt


def dataset_name_from_path(result_file):
    # <...>/<model>/<dataset>/<file>.jsonl  -> dataset is the parent dir name
    return os.path.basename(os.path.dirname(result_file))


def main():
    args = parse_args()
    files = find_result_files(args)
    if not files:
        print("No rollout jsonl files found. Provide --results_root or --result_file.")
        return

    print(f"Found {len(files)} rollout file(s) to recompute.")

    device = args.entropy_device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    print(f"[recompute] loading HF model on {device}: {args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(device).eval()

    for result_file in files:
        data_name = dataset_name_from_path(result_file)
        summary_path = find_summary_for(result_file)
        summary = load_summary(summary_path)
        temperature = args.temperature if args.temperature is not None else float(summary.get("temperature", 1.0))

        file_outputs = list(load_jsonl(result_file))
        if not file_outputs:
            print(f"[recompute][skip] empty file: {result_file}")
            continue

        rebuilt = maybe_rebuild_prompts(file_outputs, data_name, summary, tokenizer, args.data_dir)
        if rebuilt:
            print(f"[recompute] {data_name}: rebuilt prompt_text for {rebuilt} legacy items")

        total_ent_sum, total_tok = compute_entropy_for_file_outputs(
            model=model,
            tokenizer=tokenizer,
            file_outputs=file_outputs,
            temperature=temperature,
            device=device,
            progress_desc=f"entropy {data_name}",
        )
        dataset_entropy = (total_ent_sum / total_tok) if total_tok > 0 else None
        entropy_str = f"{dataset_entropy:.6f}" if dataset_entropy is not None else "None"
        print(
            f"[recompute] {data_name}: actor_entropy_token_mean={entropy_str} "
            f"over {total_tok} tokens (temp={temperature}) <- {result_file}"
        )

        if args.dry_run:
            continue

        save_jsonl(file_outputs, result_file)
        if summary_path:
            summary["actor_entropy_token_mean"] = dataset_entropy
            summary["total_response_tokens"] = total_tok
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            print(f"[recompute] updated summary -> {summary_path}")

    del model
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
