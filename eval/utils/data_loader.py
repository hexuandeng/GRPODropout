import os
import json
import random
from utils.utils import load_jsonl, lower_keys

# Mapping from dataset name to file name
DATASET_FILES = {
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

CHOICE_DATASETS = {"arc_challenge", "mmlu_pro", "gpqa_diamond", "supergpqa"}

def read_parquet_records(path):
    try:
        import pandas as pd
        return pd.read_parquet(path).to_dict("records")
    except ImportError:
        import pyarrow.parquet as pq
        return pq.read_table(path).to_pylist()

def normalize_choice_answer(answer, choices):
    answer = "" if answer is None else str(answer).strip()
    if not answer or not isinstance(choices, list):
        return answer

    labels = [str(choice.get("label", "")).strip().upper() for choice in choices if isinstance(choice, dict)]
    answer_upper = answer.upper()
    if answer_upper in labels:
        return answer_upper

    if answer.isdigit():
        idx = int(answer) - 1
        if 0 <= idx < len(labels):
            return labels[idx]

    return answer_upper

def load_data(data_name, split, data_dir='./data'):
    """
    Load data and return it uniformly as a list format
    """
    # 1. Determine the file name
    if data_name in DATASET_FILES:
        filename = DATASET_FILES[data_name]
    else:
        # If not in the mapping, fall back to the default naming rule
        filename = f"{data_name}.jsonl"

    file_path = os.path.join(data_dir, filename)

    if not os.path.exists(file_path):
        # Try the default HF structure (fallback)
        fallback_path = f"{data_dir}/{data_name}/{split}.jsonl"
        if os.path.exists(fallback_path):
            file_path = fallback_path
        else:
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

    print(f"Loading data from {file_path}...")

    # 2. Load data based on the file extension
    examples = []
    if file_path.endswith('.parquet'):
        # Requires pandas and pyarrow: pip install pandas pyarrow
        examples = read_parquet_records(file_path)
    elif file_path.endswith('.jsonl'):
        examples = list(load_jsonl(file_path))
    elif file_path.endswith('.json'):
        with open(file_path, 'r', encoding='utf-8') as f:
            examples = json.load(f)
            if isinstance(examples, dict): # Handle cases where the json is wrapped in an extra layer
                examples = list(examples.values())[0]
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

    # 3. Normalize key names (standardization)
    # Convert all data to lowercase keys and ensure a 'question' field exists
    normalized_examples = []
    for idx, ex in enumerate(examples):
        new_ex = lower_keys(ex)

        # Map the Question field
        if 'question' not in new_ex:
            for key in ['problem', 'input', 'content']:
                if key in new_ex:
                    new_ex['question'] = new_ex[key]
                    break

        # Map the Answer field (keep the original ground truth for evaluation)
        if 'answer' not in new_ex:
            for key in ['solution', 'final_answer', 'output']:
                if key in new_ex:
                    new_ex['answer'] = new_ex[key]
                    break

        if data_name in CHOICE_DATASETS and 'answer' in new_ex:
            new_ex['answer'] = normalize_choice_answer(new_ex.get('answer'), new_ex.get('choices'))

        # Ensure an ID exists
        if 'id' not in new_ex:
            new_ex['id'] = idx
            
        normalized_examples.append(new_ex)

    print(f"Loaded {len(normalized_examples)} examples from {data_name}")
    return normalized_examples
