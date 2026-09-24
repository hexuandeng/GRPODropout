"""Dataset adapter for the prepared MMLU-Pro RLVR JSONL files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import datasets

from verl.utils.dataset.rl_dataset import RLHFDataset

LABELS = tuple("ABCDEFGHIJ")


def format_mmlu_pro_prompt(question: str, choices: list[dict[str, str]]) -> str:
    options = "\n".join(f"{choice['label']}. {choice['text']}" for choice in choices)
    return (
        "Answer the following multiple-choice question.\n\n"
        f"Question:\n{question}\n\n"
        f"Options:\n{options}\n\n"
        "Reason through the problem carefully. Finish with exactly "
        "\\boxed{LETTER}, where LETTER is one of the listed option labels."
    )


def _read_json_records(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("data", payload.get("train", payload))
        if not isinstance(payload, list):
            raise ValueError(f"{path}: JSON must contain a list of records")
        return payload

    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            records.append(record)
    return records


def _validate_choice_row(row: dict[str, Any], source: str) -> tuple[str, list[dict[str, str]], str]:
    question = row.get("question")
    choices = row.get("choices")
    answer = row.get("answer")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"{source}: question must be a non-empty string")
    if not isinstance(choices, list) or not choices:
        raise ValueError(f"{source}: choices must be a non-empty list")

    normalised = []
    labels = []
    for choice in choices:
        if not isinstance(choice, dict):
            raise ValueError(f"{source}: each choice must be an object")
        label = choice.get("label")
        text = choice.get("text")
        if not isinstance(label, str) or label.strip() not in LABELS:
            raise ValueError(f"{source}: invalid choice label {label!r}")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{source}: choice text must be a non-empty string")
        label = label.strip()
        labels.append(label)
        normalised.append({"label": label, "text": text.strip()})

    if len(set(labels)) != len(labels):
        raise ValueError(f"{source}: duplicate choice labels")
    if len(normalised) != 10 or tuple(labels) != LABELS:
        raise ValueError(f"{source}: ACL 2026 preparation requires exactly ordered A-J choices")
    if not isinstance(answer, str) or answer.strip() not in labels:
        raise ValueError(f"{source}: answer {answer!r} is not a choice label")
    return question.strip(), normalised, answer.strip()


def convert_mmlu_pro_row(row: dict[str, Any], row_index: int) -> dict[str, Any]:
    question, choices, answer = _validate_choice_row(row, f"row {row_index}")
    question_id = row.get("id")
    if not isinstance(question_id, (str, int)) or isinstance(question_id, bool) or not str(question_id).strip():
        raise ValueError(f"row {row_index}: id must be a non-empty string or integer")
    category = row.get("category")
    if not isinstance(category, str) or not category.strip():
        raise ValueError(f"row {row_index}: category must be a non-empty string")
    question_id = str(question_id).strip()
    category = category.strip()
    split = str(row.get("split", "unspecified"))
    prompt = format_mmlu_pro_prompt(question, choices)
    return {
        "data_source": "mmlu_pro",
        "prompt": [{"role": "user", "content": prompt}],
        "ability": category,
        "reward_model": {"style": "rule", "ground_truth": answer},
        "extra_info": {
            "index": question_id,
            "question_id": question_id,
            "category": category,
            "src": row.get("src"),
            "split": split,
            "num_choices": len(choices),
        },
    }


class MMLUProDataset(RLHFDataset):
    """Load prepared MMLU-Pro JSONL while retaining verl's tokenization path."""

    def _read_files_and_tokenize(self):
        suffixes = {Path(str(data_file)).suffix.lower() for data_file in self.data_files}
        json_suffixes = {".json", ".jsonl"}
        parquet_suffixes = {".parquet", ".pq"}
        if not suffixes.issubset(json_suffixes | parquet_suffixes):
            raise ValueError(f"Unsupported MMLU-Pro data file extension(s): {sorted(suffixes)}")
        if suffixes & json_suffixes and suffixes & parquet_suffixes:
            raise ValueError("MMLU-Pro dataset cannot mix JSON/JSONL and Parquet files")

        dataframes = []
        for data_file in self.data_files:
            suffix = Path(str(data_file)).suffix.lower()
            if suffix in json_suffixes:
                records = _read_json_records(data_file)
                if not records:
                    raise ValueError(f"{data_file}: prepared MMLU-Pro file is empty")
                converted = [convert_mmlu_pro_row(row, i) for i, row in enumerate(records)]
                dataframes.append(datasets.Dataset.from_list(converted))
            elif suffix in {".parquet", ".pq"}:
                dataframes.append(datasets.load_dataset("parquet", data_files=data_file)["train"])
            else:
                raise ValueError(f"Unsupported MMLU-Pro data file extension: {data_file}")

        if not dataframes:
            raise ValueError("No MMLU-Pro data files were provided")
        self.dataframe = datasets.concatenate_datasets(dataframes)
        print(f"dataset len: {len(self.dataframe)}")
        self.dataframe = self.maybe_filter_out_long_prompts(self.dataframe)

    def _apply_chat_template(self, processing_class, messages, add_generation_prompt=True, tokenize=False):
        """Keep prompt filtering compatible with older verl RLHFDataset versions."""
        return processing_class.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=tokenize,
            **getattr(self, "chat_template_kwargs", {}),
        )
