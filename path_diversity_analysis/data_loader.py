"""Discover sampling-output directories, load jsonl, and extract each question's rollout path texts.

Directory convention (matching the output of eval/eval.py):
    <ROOT>/<MODEL_NAME>/<dataset>/test_*.jsonl
Each line is one question, containing fields like generated_responses: list[str].
"""
import glob
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from text_utils import get_path_text


@dataclass
class QuestionUnit:
    """One (model, dataset, question) analysis unit."""
    model: str
    dataset: str
    qid: str
    question: str
    responses: List[str]                      # raw rollout texts
    correctness: Optional[List[bool]] = None  # whether each is correct
    gold_answer: Optional[str] = None
    answers: Optional[List[str]] = None        # the final answer extracted from each

    def _subset_idx(self, subset: str) -> List[int]:
        """Return the rollout indices for the given subset.
        subset='all'    : all
        subset='correct': only the correct ones (requires correctness); empty if no labels
        """
        if subset == "all":
            return list(range(len(self.responses)))
        if subset == "correct":
            if not self.correctness:
                return []
            return [i for i, c in enumerate(self.correctness) if c]
        raise ValueError(f"unknown subset: {subset}")

    def path_texts(self, mode: str, subset: str = "all") -> List[str]:
        idx = self._subset_idx(subset)
        return [get_path_text(self.responses[i], mode) for i in idx]

    def responses_subset(self, subset: str = "all") -> List[str]:
        return [self.responses[i] for i in self._subset_idx(subset)]

    def answers_subset(self, subset: str = "all") -> List[str]:
        if not self.answers:
            return []
        return [self.answers[i] for i in self._subset_idx(subset)]

    def correctness_subset(self, subset: str = "all") -> Optional[List[bool]]:
        if not self.correctness:
            return None
        return [self.correctness[i] for i in self._subset_idx(subset)]

    def n_subset(self, subset: str = "all") -> int:
        return len(self._subset_idx(subset))

    @property
    def n(self) -> int:
        return len(self.responses)

    @property
    def accuracy(self) -> Optional[float]:
        if not self.correctness:
            return None
        return sum(1 for x in self.correctness if x) / len(self.correctness)


@dataclass
class ModelDatasetGroup:
    model: str
    dataset: str
    jsonl_path: str
    units: List[QuestionUnit] = field(default_factory=list)


def _iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _pick_jsonl_from(cands: List[str]) -> Optional[str]:
    """Pick the main result from candidate jsonls (prefer test_*.jsonl, take the largest)."""
    tests = [p for p in cands if os.path.basename(p).startswith("test_")]
    pool = tests or cands
    if not pool:
        return None
    return max(pool, key=lambda p: os.path.getsize(p))


def discover_groups(root: str,
                    models: Optional[List[str]] = None,
                    datasets: Optional[List[str]] = None,
                    exclude_models: Optional[List[str]] = None,
                    exclude_datasets: Optional[List[str]] = None) -> List[ModelDatasetGroup]:
    """Recursively discover <...>/<dataset>/test_*.jsonl under root.

    Handles different nesting depths:
      - <root>/<model>/<dataset>/test_*.jsonl
      - <root>/<model>/<model>/<dataset>/test_*.jsonl   (double-level naming from the eval script)
    Convention: dataset = the directory holding the jsonl; model = the first-level directory under root.

    models/datasets: allowlist (exact match), None means all.
    exclude_*: exact-match exclusion, takes priority over the allowlist.
    Exact matching ensures 'balance-neg' does not accidentally hit 'balance-neg1'.
    """
    if not os.path.isdir(root):
        raise FileNotFoundError(f"DATA_ROOT does not exist: {root}")

    root = os.path.abspath(root)
    excl_m = set(exclude_models or [])
    excl_d = set(exclude_datasets or [])

    # Collect every directory containing jsonl -> list of jsonls in that directory
    dir_to_jsonls: Dict[str, List[str]] = {}
    for dirpath, _dirnames, filenames in os.walk(root):
        jsonls = [os.path.join(dirpath, fn) for fn in filenames if fn.endswith(".jsonl")]
        if jsonls:
            dir_to_jsonls[dirpath] = jsonls

    groups: List[ModelDatasetGroup] = []
    for ds_dir, jsonls in sorted(dir_to_jsonls.items()):
        rel = os.path.relpath(ds_dir, root)
        parts = rel.split(os.sep)
        if len(parts) < 2:
            # jsonl sits directly under root; can't tell model/dataset apart, skip
            continue
        model_name = parts[0]           # first level under root = model
        ds_name = parts[-1]             # directory holding the jsonl = dataset

        if models and model_name not in models:
            continue
        if model_name in excl_m:
            continue
        if datasets and ds_name not in datasets:
            continue
        if ds_name in excl_d:
            continue

        jsonl = _pick_jsonl_from(jsonls)
        if jsonl is None:
            continue
        groups.append(ModelDatasetGroup(model=model_name, dataset=ds_name, jsonl_path=jsonl))

    return groups


def load_units(group: ModelDatasetGroup,
               max_questions: Optional[int] = None,
               min_rollouts: int = 2) -> List[QuestionUnit]:
    """Load the question units of a group. Skip questions with too few rollouts."""
    units: List[QuestionUnit] = []
    for idx, rec in enumerate(_iter_jsonl(group.jsonl_path)):
        if max_questions is not None and len(units) >= max_questions:
            break
        raw_responses = rec.get("generated_responses") or []
        raw_answers = rec.get("generated_answers") or []
        raw_correct = rec.get("answers_correctness") or []

        # Align by original index; drop only entries with empty response, keeping responses/answers/correctness consistent
        responses, answers, correctness = [], [], []
        for i, r in enumerate(raw_responses):
            if not (isinstance(r, str) and r.strip()):
                continue
            responses.append(r)
            answers.append(str(raw_answers[i]) if i < len(raw_answers) else "")
            correctness.append(bool(raw_correct[i]) if i < len(raw_correct) else None)

        if len(responses) < min_rollouts:
            continue
        qid = str(rec.get("id", idx))
        has_correct = any(c is not None for c in correctness)
        units.append(QuestionUnit(
            model=group.model,
            dataset=group.dataset,
            qid=qid,
            question=str(rec.get("question", ""))[:500],
            responses=responses,
            correctness=correctness if has_correct else None,
            answers=answers if any(a for a in answers) else None,
            gold_answer=str(rec.get("gold_answer", "")) if "gold_answer" in rec else None,
        ))
    group.units = units
    return units
