"""Answer diversity.

Distinct from "path diversity": here we only look at the final answer extracted
from each rollout (generated_answers) and measure how spread out the N answers
for a given question are. Fully local, no model or API needed.

Metrics (for the N answers of a question):
  - n_distinct            : number of distinct answers after dedup (empty answers grouped as "<empty>")
  - distinct_ratio        : n_distinct / n
  - normalized_entropy    : normalized entropy of the answer distribution, in [0,1]; higher = more spread out
  - top1_share            : share of the mode (most frequent answer); higher = more concentrated
  - empty_ratio           : share of empty answers (no \\boxed{} etc. extracted)
  - gold_share            : fraction matching the gold answer (if gold is provided; correctness is more reliable)
  - correct_answer_present: whether at least one is correct
"""
import math
import re
from collections import Counter
from typing import Dict, List, Optional


def _norm_answer(a: Optional[str]) -> str:
    if a is None:
        return "<empty>"
    s = str(a).strip()
    if s == "" or s.lower() in ("none", "null", "nan"):
        return "<empty>"
    # Light normalization: strip whitespace, outer $, and trailing period
    s = s.replace("$", "").strip()
    s = re.sub(r"\s+", "", s)
    s = s.rstrip(".")
    return s if s else "<empty>"


def answer_diversity(answers: List[str],
                     correctness: Optional[List[bool]] = None,
                     gold: Optional[str] = None) -> Dict[str, float]:
    n = len(answers)
    if n == 0:
        return {"n": 0, "n_distinct": 0, "distinct_ratio": 0.0,
                "normalized_entropy": 0.0, "top1_share": 0.0,
                "empty_ratio": 0.0, "gold_share": None,
                "correct_answer_present": None}

    norm = [_norm_answer(a) for a in answers]
    empty = sum(1 for x in norm if x == "<empty>")

    counts = Counter(norm)
    total = sum(counts.values())
    probs = [c / total for c in counts.values()]
    entropy = -sum(p * math.log(p) for p in probs if p > 0)
    norm_entropy = entropy / math.log(n) if n > 1 else 0.0
    top1 = max(counts.values()) / total

    # Correctness (prefer precomputed correctness when available; otherwise fall back to gold string matching)
    gold_share = None
    correct_present = None
    if correctness and len(correctness) == n and any(c is not None for c in correctness):
        cc = [bool(c) for c in correctness]
        gold_share = sum(1 for c in cc if c) / n
        correct_present = any(cc)
    elif gold is not None:
        g = _norm_answer(gold)
        gold_share = sum(1 for x in norm if x == g) / n
        correct_present = gold_share > 0

    return {
        "n": n,
        "n_distinct": len(counts),
        "distinct_ratio": len(counts) / n,
        "normalized_entropy": norm_entropy,
        "top1_share": top1,
        "empty_ratio": empty / n,
        "gold_share": gold_share,
        "correct_answer_present": correct_present,
    }
