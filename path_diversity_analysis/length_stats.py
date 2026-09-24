"""Response length statistics. Used to explain the empty-answer rate:
overly long reasoning -> 8k truncation -> no \\boxed{} -> empty answer.

Everything is computed directly from the raw rollout text; no model/tokenizer needed.
Key signal think_unclosed_ratio: fraction missing </think> = a proxy for the truncation rate.
"""
import statistics
from typing import Dict, List, Optional


def _is_empty_answer(a: Optional[str]) -> bool:
    if a is None:
        return True
    s = str(a).strip().replace("$", "")
    return s == "" or s.lower() in ("none", "null", "nan")


def length_stats(responses: List[str],
                 answers: Optional[List[str]] = None) -> Dict[str, float]:
    n = len(responses)
    if n == 0:
        return {"n": 0}

    char_lens = [len(r) for r in responses]
    word_lens = [len(r.split()) for r in responses]
    unclosed = sum(1 for r in responses if "</think>" not in r)

    out = {
        "n": n,
        "char_len_mean": statistics.mean(char_lens),
        "char_len_median": statistics.median(char_lens),
        "char_len_max": max(char_lens),
        "word_len_mean": statistics.mean(word_lens),
        "think_unclosed_ratio": unclosed / n,
    }

    # Compare lengths of empty vs non-empty answers (requires answers aligned)
    if answers and len(answers) == n:
        empty_lens = [len(responses[i]) for i in range(n) if _is_empty_answer(answers[i])]
        nonempty_lens = [len(responses[i]) for i in range(n) if not _is_empty_answer(answers[i])]
        em = statistics.mean(empty_lens) if empty_lens else None
        nm = statistics.mean(nonempty_lens) if nonempty_lens else None
        out["char_len_empty_mean"] = em
        out["char_len_nonempty_mean"] = nm
        out["len_gap_empty_minus_nonempty"] = (em - nm) if (em is not None and nm is not None) else None
    return out
