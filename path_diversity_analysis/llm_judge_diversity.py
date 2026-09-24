"""LLM-as-a-judge path diversity (OpenAI-compatible chat completions).

Two methods:
  A) Solution clustering + cluster entropy  cluster_diversity()
     - Batch-ask the LLM to assign each rollout a short "solving strategy" label
     - Merge/dedup -> number of clusters + normalized entropy (higher = more diverse)
  B) Sampled pairwise scoring     pairwise_diversity()
     - Sample M pairs at random, ask the LLM for a similarity 1-5 / same-approach judgment
     - diversity = mean(1 - (sim-1)/4)

Calls use concurrency + exponential-backoff retry + disk cache (keyed by request-content hash).
When the judge is not configured (missing base_url/key/model), the caller should check
config.judge_configured() first and skip.
"""
import hashlib
import json
import math
import os
import pickle
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import config
from text_utils import truncate

_CLIENT = None
_CACHE_LOCK = threading.Lock()
_CACHE: Optional[Dict[str, dict]] = None


# --------------------------------------------------------------------------- #
# Client & cache
# --------------------------------------------------------------------------- #
def _get_client():
    global _CLIENT
    if _CLIENT is None:
        from openai import OpenAI
        _CLIENT = OpenAI(
            base_url=config.JUDGE_BASE_URL,
            api_key=config.JUDGE_API_KEY,
            timeout=config.JUDGE_TIMEOUT,
        )
    return _CLIENT


def _cache_file() -> str:
    tag = re.sub(r"[^A-Za-z0-9_.-]", "_", config.JUDGE_MODEL or "judge")
    return os.path.join(config.CACHE_DIR, f"judge_{tag}.pkl")


def _load_cache() -> Dict[str, dict]:
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    _CACHE = {}
    if config.USE_CACHE and os.path.exists(_cache_file()):
        try:
            with open(_cache_file(), "rb") as f:
                _CACHE = pickle.load(f)
        except Exception:
            _CACHE = {}
    return _CACHE


def _flush_cache():
    if not config.USE_CACHE:
        return
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    with _CACHE_LOCK:
        with open(_cache_file(), "wb") as f:
            pickle.dump(_CACHE, f)


def _req_hash(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(blob.encode("utf-8", "ignore")).hexdigest()


def _chat(messages: List[dict], cache_key: str) -> str:
    """One cached + retried chat call, returning the text content."""
    cache = _load_cache()
    if cache_key in cache:
        return cache[cache_key]["content"]

    client = _get_client()
    last_err = None
    for attempt in range(config.JUDGE_MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=config.JUDGE_MODEL,
                messages=messages,
                temperature=config.JUDGE_TEMPERATURE,
            )
            content = resp.choices[0].message.content or ""
            with _CACHE_LOCK:
                cache[cache_key] = {"content": content}
            return content
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(min(2 ** attempt, 30) + random.random())
    raise RuntimeError(f"judge chat failed after retries: {last_err}")


def _extract_json(text: str):
    """Best-effort JSON extraction from model output (tolerant of ```json wrapping or surrounding noise)."""
    if not text:
        return None
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if m:
        text = m.group(1)
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    # Fallback: grab the first { ... } or [ ... ]
    for pat in (r"\{.*\}", r"\[.*\]"):
        m = re.search(pat, text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                continue
    return None


# --------------------------------------------------------------------------- #
# Method A: solution clustering + cluster entropy
# --------------------------------------------------------------------------- #
_LABEL_SYS = (
    "You are an expert at analyzing mathematical problem-solving methods. "
    "Given several solutions to the same problem, summarize the CORE solving "
    "strategy/approach of each with a short English label "
    "(e.g., 'relative speed', 'set up equation', 'geometric similarity', "
    "'casework enumeration', 'algebraic manipulation'). "
    "Focus only on the method itself; ignore wording, formatting, and the final numeric answer. "
    "Output strictly JSON only, with no extra text."
)


def _label_batch(question: str, batch: List[Tuple[int, str]]) -> Dict[int, str]:
    """Assign a strategy label to a batch of (idx, solution); return {idx: label}."""
    items = []
    for idx, sol in batch:
        items.append({"index": idx, "solution": truncate(sol, config.JUDGE_MAX_CHARS)})
    user = (
        f"Problem:\n{truncate(question, 1000)}\n\n"
        f"Below are {len(items)} solutions (a JSON array, each with 'index' and 'solution').\n"
        f"Output one strategy label per solution as a JSON array, each item like "
        f'{{"index": <original index>, "strategy": "<short label>"}}.\n\n'
        f"Solutions:\n{json.dumps(items, ensure_ascii=False)}"
    )
    messages = [
        {"role": "system", "content": _LABEL_SYS},
        {"role": "user", "content": user},
    ]
    key = _req_hash({"task": "label", "model": config.JUDGE_MODEL, "q": question[:200],
                     "batch": [(i, s[:config.JUDGE_MAX_CHARS]) for i, s in batch]})
    content = _chat(messages, key)
    data = _extract_json(content)
    out: Dict[int, str] = {}
    if isinstance(data, list):
        for row in data:
            try:
                out[int(row["index"])] = str(row["strategy"]).strip().lower()
            except Exception:
                continue
    return out


def _normalize_label(lbl: str) -> str:
    lbl = lbl.strip().lower()
    lbl = re.sub(r"\s+", "", lbl)
    # Strip Chinese and English punctuation to help merge near-synonym labels
    lbl = re.sub(r"[。.,,、;;:：\"'`()（）\[\]【】\-_/]", "", lbl)
    return lbl


def cluster_diversity(question: str, solutions: List[str]) -> Dict[str, float]:
    """Method A: return the number of solution clusters and the normalized entropy."""
    n = len(solutions)
    if n < 2:
        return {"n": n, "n_distinct": n, "entropy": 0.0, "normalized_entropy": 0.0}

    labels: Dict[int, str] = {}
    batch_size = max(1, config.CLUSTER_BATCH)
    batches = [list(enumerate(solutions))[i:i + batch_size] for i in range(0, n, batch_size)]
    # Process batches sequentially (one request per batch; instant when cached)
    for b in batches:
        labels.update(_label_batch(question, b))

    # Missing entries each form their own cluster, to avoid failing the whole question
    norm = []
    for i in range(n):
        raw = labels.get(i, f"__unlabeled_{i}__")
        norm.append(_normalize_label(raw) or f"__empty_{i}__")

    from collections import Counter
    counts = Counter(norm)
    total = sum(counts.values())
    probs = [c / total for c in counts.values()]
    entropy = -sum(p * math.log(p) for p in probs if p > 0)
    norm_entropy = entropy / math.log(n) if n > 1 else 0.0
    return {
        "n": n,
        "n_distinct": len(counts),
        "entropy": entropy,
        "normalized_entropy": norm_entropy,
    }


# --------------------------------------------------------------------------- #
# Method B: sampled pairwise scoring
# --------------------------------------------------------------------------- #
_PAIR_SYS = (
    "You are an expert at comparing mathematical problem-solving methods. "
    "Given two solutions to the same problem, decide whether their CORE "
    "solving path/approach is the same, and give a similarity score. "
    "Focus only on the solving method; ignore wording, formatting, layout, "
    "and whether the final numeric answers match. "
    "Output strictly JSON only, with no extra text."
)


def _judge_pair(question: str, a: str, b: str) -> Optional[Dict[str, float]]:
    user = (
        f"Problem:\n{truncate(question, 1000)}\n\n"
        f"Solution A:\n{truncate(a, config.JUDGE_MAX_CHARS)}\n\n"
        f"Solution B:\n{truncate(b, config.JUDGE_MAX_CHARS)}\n\n"
        f'Output JSON: {{"similarity": <integer 1-5, 1=completely different approach, '
        f'5=identical approach>, "same_path": <true/false>}}.'
    )
    messages = [
        {"role": "system", "content": _PAIR_SYS},
        {"role": "user", "content": user},
    ]
    key = _req_hash({"task": "pair", "model": config.JUDGE_MODEL, "q": question[:200],
                     "a": a[:config.JUDGE_MAX_CHARS], "b": b[:config.JUDGE_MAX_CHARS]})
    content = _chat(messages, key)
    data = _extract_json(content)
    if not isinstance(data, dict):
        return None
    try:
        sim = float(data.get("similarity"))
        sim = max(1.0, min(5.0, sim))
    except Exception:
        return None
    same = data.get("same_path")
    same = bool(same) if same is not None else (sim >= 4)
    return {"similarity": sim, "same_path": same}


def pairwise_diversity(question: str, solutions: List[str],
                       m_pairs: Optional[int] = None,
                       seed: int = 3407) -> Dict[str, float]:
    """Method B: sampled pairwise scoring."""
    n = len(solutions)
    if n < 2:
        return {"n": n, "n_pairs": 0, "pairwise_diversity": 0.0, "distinct_path_ratio": 0.0}

    m_pairs = m_pairs or config.M_PAIRS
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    rng = random.Random(seed)
    rng.shuffle(all_pairs)
    pairs = all_pairs[:m_pairs]

    results: List[Dict[str, float]] = []
    with ThreadPoolExecutor(max_workers=config.JUDGE_CONCURRENCY) as ex:
        futs = {ex.submit(_judge_pair, question, solutions[i], solutions[j]): (i, j)
                for (i, j) in pairs}
        for fut in as_completed(futs):
            try:
                r = fut.result()
            except Exception:
                r = None
            if r is not None:
                results.append(r)

    if not results:
        return {"n": n, "n_pairs": 0, "pairwise_diversity": 0.0, "distinct_path_ratio": 0.0}

    # Similarity 1-5 -> normalized diversity 1-(sim-1)/4
    divs = [1.0 - (r["similarity"] - 1.0) / 4.0 for r in results]
    distinct = [0.0 if r["same_path"] else 1.0 for r in results]
    return {
        "n": n,
        "n_pairs": len(results),
        "pairwise_diversity": sum(divs) / len(divs),
        "distinct_path_ratio": sum(distinct) / len(distinct),
    }


def flush():
    """Flush the cache to disk (call after processing a group)."""
    _flush_cache()
