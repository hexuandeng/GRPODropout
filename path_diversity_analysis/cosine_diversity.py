"""Cosine-similarity path diversity.

For the N path texts of a question, embed them and compute pairwise cosine
similarity (upper triangle); diversity = 1 - mean_pairwise_similarity. Higher = more diverse.

Embedding backend: local sentence-transformers (bge-m3). The model is loaded once and reused.
Disk cache: vectors are cached by (embed_model, text sha1), so reruns don't recompute.
"""
import hashlib
import os
import pickle
from typing import Dict, List, Optional

import numpy as np

import config
from text_utils import truncate

_MODEL = None          # lazily loaded SentenceTransformer
_MODEL_TAG = None      # model identifier used in the cache key


def _get_model():
    global _MODEL, _MODEL_TAG
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        path = config.EMBED_MODEL
        if not os.path.isdir(path):
            raise FileNotFoundError(
                f"Embedding model directory not found: {path}\n"
                f"Download bge-m3 into that path first (or set the EMBED_MODEL environment variable)."
            )
        _MODEL = SentenceTransformer(path, device=config.EMBED_DEVICE)
        try:
            _MODEL.max_seq_length = config.EMBED_MAX_LENGTH
        except Exception:
            pass
        _MODEL_TAG = os.path.basename(path.rstrip("/"))
    return _MODEL


def _text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()


def _cache_path(model_tag: str) -> str:
    return os.path.join(config.CACHE_DIR, f"emb_{model_tag}.pkl")


def _load_cache(model_tag: str) -> Dict[str, np.ndarray]:
    if not config.USE_CACHE:
        return {}
    p = _cache_path(model_tag)
    if os.path.exists(p):
        try:
            with open(p, "rb") as f:
                return pickle.load(f)
        except Exception:
            return {}
    return {}


def _save_cache(model_tag: str, cache: Dict[str, np.ndarray]):
    if not config.USE_CACHE:
        return
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    with open(_cache_path(model_tag), "wb") as f:
        pickle.dump(cache, f)


def embed_texts(texts: List[str]) -> np.ndarray:
    """Return an (len(texts), dim) matrix of L2-normalized vectors, with caching."""
    model = _get_model()
    cache = _load_cache(_MODEL_TAG)

    trunc = [truncate(t, config.EMBED_MAX_LENGTH * 4) for t in texts]  # coarse truncation; the model truncates again by token
    keys = [_text_hash(t) for t in trunc]

    missing_idx = [i for i, k in enumerate(keys) if k not in cache]
    if missing_idx:
        to_encode = [trunc[i] for i in missing_idx]
        vecs = model.encode(
            to_encode,
            batch_size=config.EMBED_BATCH_SIZE,
            normalize_embeddings=config.EMBED_NORMALIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        for j, i in enumerate(missing_idx):
            cache[keys[i]] = vecs[j].astype(np.float32)
        _save_cache(_MODEL_TAG, cache)

    return np.stack([cache[k] for k in keys], axis=0)


def pairwise_cosine_stats(vectors: np.ndarray) -> Dict[str, float]:
    """Given an (N, dim) matrix of normalized vectors, compute upper-triangle pairwise cosine stats and diversity."""
    n = vectors.shape[0]
    if n < 2:
        return {"diversity": 0.0, "mean_sim": 1.0, "std_sim": 0.0,
                "min_sim": 1.0, "max_sim": 1.0, "n_pairs": 0, "n": n}
    sim = vectors @ vectors.T
    iu = np.triu_indices(n, k=1)
    pair_sims = sim[iu]
    mean_sim = float(np.mean(pair_sims))
    return {
        "diversity": 1.0 - mean_sim,
        "mean_sim": mean_sim,
        "std_sim": float(np.std(pair_sims)),
        "min_sim": float(np.min(pair_sims)),
        "max_sim": float(np.max(pair_sims)),
        "n_pairs": int(len(pair_sims)),
        "n": n,
    }


def cosine_diversity_for_texts(texts: List[str]) -> Dict[str, float]:
    if len(texts) < 2:
        return {"diversity": 0.0, "mean_sim": 1.0, "std_sim": 0.0,
                "min_sim": 1.0, "max_sim": 1.0, "n_pairs": 0, "n": len(texts)}
    vecs = embed_texts(texts)
    return pairwise_cosine_stats(vecs)
