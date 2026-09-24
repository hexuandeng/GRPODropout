"""Central configuration. All tunable parameters live here, or can be
overridden via the matching environment variables.

The judge endpoint is left empty by default: when unset, run_analysis.py
automatically skips the judge metrics and only runs cosine. Once you have an
OpenAI-compatible endpoint, fill in JUDGE_BASE_URL / JUDGE_API_KEY / JUDGE_MODEL.
"""
import os

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))

# Root directory of the sampling outputs to analyze. Defaults to data_root/
# under this directory.
# Directory layout: <ROOT>/<MODEL_NAME>/<dataset>/test_*.jsonl
DATA_ROOT = os.environ.get("DATA_ROOT", os.path.join(HERE, "data_root"))

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.path.join(HERE, "outputs"))
PER_QUESTION_DIR = os.path.join(OUTPUT_DIR, "per_question")
SUMMARY_DIR = os.path.join(OUTPUT_DIR, "summary")
CACHE_DIR = os.path.join(OUTPUT_DIR, "cache")

# ----------------------------------------------------------------------------
# Path-text extraction
# ----------------------------------------------------------------------------
# Which text views to analyze: think (only the <think> reasoning) / full (the whole response)
PATH_TEXT_MODES = ["think", "full"]
# Separately control which text views the judge runs on (to save cost, e.g. think only)
JUDGE_PATH_TEXT_MODES = ["think", "full"]

# ----------------------------------------------------------------------------
# Cosine / embedding backend (sentence-transformers, local)
# ----------------------------------------------------------------------------
# Local bge-m3 weights directory. Defaults to models/bge-m3 under this directory.
# NOTE: the bge-m3 weights are NOT shipped with this repo. Download them (e.g.
# from HuggingFace `BAAI/bge-m3`) into models/bge-m3/, or point EMBED_MODEL at
# an existing local copy.
EMBED_MODEL = os.environ.get("EMBED_MODEL", os.path.join(HERE, "models", "bge-m3"))
EMBED_DEVICE = os.environ.get("EMBED_DEVICE", "cuda")   # use "cpu" if no GPU is available
EMBED_BATCH_SIZE = int(os.environ.get("EMBED_BATCH_SIZE", "32"))
EMBED_MAX_LENGTH = int(os.environ.get("EMBED_MAX_LENGTH", "2048"))  # truncation length (tokens)
EMBED_NORMALIZE = True  # after L2 normalization, the dot product equals cosine similarity

# ----------------------------------------------------------------------------
# LLM-as-a-judge (OpenAI-compatible chat completions)
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# LLM-as-a-judge (OpenAI-compatible chat completions)
# Works with both cloud APIs and locally deployed models (vLLM / Ollama /
# LMDeploy / TGI, etc.), as long as they expose an OpenAI-compatible
# /v1/chat/completions endpoint -- just change base_url.
#   Local vLLM example:  JUDGE_BASE_URL=http://127.0.0.1:8000/v1  JUDGE_API_KEY=EMPTY  JUDGE_MODEL=<name of the model loaded by vllm>
#   Local Ollama example: JUDGE_BASE_URL=http://127.0.0.1:11434/v1 JUDGE_API_KEY=ollama JUDGE_MODEL=qwen2.5
# ----------------------------------------------------------------------------
JUDGE_BASE_URL = os.environ.get("JUDGE_BASE_URL", "")   # e.g. https://xxx/v1 or http://127.0.0.1:8000/v1
JUDGE_API_KEY = os.environ.get("JUDGE_API_KEY", "")     # for local deployments, any non-empty string works, e.g. EMPTY
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "")         # cloud: gpt-4o-mini; local: the model name in vLLM/Ollama

JUDGE_CONCURRENCY = int(os.environ.get("JUDGE_CONCURRENCY", "8"))
JUDGE_MAX_RETRIES = int(os.environ.get("JUDGE_MAX_RETRIES", "5"))
JUDGE_TIMEOUT = float(os.environ.get("JUDGE_TIMEOUT", "120"))
JUDGE_TEMPERATURE = float(os.environ.get("JUDGE_TEMPERATURE", "0.0"))

# Method A (solution clustering): how many rollouts to send per request for batch labeling
CLUSTER_BATCH = int(os.environ.get("CLUSTER_BATCH", "8"))
# Method B (pairwise scoring): how many random pairs to sample per question
M_PAIRS = int(os.environ.get("M_PAIRS", "40"))
# Max characters per rollout sent to the judge (controls token count / cost)
JUDGE_MAX_CHARS = int(os.environ.get("JUDGE_MAX_CHARS", "6000"))

# ----------------------------------------------------------------------------
# General
# ----------------------------------------------------------------------------
USE_CACHE = os.environ.get("USE_CACHE", "1") != "0"
RANDOM_SEED = int(os.environ.get("RANDOM_SEED", "3407"))


def judge_configured() -> bool:
    return bool(JUDGE_BASE_URL and JUDGE_API_KEY and JUDGE_MODEL)


def ensure_output_dirs():
    for d in (PER_QUESTION_DIR, SUMMARY_DIR, CACHE_DIR):
        os.makedirs(d, exist_ok=True)
