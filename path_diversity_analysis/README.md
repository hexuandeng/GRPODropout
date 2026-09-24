# Rollout Path Diversity Analysis

For the rollouts produced by sampling evaluation (`eval/eval.py`), this tool measures the
**solving-path diversity** of the multiple rollouts for **the same test set and the same
question**. It supports comparison across directories and across models, reporting per-question
diversity as well as (model, dataset) averages and a model-level overall score.

Two kinds of metrics:
- **Cosine similarity** (local `bge-m3` embeddings): `diversity = 1 - mean pairwise cosine`; higher = more diverse.
- **LLM as a judge** (OpenAI-compatible endpoint), two methods:
  - Solution clustering + cluster entropy (`normalized_entropy` in [0,1])
  - Sampled pairwise scoring (`pairwise_diversity` in [0,1])

Two "path-text views" are computed for each: `think` (only the `<think>` reasoning) and `full`
(the whole response); column names carry the `__think` / `__full` suffix.

## Layout

```
config.py              # config (paths / embedding / judge endpoint / sampling params), overridable via env vars
data_loader.py         # discover <model>/<dataset>/test_*.jsonl, load rollouts per question
text_utils.py          # <think> extraction / cleaning / truncation
cosine_diversity.py    # bge-m3 embeddings + pairwise cosine (with a vector cache)
llm_judge_diversity.py # the two judge methods (concurrency + retry + cache)
run_analysis.py        # main entry point, aggregation and comparison output
check_env.sh           # probe environment / dependencies / connectivity / model
requirements.txt
models/bge-m3/         # embedding model weights (NOT shipped, see note below)
data_root/             # put the <MODEL_NAME>/ directories to analyze here (default scan root)
outputs/               # per_question/ + summary/ + cache/
```

## Getting the bge-m3 weights

The `bge-m3` weights are **not shipped** with this repo. Download them (e.g. from HuggingFace
[`BAAI/bge-m3`](https://huggingface.co/BAAI/bge-m3)) into `models/bge-m3/`, or point the
`EMBED_MODEL` env var at an existing local copy:

```bash
huggingface-cli download BAAI/bge-m3 --local-dir models/bge-m3
# or, if the default HuggingFace host is unreachable, use a mirror:
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download BAAI/bge-m3 --local-dir models/bge-m3
```

`config.EMBED_MODEL` defaults to `models/bge-m3` under this directory.

## Steps

```bash
cd path_diversity_analysis

# 1. Probe the environment (any python in your conda env works; override with PYTHON=python3)
bash check_env.sh

# 2. Install missing dependencies (torch/openai may already be present)
pip install "sentence-transformers>=2.7" scikit-learn

# 3. Provide data: copy your <MODEL_NAME>/<dataset>/test_*.jsonl directories into data_root/
#    or pass --root pointing at your own sampling-output root

# 4. Run cosine first (no API needed), both think + full
python run_analysis.py --metrics cosine

# 5. Once the judge endpoint is configured, run all metrics
export JUDGE_BASE_URL=... JUDGE_API_KEY=... JUDGE_MODEL=...
python run_analysis.py
```

If you have a GPU, `EMBED_DEVICE` defaults to `cuda`; without a GPU, set `EMBED_DEVICE=cpu`.

## Common options

| Option | Description |
|---|---|
| `--root` | root of the sampling outputs (default `data_root/`) |
| `--models` | comma-separated model directory names, or `all` |
| `--datasets` | comma-separated dataset names, or `all` |
| `--metrics` | any combination of `cosine,judge_cluster,judge_pairwise` |
| `--path_text` | views for cosine: `think,full` |
| `--judge_path_text` | views for the judge (use `think` only to save cost) |
| `--max_questions` | max questions per group (for smoke tests) |

## Output

- `outputs/per_question/<model>__<dataset>.json` — per-question detail for each metric
- `outputs/summary/per_model_dataset.csv` — per (model, dataset) metric averages
- `outputs/summary/cross_model_compare.csv` — each model re-averaged across datasets (model-level score)
- `outputs/cache/` — embedding vectors & judge-call cache, so reruns don't recompute / re-spend

## Notes

- **When the judge endpoint is not configured**, `run_analysis.py` automatically skips the judge
  metrics, computes cosine only, and prints a hint; fill in the endpoint and rerun to add the judge
  results (cosine is served from cache and not recomputed).
- **Cost**: cosine is fully local with no API cost; running the judge on both `think` and `full`
  doubles the call count, so use `--judge_path_text think` to save money.
- The cache key includes the model name and a text hash, so switching the judge model or changing
  the text naturally triggers recomputation.
