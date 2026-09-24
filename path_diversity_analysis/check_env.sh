#!/usr/bin/env bash
# One-shot probe for environment / dependencies / connectivity / model. Run inside your conda env.
set -uo pipefail

PY=${PYTHON:-python}
echo "================ 1. Python environment ================"
which "$PY"
"$PY" --version

echo
echo "================ 2. Dependency check ================"
"$PY" - <<'PYEOF'
mods = [
    ("torch", "torch"),
    ("sentence_transformers", "sentence-transformers"),
    ("sklearn", "scikit-learn"),
    ("numpy", "numpy"),
    ("openai", "openai"),
    ("tqdm", "tqdm"),
]
missing = []
for imp, pip_name in mods:
    try:
        m = __import__(imp)
        ver = getattr(m, "__version__", "?")
        print(f"  [OK]   {pip_name:<24} {ver}")
    except Exception:
        print(f"  [MISS] {pip_name:<24} (not installed)")
        missing.append(pip_name)
try:
    import torch
    print(f"  cuda available: {torch.cuda.is_available()}")
except Exception:
    pass
if missing:
    print()
    print("  Install missing dependencies:")
    print("    pip install " + " ".join(f'"{m}"' for m in missing))
PYEOF

echo
echo "================ 3. Connectivity probe (for downloading bge-m3) ================"
for host in huggingface.co hf-mirror.com modelscope.cn; do
    code=$(curl -sI --connect-timeout 6 "https://$host" 2>/dev/null | head -1)
    if [ -n "$code" ]; then
        echo "  [REACH] $host -> $code"
    else
        echo "  [DOWN]  $host  (unreachable)"
    fi
done

echo
echo "================ 4. Embedding model local path check ================"
# EMBED_MODEL can override; defaults to models/bge-m3 next to this script
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EMBED_MODEL=${EMBED_MODEL:-"$SCRIPT_DIR/models/bge-m3"}
echo "  EMBED_MODEL = $EMBED_MODEL"
if [ -d "$EMBED_MODEL" ]; then
    if [ -f "$EMBED_MODEL/model.safetensors" ] || [ -f "$EMBED_MODEL/pytorch_model.bin" ]; then
        echo "  [OK] model weights are ready"
        ls -lh "$EMBED_MODEL" | grep -iE "safetensors|bin|config.json|sentence" || true
    else
        echo "  [WARN] directory exists but no weight file found (model.safetensors / pytorch_model.bin)"
    fi
else
    echo "  [MISS] model directory does not exist. Get bge-m3 via any of the following:"
    echo "    A) download it on a networked machine and copy it to this path"
    echo "    B) mirror:  HF_ENDPOINT=https://hf-mirror.com huggingface-cli download BAAI/bge-m3 --local-dir \"$EMBED_MODEL\""
    echo "    C) ModelScope: pip install modelscope && modelscope download --model BAAI/bge-m3 --local_dir \"$EMBED_MODEL\""
fi

echo
echo "================ 5. Judge endpoint (OpenAI-compatible) check ================"
if [ -n "${JUDGE_BASE_URL:-}" ] && [ -n "${JUDGE_API_KEY:-}" ]; then
    echo "  [SET] JUDGE_BASE_URL=$JUDGE_BASE_URL  JUDGE_MODEL=${JUDGE_MODEL:-<unset>}"
else
    echo "  [SKIP] JUDGE_BASE_URL/JUDGE_API_KEY not configured; judge metrics will be skipped (cosine only)."
    echo "         Once configured, set them in config.py or export the environment variables and rerun to add the judge results."
fi
echo
echo "================ Done ================"
