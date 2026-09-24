#!/usr/bin/env bash
set -euo pipefail

# ===== Optional switches =====
USE_MEGATRON=${USE_MEGATRON:-0}
USE_SGLANG=${USE_SGLANG:-0}
USE_FLASH_ATTN=${USE_FLASH_ATTN:-0}
USE_FLASHINFER=${USE_FLASHINFER:-0}
REPO_DIR=${REPO_DIR:-/root/yzh/verl}

export MAX_JOBS=${MAX_JOBS:-32}

echo "[1/8] Upgrade pip tooling"
python -m pip install -U pip setuptools wheel

echo "[2/8] Clean conflicting packages (safe if not installed)"
pip uninstall -y sglang sgl-kernel flashinfer-python flashinfer \
  opentelemetry-api opentelemetry-sdk \
  opentelemetry-exporter-otlp-proto-grpc opentelemetry-exporter-otlp-proto-http || true

echo "[3/8] Install torch stack + vLLM"
pip install --no-cache-dir \
  "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0" "torchdata" \
  "vllm==0.8.5.post1"

echo "[4/8] Install core training dependencies"
pip install --no-cache-dir \
  "transformers[hf_xet]>=4.51.0,<5" \
  accelerate datasets peft hf-transfer \
  "numpy==1.26.4" "pyarrow>=19.0.0" pandas \
  "ray[default]==2.49.2" \
  codetiming hydra-core pylatexenc wandb dill pybind11 \
  liger-kernel math_verify qwen-vl-utils \
  "tensordict>=0.8.0,<=0.9.1,!=0.9.0" \
  pytest py-spy pyext pre-commit ruff \
  "nvidia-ml-py>=12.560.30" \
  "fastapi[standard]>=0.115.0" \
  "optree>=0.13.0" \
  "pydantic>=2.9,<3" \
  "grpcio>=1.62.1,<2"

echo "[5/8] Pin telemetry to match vLLM"
pip install --no-cache-dir \
  "opentelemetry-api>=1.26,<1.27" \
  "opentelemetry-sdk>=1.26,<1.27" \
  "opentelemetry-exporter-otlp-proto-grpc>=1.26,<1.27" \
  "opentelemetry-exporter-otlp-proto-http>=1.26,<1.27"

echo "[6/8] Optional: FlashAttention"
if [ "$USE_FLASH_ATTN" -eq 1 ]; then
  pip install --no-cache-dir \
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
fi

echo "[7/8] Optional: sglang / flashinfer / megatron"
if [ "$USE_SGLANG" -eq 1 ]; then
  # 不使用 [all]，避免默认强拉 flashinfer 导致卡下载
  pip install --no-cache-dir "sglang==0.4.6.post1" "torch-memory-saver"
fi

if [ "$USE_FLASHINFER" -eq 1 ]; then
  pip install --no-cache-dir \
    "https://github.com/flashinfer-ai/flashinfer/releases/download/v0.2.2.post1/flashinfer_python-0.2.2.post1+cu124torch2.6-cp38-abi3-linux_x86_64.whl"
fi

if [ "$USE_MEGATRON" -eq 1 ]; then
  NVTE_FRAMEWORK=pytorch pip install --no-deps \
    "git+https://github.com/NVIDIA/TransformerEngine.git@v2.2.1"
  pip install --no-deps \
    "git+https://github.com/NVIDIA/Megatron-LM.git@core_v0.12.2"
fi

echo "[8/8] Install local verl + sanity checks"
pip install -e "${REPO_DIR}"
python - <<'PY'
import ray, transformers
print("ray:", ray.__version__)
print("transformers:", transformers.__version__)
print("has AutoModelForVision2Seq:", hasattr(transformers, "AutoModelForVision2Seq"))
PY
pip check || true

echo "Done."
echo "Runtime建议：export RAY_TMPDIR=/dev/shm/ray; export RAY_DISABLE_DASHBOARD=1"
