#!/usr/bin/env bash
set -euo pipefail

# Run inside the AMD MI308X / ROCm image after installing requirements-amd.txt.
# Authentication, when needed for a private model, is read by the ModelScope SDK
# from the platform secret store. Never place tokens in this file.

BASE_MODEL_ID="${SOULTUNER_BASE_MODEL_ID:-Qwen/Qwen3.6-35B-A3B}"
ADAPTER_MODEL_ID="${SOULTUNER_ADAPTER_MODEL_ID:-hgsanyang/SoulTuner-Planner-V4.2-35B-LoRA}"
SERVED_MODEL_NAME="${SOULTUNER_PLANNER_MODEL:-soultuner-v4.2-35b}"
PORT="${SOULTUNER_PLANNER_PORT:-8000}"
CACHE_DIR="${SOULTUNER_MODEL_CACHE:-./model_cache}"

mkdir -p "${CACHE_DIR}"

BASE_MODEL_DIR="$(python -c 'from modelscope import snapshot_download; import os; print(snapshot_download(os.environ.get("SOULTUNER_BASE_MODEL_ID", "Qwen/Qwen3.6-35B-A3B"), cache_dir=os.environ.get("SOULTUNER_MODEL_CACHE", "./model_cache")))')"
ADAPTER_MODEL_DIR="$(python -c 'from modelscope import snapshot_download; import os; print(snapshot_download(os.environ.get("SOULTUNER_ADAPTER_MODEL_ID", "hgsanyang/SoulTuner-Planner-V4.2-35B-LoRA"), cache_dir=os.environ.get("SOULTUNER_MODEL_CACHE", "./model_cache")))')"

test -f "${ADAPTER_MODEL_DIR}/adapter_config.json"
test -f "${ADAPTER_MODEL_DIR}/adapter_model.safetensors"

python hardware.py

exec swift deploy \
  --model "${BASE_MODEL_DIR}" \
  --adapters "${ADAPTER_MODEL_DIR}" \
  --served_model_name "${SERVED_MODEL_NAME}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --max_model_len "${SOULTUNER_MAX_MODEL_LEN:-4096}" \
  --infer_backend "${SOULTUNER_INFER_BACKEND:-vllm}"
