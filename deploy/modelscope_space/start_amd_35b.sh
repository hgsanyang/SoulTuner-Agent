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
BASE_MODEL_DIR="${SOULTUNER_BASE_MODEL_DIR:-${CACHE_DIR}/qwen3.6-35b-a3b}"
ADAPTER_MODEL_DIR="${SOULTUNER_ADAPTER_DIR:-${CACHE_DIR}/soultuner_adapter}"

mkdir -p "${CACHE_DIR}"

# ModelScope resumes incomplete downloads and reuses already cached files. Running
# both commands on every boot also catches a directory that contains config.json
# but was interrupted before all weight shards arrived.
modelscope download "${BASE_MODEL_ID}" --repo-type model \
  --revision "${SOULTUNER_BASE_MODEL_REVISION:-master}" --local-dir "${BASE_MODEL_DIR}"
modelscope download "${ADAPTER_MODEL_ID}" --repo-type model \
  --revision "${SOULTUNER_ADAPTER_REVISION:-master}" --local-dir "${ADAPTER_MODEL_DIR}"

test -f "${BASE_MODEL_DIR}/config.json"
test -f "${ADAPTER_MODEL_DIR}/adapter_config.json"
test -f "${ADAPTER_MODEL_DIR}/adapter_model.safetensors"
if [[ -f "${ADAPTER_MODEL_DIR}/SHA256SUMS" ]]; then
  (cd "${ADAPTER_MODEL_DIR}" && sha256sum --check SHA256SUMS)
fi

export SOULTUNER_ADAPTER_DIR="${ADAPTER_MODEL_DIR}"
SOULTUNER_REQUIRE_ROCM="${SOULTUNER_REQUIRE_ROCM:-1}" \
  python amd_readiness.py --skip-endpoint

backend="${SOULTUNER_INFER_BACKEND:-vllm}"
args=(
  deploy
  --model "${BASE_MODEL_DIR}"
  --adapters "${ADAPTER_MODEL_DIR}"
  --served_model_name "${SERVED_MODEL_NAME}"
  --host 0.0.0.0
  --port "${PORT}"
  --infer_backend "${backend}"
  --enable_thinking false
  --temperature 0
  --max_new_tokens "${SOULTUNER_MAX_NEW_TOKENS:-1024}"
)

# ms-swift renamed several vLLM flags after 3.7. Detect the installed CLI
# instead of pinning the deployment to one spelling.
if [[ "${backend}" == "vllm" ]]; then
  deploy_help="$(swift deploy --help 2>&1)"
  append_supported_flag() {
    local preferred="$1" legacy="$2" value="$3"
    if grep -q -- "--${preferred}" <<<"${deploy_help}"; then
      args+=("--${preferred}" "${value}")
    elif grep -q -- "--${legacy}" <<<"${deploy_help}"; then
      args+=("--${legacy}" "${value}")
    else
      echo "swift deploy does not support --${preferred} or --${legacy}" >&2
      exit 5
    fi
  }
  append_supported_flag vllm_max_model_len max_model_len \
    "${SOULTUNER_MAX_MODEL_LEN:-4096}"
  append_supported_flag vllm_gpu_memory_utilization gpu_memory_utilization \
    "${SOULTUNER_GPU_MEMORY_UTILIZATION:-0.90}"
  append_supported_flag vllm_max_num_seqs max_num_seqs \
    "${SOULTUNER_MAX_NUM_SEQS:-16}"
  append_supported_flag vllm_enable_prefix_caching enable_prefix_caching \
    "${SOULTUNER_ENABLE_PREFIX_CACHING:-true}"
fi

if [[ -n "${SOULTUNER_SERVE_API_KEY:-}" ]]; then
  args+=(--api_key "${SOULTUNER_SERVE_API_KEY}")
fi

exec swift "${args[@]}"
