#!/usr/bin/env bash
# Start the verified best V4.2 adapter as an OpenAI-compatible *candidate* endpoint.
set -euo pipefail

: "${SOULTUNER_BASE_MODEL:?Set SOULTUNER_BASE_MODEL to the complete 35B base model directory}"
: "${SOULTUNER_ADAPTER:?Set SOULTUNER_ADAPTER to the checkpoint-450 adapter directory}"

if [[ ! -d "$SOULTUNER_BASE_MODEL" ]]; then
  echo "base model directory not found: $SOULTUNER_BASE_MODEL" >&2
  exit 2
fi
if [[ ! -f "$SOULTUNER_ADAPTER/adapter_model.safetensors" ]]; then
  echo "adapter_model.safetensors not found under: $SOULTUNER_ADAPTER" >&2
  exit 3
fi
if ! command -v swift >/dev/null 2>&1; then
  echo "ms-swift CLI not found in PATH" >&2
  exit 4
fi

host="${SOULTUNER_SERVE_HOST:-127.0.0.1}"
port="${SOULTUNER_SERVE_PORT:-8000}"
served_name="${SOULTUNER_SERVED_MODEL:-soultuner-planner-v4.2-35b}"
backend="${SOULTUNER_INFER_BACKEND:-vllm}"

args=(
  deploy
  --model "$SOULTUNER_BASE_MODEL"
  --adapters "$SOULTUNER_ADAPTER"
  --host "$host"
  --port "$port"
  --served_model_name "$served_name"
  --infer_backend "$backend"
  --enable_thinking false
  --temperature 0
  --max_new_tokens "${SOULTUNER_MAX_NEW_TOKENS:-1024}"
)

if [[ "$backend" == "vllm" ]]; then
  deploy_help="$(swift deploy --help 2>&1)"
  append_supported_flag() {
    local preferred="$1" legacy="$2" value="$3"
    if grep -q -- "--${preferred}" <<<"$deploy_help"; then
      args+=("--${preferred}" "$value")
    elif grep -q -- "--${legacy}" <<<"$deploy_help"; then
      args+=("--${legacy}" "$value")
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
  args+=(--api_key "$SOULTUNER_SERVE_API_KEY")
fi

if [[ "$host" == "0.0.0.0" && -z "${SOULTUNER_SERVE_API_KEY:-}" ]]; then
  echo "Refusing an unauthenticated endpoint on 0.0.0.0; set SOULTUNER_SERVE_API_KEY" >&2
  exit 6
fi

echo "Starting guarded candidate endpoint on ${host}:${port} as ${served_name} (backend=${backend})"
exec swift "${args[@]}"
