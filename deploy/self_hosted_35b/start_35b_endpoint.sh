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

host="${SOULTUNER_SERVE_HOST:-0.0.0.0}"
port="${SOULTUNER_SERVE_PORT:-8000}"
served_name="${SOULTUNER_SERVED_MODEL:-soultuner-planner-v4.2-35b}"

args=(
  deploy
  --model "$SOULTUNER_BASE_MODEL"
  --adapters "$SOULTUNER_ADAPTER"
  --host "$host"
  --port "$port"
  --served_model_name "$served_name"
  --enable_thinking false
  --temperature 0
  --max_new_tokens 1024
)

if [[ -n "${SOULTUNER_INFER_BACKEND:-}" ]]; then
  args+=(--infer_backend "$SOULTUNER_INFER_BACKEND")
fi
if [[ -n "${SOULTUNER_SERVE_API_KEY:-}" ]]; then
  args+=(--api_key "$SOULTUNER_SERVE_API_KEY")
fi

echo "Starting guarded candidate endpoint on ${host}:${port} as ${served_name}"
exec swift "${args[@]}"
