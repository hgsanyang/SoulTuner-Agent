#!/usr/bin/env bash
set -euo pipefail

base_repo="${SOULTUNER_BASE_REPO:-Qwen/Qwen3.6-35B-A3B}"
: "${SOULTUNER_ADAPTER_REPO:?Set SOULTUNER_ADAPTER_REPO to the public PEFT adapter repository}"

model_root="${SOULTUNER_MODEL_ROOT:-./models}"
base_dir="${SOULTUNER_BASE_MODEL:-${model_root}/qwen3.6-35b-a3b}"
adapter_dir="${SOULTUNER_ADAPTER:-${model_root}/soultuner-v4.2-adapter}"
base_revision="${SOULTUNER_BASE_REVISION:-main}"
adapter_revision="${SOULTUNER_ADAPTER_REVISION:-main}"

command -v hf >/dev/null 2>&1 || {
  echo 'Hugging Face CLI is required: python -m pip install -U "huggingface_hub[hf_xet]"' >&2
  exit 2
}

mkdir -p "$model_root"
hf download "$base_repo" --revision "$base_revision" --local-dir "$base_dir"
hf download "$SOULTUNER_ADAPTER_REPO" \
  --revision "$adapter_revision" --local-dir "$adapter_dir"

test -f "$base_dir/config.json"
test -f "$adapter_dir/adapter_config.json"
test -f "$adapter_dir/adapter_model.safetensors"
if [[ -f "$adapter_dir/SHA256SUMS" ]]; then
  (cd "$adapter_dir" && sha256sum --check SHA256SUMS)
fi

printf 'SOULTUNER_BASE_MODEL=%s\nSOULTUNER_ADAPTER=%s\n' "$base_dir" "$adapter_dir"
