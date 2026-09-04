#!/usr/bin/env bash
# Optional ModelScope-specific downloader for the platform deployment profile.
set -euo pipefail

base_repo="${SOULTUNER_BASE_REPO:-Qwen/Qwen3.6-35B-A3B}"
adapter_repo="${SOULTUNER_ADAPTER_REPO:-hgsanyang/SoulTuner-Planner-V4.2-35B-LoRA}"
model_root="${SOULTUNER_MODEL_ROOT:-$PWD/models}"
base_revision="${SOULTUNER_BASE_REVISION:-913c459c5c83fa016a0e54a52e5b95f6c894e0fe}"
adapter_revision="${SOULTUNER_ADAPTER_REVISION:-f3685197007a5d7e41a20c484865f144af101804}"

base_dir="$model_root/qwen3.6-35b-a3b"
adapter_dir="$model_root/soultuner-v4.2-adapter"

command -v modelscope >/dev/null 2>&1 || {
  echo "modelscope CLI is required: python -m pip install -U modelscope" >&2
  exit 2
}

mkdir -p "$base_dir" "$adapter_dir"

modelscope download "$base_repo" \
  --repo-type model \
  --revision "$base_revision" \
  --local-dir "$base_dir"

modelscope download "$adapter_repo" \
  --repo-type model \
  --revision "$adapter_revision" \
  --local-dir "$adapter_dir"

(
  cd "$adapter_dir"
  sha256sum --check SHA256SUMS
)

printf 'SOULTUNER_BASE_MODEL=%s\n' "$base_dir"
printf 'SOULTUNER_ADAPTER=%s\n' "$adapter_dir"
