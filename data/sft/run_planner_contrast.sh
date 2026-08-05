#!/usr/bin/env bash
"${BASH_VERSION:?run this script with bash}"
set -euo pipefail

# Fair 9B/35B comparison: same frozen data and seed, isolated output trees.
# Run sequentially. Concurrent training would change memory pressure and
# throughput, so the comparison would no longer isolate model quality.
: "${MANIFEST_FILE:?set MANIFEST_FILE to the frozen V4 manifest}"
: "${TRAIN_FILE:?set TRAIN_FILE to the manifest train split}"
: "${VAL_FILE:?set VAL_FILE to the manifest regression split}"
: "${SEALED_FILE:?set SEALED_FILE to the manifest sealed split}"

SEED="${SEED:-42}"
RUN_FULL="${RUN_FULL:-0}"
BASE_RUN_ID="${BASE_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./output}"
MODELS=("Qwen/Qwen3.5-9B" "Qwen/Qwen3.6-35B-A3B")
OUTPUTS=()

for model in "${MODELS[@]}"; do
  tag="$(printf '%s' "$model" | tr -c 'A-Za-z0-9._-' '_')"
  run_id="${BASE_RUN_ID}-${tag}"
  output_dir="${OUTPUT_ROOT}/planner-student-${tag}-lora/${run_id}"
  OUTPUTS+=("$output_dir")
  echo "== contrast model: $model =="
  MODEL="$model" \
  RUN_ID="$run_id" \
  PREFLIGHT_RUN_ID="$run_id" \
  OUTPUT_DIR="$output_dir" \
  SEED="$SEED" \
  RUN_FULL="$RUN_FULL" \
  MANIFEST_FILE="$MANIFEST_FILE" \
  TRAIN_FILE="$TRAIN_FILE" \
  VAL_FILE="$VAL_FILE" \
  SEALED_FILE="$SEALED_FILE" \
  bash data/sft/train_planner_student.sh
done

if [ "$RUN_FULL" = "1" ]; then
  for output_dir in "${OUTPUTS[@]}"; do
    python -m data.sft.check_planner_release \
      --manifest "$MANIFEST_FILE" \
      --regression "${output_dir}/eval_score.json" \
      --sealed "${output_dir}/sealed_score.json" \
      --json "${output_dir}/release_gate.json"
  done
  read -r CONTRAST_TOLERANCE SPLIT_GAP_TOLERANCE < <(
    python - "$MANIFEST_FILE" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as handle:
    gates = json.load(handle)["sealed_policy"]["release_gates"]
print(
    float(gates["per_kind_max_regression_pp"]) / 100.0,
    float(gates["sealed_vs_regression_max_gap_pp"]) / 100.0,
)
PY
  )
  python -m data.sft.compare_planner_scores \
    --baseline "${OUTPUTS[0]}/eval_score.json" \
    --candidate "${OUTPUTS[1]}/eval_score.json" \
    --tolerance "$CONTRAST_TOLERANCE" \
    --json "${OUTPUT_ROOT}/contrast-regression-${BASE_RUN_ID}.json"
  python -m data.sft.compare_planner_scores \
    --baseline "${OUTPUTS[0]}/sealed_score.json" \
    --candidate "${OUTPUTS[1]}/sealed_score.json" \
    --candidate-regression "${OUTPUTS[1]}/eval_score.json" \
    --tolerance "$CONTRAST_TOLERANCE" \
    --split-gap-tolerance "$SPLIT_GAP_TOLERANCE" \
    --json "${OUTPUT_ROOT}/contrast-sealed-${BASE_RUN_ID}.json"
fi
