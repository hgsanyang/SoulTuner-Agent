#!/usr/bin/env bash
# Evaluate one Planner V4 adapter with a reproducible, fail-closed inference path.
#
# Required for EVAL_SCOPE=all:
#   ADAPTER, REGRESSION_FILE, FROZEN_SEALED_FILE, PROMPT_REFERENCE,
#   MANIFEST_FILE, EVAL_OUTPUT_DIR
# Required for EVAL_SCOPE=regression:
#   ADAPTER, REGRESSION_FILE, EVAL_OUTPUT_DIR
#
# The frozen sealed file is never rewritten.  The release score uses a derived
# copy whose system message is taken from PROMPT_REFERENCE (the frozen training
# split); the original short-prompt split is evaluated separately as a
# non-gating prompt-contract stress test.
set -euo pipefail

: "${ADAPTER:?set ADAPTER to the checkpoint or adapter directory}"
: "${REGRESSION_FILE:?set REGRESSION_FILE to the frozen regression JSONL}"
: "${EVAL_OUTPUT_DIR:?set EVAL_OUTPUT_DIR to a fresh evaluation directory}"

EVAL_SCOPE="${EVAL_SCOPE:-all}"
SEED="${SEED:-42}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
# Deliberately not caller-overridable: this is a release evaluator, not a
# sampling harness.  In ms-swift 4.4.2 temperature=0 means do_sample=False.
TEMPERATURE="0"

case "$EVAL_SCOPE" in
  all|regression) ;;
  *) echo "FAIL: EVAL_SCOPE must be all or regression (got '$EVAL_SCOPE')"; exit 4 ;;
esac

for required in "$ADAPTER" "$REGRESSION_FILE"; do
  [ -e "$required" ] || { echo "FAIL: missing evaluation input: $required"; exit 4; }
done

if [ -e "$EVAL_OUTPUT_DIR" ]; then
  [ -d "$EVAL_OUTPUT_DIR" ] || {
    echo "FAIL: EVAL_OUTPUT_DIR exists and is not a directory: $EVAL_OUTPUT_DIR"
    exit 4
  }
  if find "$EVAL_OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
    echo "FAIL: EVAL_OUTPUT_DIR must be new or empty: $EVAL_OUTPUT_DIR"
    exit 8
  fi
else
  mkdir -p "$EVAL_OUTPUT_DIR"
fi

# ms-swift 4.4.2 has no --do_sample flag.  Its generation arguments define
# temperature=0 as greedy/non-sampling, so lock temperature and seed explicitly
# instead of relying on values inherited from an adapter's args.json.
python -m data.sft.check_swift_flags --subcommand infer \
  --json "$EVAL_OUTPUT_DIR/swift_flags_infer.json" \
  --flags adapters val_dataset enable_thinking max_new_tokens temperature seed result_path

DECODE_RECORD="$(python - "$SEED" "$MAX_NEW_TOKENS" "$TEMPERATURE" <<'PY'
import json
import sys

seed, max_new_tokens, temperature = sys.argv[1:4]
print(json.dumps({
    "do_sample": False,
    "temperature": float(temperature),
    "seed": int(seed),
    "max_new_tokens": int(max_new_tokens),
    "enable_thinking": False,
}, separators=(",", ":")))
PY
)"

run_infer() {
  local label="$1"
  local input_file="$2"
  local score_mode="$3"
  local pred="$EVAL_OUTPUT_DIR/${label}_predictions.jsonl"

  python -m data.sft.check_infer_contract --reserve "$pred" \
    --json "$EVAL_OUTPUT_DIR/${label}_reserve.json"

  swift infer \
    --adapters "$ADAPTER" \
    --val_dataset "$input_file" \
    --enable_thinking false \
    --temperature "$TEMPERATURE" \
    --seed "$SEED" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --result_path "$pred"

  # This is deliberately before thinking/schema checks and scoring.  A 452-row
  # file for 226 inputs is not two convenient samples; it is one invalid run.
  python -m data.sft.check_infer_contract \
    --input "$input_file" \
    --pred "$pred" \
    --record "$DECODE_RECORD" \
    --json "$EVAL_OUTPUT_DIR/${label}_infer_contract.json"

  python -m data.sft.verify_infer_output \
    --pred "$pred" \
    --schema planner_v3 \
    --json "$EVAL_OUTPUT_DIR/${label}_thinking_check.json"

  local score_args=(
    --eval "$input_file"
    --pred "$pred"
    --json "$EVAL_OUTPUT_DIR/${label}_score.json"
  )
  if [ "$score_mode" = "stress" ]; then
    # An invalid schema is the observed result of the short-prompt stress case;
    # report it without letting that non-canonical condition become a release gate.
    score_args+=(--no-strict)
  fi
  python -m data.sft.score_student "${score_args[@]}"
}

if [ "$EVAL_SCOPE" = "all" ]; then
  # Validate and derive every CPU-side input before loading an adapter onto the
  # GPU.  A missing manifest must not be discovered after regression inference.
  : "${FROZEN_SEALED_FILE:?set FROZEN_SEALED_FILE for EVAL_SCOPE=all}"
  : "${PROMPT_REFERENCE:?set PROMPT_REFERENCE to the frozen training split}"
  : "${MANIFEST_FILE:?set MANIFEST_FILE for EVAL_SCOPE=all}"
  for required in "$FROZEN_SEALED_FILE" "$PROMPT_REFERENCE" "$MANIFEST_FILE"; do
    [ -s "$required" ] || { echo "FAIL: missing evaluation input: $required"; exit 4; }
  done

  CANONICAL_SEALED="$EVAL_OUTPUT_DIR/sealed_canonical_prompt.jsonl"
  python -m data.sft.derive_canonical_prompt_eval \
    --source "$FROZEN_SEALED_FILE" \
    --reference "$PROMPT_REFERENCE" \
    --target "$CANONICAL_SEALED" \
    --json "$EVAL_OUTPUT_DIR/sealed_canonical_derivation.json"
fi

run_infer regression "$REGRESSION_FILE" release

if [ "$EVAL_SCOPE" = "regression" ]; then
  echo "EVAL OK: regression-only characterization completed"
  exit 0
fi

run_infer sealed_canonical "$CANONICAL_SEALED" release
run_infer sealed_prompt_contract_stress "$FROZEN_SEALED_FILE" stress

# Run this last so a quality failure still leaves the complete canonical and
# stress evidence on disk.  Exit 14 means the model failed a release gate; it
# does not mean the training or persistence step was lost.
python -m data.sft.check_planner_release \
  --manifest "$MANIFEST_FILE" \
  --regression "$EVAL_OUTPUT_DIR/regression_score.json" \
  --sealed "$EVAL_OUTPUT_DIR/sealed_canonical_score.json" \
  --json "$EVAL_OUTPUT_DIR/release_gate.json"
