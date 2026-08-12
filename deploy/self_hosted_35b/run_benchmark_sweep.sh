#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/benchmark-results}"
CONCURRENCY_LEVELS="${CONCURRENCY_LEVELS:-1 4 8 16}"
REPEAT="${REPEAT:-3}"
WARMUP="${WARMUP:-1}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-300}"
RUN_LABEL="${RUN_LABEL:-vllm}"
ENDPOINT="${SOULTUNER_BENCHMARK_ENDPOINT:-http://127.0.0.1:8000/v1/chat/completions}"
MODEL="${SOULTUNER_BENCHMARK_MODEL:-soultuner-planner-v4.2-35b}"

mkdir -p "${OUTPUT_DIR}"
monitor_pid=""

cleanup() {
  if [[ -n "${monitor_pid}" ]] && kill -0 "${monitor_pid}" 2>/dev/null; then
    kill "${monitor_pid}" 2>/dev/null || true
    wait "${monitor_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if command -v rocm-smi >/dev/null 2>&1; then
  (
    while :; do
      printf '%s ' "$(date -Is)"
      rocm-smi --showuse --showmemuse --csv 2>/dev/null | tail -n +2
      sleep 2
    done
  ) >"${OUTPUT_DIR}/${RUN_LABEL}-gpu.csv" 2>&1 &
  monitor_pid="$!"
fi

for concurrency in ${CONCURRENCY_LEVELS}; do
  output_json="${OUTPUT_DIR}/${RUN_LABEL}-c${concurrency}.json"
  if [[ -s "${output_json}" ]]; then
    echo "SKIP existing ${output_json}"
    continue
  fi
  echo "BENCH_START concurrency=${concurrency} $(date -Is)"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/benchmark_endpoint.py" \
    --endpoint "${ENDPOINT}" \
    --model "${MODEL}" \
    --warmup "${WARMUP}" \
    --repeat "${REPEAT}" \
    --concurrency "${concurrency}" \
    --timeout "${TIMEOUT_SECONDS}" \
    --json "${output_json}" \
    >"${OUTPUT_DIR}/${RUN_LABEL}-c${concurrency}.stdout"
  echo "BENCH_DONE concurrency=${concurrency} $(date -Is)"
done

cleanup
monitor_pid=""
sha256sum "${OUTPUT_DIR}/${RUN_LABEL}"-c*.json \
  "${OUTPUT_DIR}/${RUN_LABEL}"-gpu.csv 2>/dev/null \
  >"${OUTPUT_DIR}/${RUN_LABEL}-results.sha256" || true
