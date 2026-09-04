#!/usr/bin/env bash
set -euo pipefail

if [[ "${SOULTUNER_ENABLE_FULL_SPACE:-0}" != "1" ]]; then
  echo "Full Space entrypoint is gated; set SOULTUNER_ENABLE_FULL_SPACE=1 in a copied test Space." >&2
  exit 2
fi

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE_ROOT="${SOULTUNER_WORKSPACE_ROOT:-/mnt/workspace/soultuner}"
WEB_SERVER="${ROOT}/web/.next/standalone/server.js"

test -d /mnt/workspace || { echo "/mnt/workspace is unavailable" >&2; exit 3; }
test -w /mnt/workspace || { echo "/mnt/workspace is not writable" >&2; exit 3; }
test -f "${WEB_SERVER}" || {
  echo "Next.js standalone artifact missing; run deploy/modelscope_full/build_frontend.sh during image build." >&2
  exit 4
}

: "${NEO4J_URI:?Set a private external NEO4J_URI; embedded Neo4j is not enabled in Studio}"
: "${NEO4J_USER:?Set NEO4J_USER}"
: "${NEO4J_PASSWORD:?Set NEO4J_PASSWORD as a secret}"

mkdir -p \
  "${WORKSPACE_ROOT}/model_cache" \
  "${WORKSPACE_ROOT}/hf" \
  "${WORKSPACE_ROOT}/data" \
  "${WORKSPACE_ROOT}/logs"

export SOULTUNER_REQUIRE_ROCM=1
export SOULTUNER_MODEL_PROFILE="${SOULTUNER_MODEL_PROFILE:-soultuner-v4.2-35b}"
export SOULTUNER_DUAL_ROLE_MODELS="${SOULTUNER_DUAL_ROLE_MODELS:-1}"
export SOULTUNER_CHAT_MODEL="${SOULTUNER_CHAT_MODEL:-qwen3.6-35b-a3b}"
export SOULTUNER_PLANNER_MODEL="${SOULTUNER_PLANNER_MODEL:-soultuner-v4.2-35b}"
export SOULTUNER_MODEL_CACHE="${SOULTUNER_MODEL_CACHE:-${WORKSPACE_ROOT}/model_cache}"
export HF_HOME="${HF_HOME:-${WORKSPACE_ROOT}/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export MUSIC_DATA_ROOT="${MUSIC_DATA_ROOT:-${WORKSPACE_ROOT}/data}"
export MUSIC_AUDIO_DATA_DIR="${MUSIC_AUDIO_DATA_DIR:-${MUSIC_DATA_ROOT}/processed_audio/audio}"
export SOULTUNER_OPEN_AUDIO_DIR="${SOULTUNER_OPEN_AUDIO_DIR:-${MUSIC_DATA_ROOT}/mtg_sample}"
export SOULTUNER_CATALOG_PATH="${SOULTUNER_CATALOG_PATH:-${SOULTUNER_OPEN_AUDIO_DIR}/catalog.jsonl}"
export SOULTUNER_AUDIO_ROOT="${SOULTUNER_AUDIO_ROOT:-${SOULTUNER_OPEN_AUDIO_DIR}/audio}"
export MTG_AUDIO_DIR="${MTG_AUDIO_DIR:-${SOULTUNER_AUDIO_ROOT}}"
export MUSIC_FEEDBACK_DIR="${MUSIC_FEEDBACK_DIR:-${MUSIC_DATA_ROOT}/feedback}"
export MUSIC_INGEST_QUEUE_DIR="${MUSIC_INGEST_QUEUE_DIR:-${MUSIC_DATA_ROOT}/ingest_queue}"
export SOULTUNER_PLANNER_BASE_URL="${SOULTUNER_PLANNER_BASE_URL:-http://127.0.0.1:8000/v1}"
export VLLM_BASE_URL="${VLLM_BASE_URL:-${SOULTUNER_PLANNER_BASE_URL}}"
export LLM_DEFAULT_PROVIDER="${LLM_DEFAULT_PROVIDER:-vllm}"
export LLM_DEFAULT_MODEL="${LLM_DEFAULT_MODEL:-${SOULTUNER_CHAT_MODEL}}"
export CONVERSATION_LLM_PROVIDER="${CONVERSATION_LLM_PROVIDER:-vllm}"
export CONVERSATION_LLM_MODEL="${CONVERSATION_LLM_MODEL:-${SOULTUNER_CHAT_MODEL}}"
export EXPLAIN_LLM_PROVIDER="${EXPLAIN_LLM_PROVIDER:-${CONVERSATION_LLM_PROVIDER}}"
export EXPLAIN_LLM_MODEL="${EXPLAIN_LLM_MODEL:-${CONVERSATION_LLM_MODEL}}"
export INTENT_LLM_PROVIDER="${INTENT_LLM_PROVIDER:-vllm}"
export INTENT_LLM_MODEL="${INTENT_LLM_MODEL:-${SOULTUNER_PLANNER_MODEL}}"
export BACKEND_INTERNAL_URL="http://127.0.0.1:8501"
export NEXT_PUBLIC_API_URL=""

planner_log="${WORKSPACE_ROOT}/logs/planner.log"
backend_log="${WORKSPACE_ROOT}/logs/backend.log"
open_audio_log="${WORKSPACE_ROOT}/logs/open-audio.log"

prepare_open_audio() {
  local dataset_id="${SOULTUNER_OPEN_AUDIO_DATASET_ID:-hgsanyang/SoulTuner-Open-Audio-Demo}"
  local revision="${SOULTUNER_OPEN_AUDIO_REVISION:-4aa4fb3feeaabcb16b711a08930afd97434463b6}"
  mkdir -p "${SOULTUNER_OPEN_AUDIO_DIR}"
  modelscope download "${dataset_id}" --repo-type dataset \
    --revision "${revision}" --local-dir "${SOULTUNER_OPEN_AUDIO_DIR}" --max-workers 4
  python - <<'PY'
import hashlib
import json
import os
from pathlib import Path, PurePosixPath

root = Path(os.environ["SOULTUNER_AUDIO_ROOT"]).resolve()
catalog = Path(os.environ["SOULTUNER_CATALOG_PATH"]).resolve()
rows = [json.loads(line) for line in catalog.read_text(encoding="utf-8").splitlines() if line.strip()]
if not rows:
    raise SystemExit("open-audio catalog is empty")
for row in rows:
    relpath = str(PurePosixPath(str(row.get("audio_relpath") or "")))
    candidate = (root / relpath).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise SystemExit(f"unsafe open-audio path: {relpath}") from exc
    expected = str(row.get("audio_sha256") or "")
    if not candidate.is_file() or not expected or not str(row.get("license_url") or "").startswith("http"):
        raise SystemExit(f"incomplete open-audio provenance: {row.get('song_id')}")
    actual = hashlib.sha256(candidate.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"open-audio SHA-256 mismatch: {row.get('song_id')}")
print(f"SoulTuner open-audio catalog ready: {len(rows)} tracks", flush=True)
PY
}

if [[ "${SOULTUNER_ENABLE_OPEN_AUDIO:-1}" == "1" ]]; then
  prepare_open_audio >"${open_audio_log}" 2>&1
  # Keep the 35B GPU free while the first five MuQ/M2D/OMAR vectors are built.
  # Later starts are a cheap read-only readiness check against external Neo4j.
  (cd "${ROOT}" && CUDA_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES="" \
    python -m deploy.modelscope_full.bootstrap_open_audio \
      --cache-dir "${SOULTUNER_OPEN_AUDIO_DIR}" \
      --manifest "${SOULTUNER_CATALOG_PATH}") >>"${open_audio_log}" 2>&1
fi

bash "${ROOT}/deploy/modelscope_space/start_amd_35b.sh" >"${planner_log}" 2>&1 &
planner_pid=$!

cleanup() {
  kill "${backend_pid:-}" "${planner_pid:-}" 2>/dev/null || true
  wait "${backend_pid:-}" "${planner_pid:-}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

export SOULTUNER_ENDPOINT_PID="${planner_pid}"
python - <<'PY'
import json
import os
import time
import urllib.request

url = os.environ["SOULTUNER_PLANNER_BASE_URL"].rstrip("/") + "/models"
pid = int(os.environ["SOULTUNER_ENDPOINT_PID"])
required = {os.environ["SOULTUNER_PLANNER_MODEL"]}
if os.getenv("SOULTUNER_DUAL_ROLE_MODELS", "0") == "1":
    required.add(os.environ["SOULTUNER_CHAT_MODEL"])
deadline = time.monotonic() + float(os.getenv("SOULTUNER_ENDPOINT_START_TIMEOUT", "1800"))
while time.monotonic() < deadline:
    try:
        os.kill(pid, 0)
        with urllib.request.urlopen(url, timeout=5) as response:
            payload = json.loads(response.read())
            models = {
                str(item.get("id"))
                for item in payload.get("data", [])
                if isinstance(item, dict) and item.get("id")
            }
            if required <= models:
                break
    except Exception:
        time.sleep(5)
else:
    raise SystemExit(f"35B endpoint did not expose required roles: {sorted(required)}")
PY

cd "${ROOT}"
python -m uvicorn api.server:app --host 127.0.0.1 --port 8501 >"${backend_log}" 2>&1 &
backend_pid=$!
export SOULTUNER_BACKEND_PID="${backend_pid}"

python - <<'PY'
import os
import time
import urllib.request

pid = int(os.environ["SOULTUNER_BACKEND_PID"])
deadline = time.monotonic() + 180
while time.monotonic() < deadline:
    try:
        os.kill(pid, 0)
        with urllib.request.urlopen("http://127.0.0.1:8501/health", timeout=3) as response:
            if response.status == 200:
                break
    except Exception:
        time.sleep(2)
else:
    raise SystemExit("FastAPI backend did not become healthy")
PY

cd "${ROOT}/web/.next/standalone"
export HOSTNAME=0.0.0.0
export PORT="${PORT:-7860}"
exec node server.js
