#!/usr/bin/env bash
set -euo pipefail

# One-process entrypoint for the AMD MI308X Space profile.
# The CPU Space keeps using `python app.py`; this file is used only after the
# AMD_Dev organization grants access to an ROCm image and GPU resource.

export SOULTUNER_REQUIRE_ROCM="${SOULTUNER_REQUIRE_ROCM:-1}"
export SOULTUNER_MODEL_PROFILE="${SOULTUNER_MODEL_PROFILE:-soultuner-v4.2-35b}"
export SOULTUNER_PLANNER_BASE_URL="${SOULTUNER_PLANNER_BASE_URL:-http://127.0.0.1:8000/v1}"
export SOULTUNER_PLANNER_MODEL="${SOULTUNER_PLANNER_MODEL:-soultuner-v4.2-35b}"
default_cache_dir="./model_cache"
if [[ -d "/mnt/workspace" && -w "/mnt/workspace" ]]; then
  default_cache_dir="/mnt/workspace/soultuner/model_cache"
fi
export SOULTUNER_MODEL_CACHE="${SOULTUNER_MODEL_CACHE:-${default_cache_dir}}"
export SOULTUNER_ADAPTER_DIR="${SOULTUNER_ADAPTER_DIR:-${SOULTUNER_MODEL_CACHE}/soultuner_adapter}"
default_open_audio_dir="./open_audio"
if [[ -d "/mnt/workspace" && -w "/mnt/workspace" ]]; then
  default_open_audio_dir="/mnt/workspace/soultuner/open_audio"
fi
export SOULTUNER_OPEN_AUDIO_DIR="${SOULTUNER_OPEN_AUDIO_DIR:-${default_open_audio_dir}}"
export SOULTUNER_CATALOG_PATH="${SOULTUNER_CATALOG_PATH:-${SOULTUNER_OPEN_AUDIO_DIR}/catalog.jsonl}"
export SOULTUNER_AUDIO_ROOT="${SOULTUNER_AUDIO_ROOT:-${SOULTUNER_OPEN_AUDIO_DIR}/audio}"
if [[ -n "${SOULTUNER_SERVE_API_KEY:-}" && -z "${SOULTUNER_PLANNER_API_KEY:-}" ]]; then
  export SOULTUNER_PLANNER_API_KEY="${SOULTUNER_SERVE_API_KEY}"
fi

python amd_readiness.py --skip-adapter --skip-endpoint

prepare_open_audio() {
  local dataset_id="${SOULTUNER_OPEN_AUDIO_DATASET_ID:-hgsanyang/SoulTuner-Open-Audio-Demo}"
  local revision="${SOULTUNER_OPEN_AUDIO_REVISION:-master}"
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
if not catalog.is_file():
    raise SystemExit(f"open-audio catalog is missing: {catalog}")
rows = [json.loads(line) for line in catalog.read_text(encoding="utf-8").splitlines() if line.strip()]
if not rows:
    raise SystemExit("open-audio catalog is empty")
for row in rows:
    relpath = str(PurePosixPath(str(row.get("audio_relpath") or "")))
    expected = str(row.get("audio_sha256") or "")
    licence = str(row.get("license_url") or "")
    candidate = (root / relpath).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise SystemExit(f"unsafe open-audio path: {relpath}") from exc
    if not candidate.is_file() or not expected or not licence.startswith("http"):
        raise SystemExit(f"incomplete open-audio provenance: {row.get('song_id')}")
    actual = hashlib.sha256(candidate.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"open-audio SHA-256 mismatch: {row.get('song_id')}")
print(f"SoulTuner open-audio catalog ready: {len(rows)} tracks", flush=True)
PY
}

open_audio_log="${SOULTUNER_OPEN_AUDIO_LOG:-./soultuner-open-audio.log}"
if [[ "${SOULTUNER_ENABLE_OPEN_AUDIO:-1}" == "1" ]]; then
  prepare_open_audio >"${open_audio_log}" 2>&1 &
  open_audio_pid=$!
fi

endpoint_log="${SOULTUNER_ENDPOINT_LOG:-./soultuner-35b-endpoint.log}"
bash start_amd_35b.sh >"${endpoint_log}" 2>&1 &
endpoint_pid=$!
export SOULTUNER_ENDPOINT_PID="${endpoint_pid}"

cleanup() {
  kill "${open_audio_pid:-}" 2>/dev/null || true
  kill "${endpoint_pid}" 2>/dev/null || true
  wait "${open_audio_pid:-}" 2>/dev/null || true
  wait "${endpoint_pid}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

if ! python - <<'PY'
import json
import os
import time
import urllib.request

url = os.environ["SOULTUNER_PLANNER_BASE_URL"].rstrip("/") + "/models"
endpoint_pid = int(os.environ["SOULTUNER_ENDPOINT_PID"])
# A fresh Space may need to fetch the 72 GB base model. Persistent caches make
# later starts much faster, but the first health deadline must allow for it.
deadline = time.monotonic() + float(os.getenv("SOULTUNER_ENDPOINT_START_TIMEOUT", "1800"))
last_error = "endpoint did not respond"
while time.monotonic() < deadline:
    try:
        os.kill(endpoint_pid, 0)
    except OSError:
        raise SystemExit("Planner endpoint process exited before becoming healthy")
    try:
        request = urllib.request.Request(url, method="GET")
        api_key = (
            os.getenv("SOULTUNER_PLANNER_API_KEY", "").strip()
            or os.getenv("SOULTUNER_SERVE_API_KEY", "").strip()
        )
        if api_key:
            request.add_header("Authorization", f"Bearer {api_key}")
        with urllib.request.urlopen(request, timeout=5) as response:
            body = json.loads(response.read().decode("utf-8"))
        if isinstance(body.get("data"), list):
            print(f"SoulTuner Planner endpoint ready: {url}", flush=True)
            break
    except Exception as exc:
        last_error = f"{type(exc).__name__}: {exc}"
    time.sleep(5)
else:
    raise SystemExit(f"Planner endpoint failed to start: {last_error}")
PY
then
  echo "Planner endpoint startup failed; last log lines:" >&2
  tail -n 100 "${endpoint_log}" >&2 || true
  exit 6
fi

if [[ -n "${open_audio_pid:-}" ]] && ! wait "${open_audio_pid}"; then
  echo "Open-audio dataset startup failed; last log lines:" >&2
  tail -n 100 "${open_audio_log}" >&2 || true
  exit 7
fi

python amd_readiness.py

python app.py
