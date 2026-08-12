#!/usr/bin/env bash
set -euo pipefail

# One-process entrypoint for the AMD MI308X Space profile.
# The CPU Space keeps using `python app.py`; this file is used only after the
# AMD_Dev organization grants access to an ROCm image and GPU resource.

export SOULTUNER_REQUIRE_ROCM="${SOULTUNER_REQUIRE_ROCM:-1}"
export SOULTUNER_MODEL_PROFILE="${SOULTUNER_MODEL_PROFILE:-soultuner-v4.2-35b}"
export SOULTUNER_PLANNER_BASE_URL="${SOULTUNER_PLANNER_BASE_URL:-http://127.0.0.1:8000/v1}"
export SOULTUNER_PLANNER_MODEL="${SOULTUNER_PLANNER_MODEL:-soultuner-v4.2-35b}"
export SOULTUNER_ADAPTER_DIR="${SOULTUNER_ADAPTER_DIR:-./model_cache/soultuner_adapter}"
if [[ -n "${SOULTUNER_SERVE_API_KEY:-}" && -z "${SOULTUNER_PLANNER_API_KEY:-}" ]]; then
  export SOULTUNER_PLANNER_API_KEY="${SOULTUNER_SERVE_API_KEY}"
fi

python amd_readiness.py --skip-adapter --skip-endpoint

endpoint_log="${SOULTUNER_ENDPOINT_LOG:-./soultuner-35b-endpoint.log}"
bash start_amd_35b.sh >"${endpoint_log}" 2>&1 &
endpoint_pid=$!
export SOULTUNER_ENDPOINT_PID="${endpoint_pid}"

cleanup() {
  kill "${endpoint_pid}" 2>/dev/null || true
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

python amd_readiness.py

python app.py
