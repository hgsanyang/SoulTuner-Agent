"""Start the optional local 35B Planner without blocking the Gradio server."""

from __future__ import annotations

import atexit
import os
import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


PROFILE_SOULTUNER = "soultuner-v4.2-35b"
DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
_planner_process: subprocess.Popen[bytes] | None = None
_planner_log = None


def _uses_local_endpoint(base_url: str) -> bool:
    host = (urlparse(base_url).hostname or "").casefold()
    return host in {"127.0.0.1", "localhost", "::1"}


def _stop_local_planner() -> None:
    global _planner_process, _planner_log
    process = _planner_process
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
    _planner_process = None
    if _planner_log is not None:
        _planner_log.close()
        _planner_log = None


def launch_local_planner_if_requested() -> dict[str, Any]:
    """Launch the local endpoint once and return a public startup summary.

    The child performs dependency checks, model downloads and vLLM startup in
    the background. Gradio can therefore bind its public port immediately while
    the first 72 GB base-model download is still in progress.
    """

    global _planner_process, _planner_log
    profile = os.getenv("SOULTUNER_MODEL_PROFILE", "").strip()
    if profile != PROFILE_SOULTUNER:
        return {"requested": False, "state": "disabled", "profile": profile or "demo-heuristic"}

    base_url = os.getenv("SOULTUNER_PLANNER_BASE_URL", "").strip() or DEFAULT_BASE_URL
    os.environ.setdefault("SOULTUNER_PLANNER_BASE_URL", base_url)
    os.environ.setdefault("SOULTUNER_PLANNER_MODEL", PROFILE_SOULTUNER)
    os.environ.setdefault("SOULTUNER_REQUIRE_ROCM", "1")

    if not _uses_local_endpoint(base_url):
        return {
            "requested": True,
            "state": "external-endpoint",
            "profile": profile,
            "base_url": base_url,
        }

    if _planner_process is not None and _planner_process.poll() is None:
        return {
            "requested": True,
            "state": "starting",
            "profile": profile,
            "base_url": base_url,
            "pid": _planner_process.pid,
        }

    root = Path(__file__).resolve().parent
    script = root / "start_amd_35b.sh"
    if not script.is_file():
        raise RuntimeError(f"missing local Planner launcher: {script.name}")

    log_path = Path(os.getenv("SOULTUNER_ENDPOINT_LOG", "soultuner-35b-endpoint.log"))
    if not log_path.is_absolute():
        log_path = root / log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _planner_log = log_path.open("ab", buffering=0)
    _planner_process = subprocess.Popen(
        ["bash", str(script)],
        cwd=root,
        env=os.environ.copy(),
        stdout=_planner_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    atexit.register(_stop_local_planner)
    return {
        "requested": True,
        "state": "starting",
        "profile": profile,
        "base_url": base_url,
        "pid": _planner_process.pid,
        "log": log_path.name,
    }


def startup_markdown(status: dict[str, Any]) -> str:
    state = status.get("state")
    if state == "disabled":
        return "Planner 启动：`CPU 安全演示`"
    if state == "external-endpoint":
        return "Planner 启动：`外部兼容端点`"
    return (
        "Planner 启动：`SoulTuner V4.2 35B 正在后台初始化`  ·  "
        "首次启动需要下载约 72 GB 基座；端点就绪前请求会安全回退。"
    )
