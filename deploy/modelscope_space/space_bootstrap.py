"""Start the optional local 35B Planner without blocking the Gradio server."""

from __future__ import annotations

import atexit
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


PROFILE_SOULTUNER = "soultuner-v4.2-35b"
DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_CHAT_MODEL = "qwen3.6-35b-a3b"
_planner_process: subprocess.Popen[bytes] | None = None
_planner_log_path: Path | None = None
_planner_log_thread: threading.Thread | None = None


def _uses_local_endpoint(base_url: str) -> bool:
    host = (urlparse(base_url).hostname or "").casefold()
    return host in {"127.0.0.1", "localhost", "::1"}


def _endpoint_ready(base_url: str) -> bool:
    endpoint = f"{base_url.rstrip('/')}/models"
    headers = {"Accept": "application/json"}
    api_key = (
        os.getenv("SOULTUNER_PLANNER_API_KEY", "").strip()
        or os.getenv("SOULTUNER_SERVE_API_KEY", "").strip()
    )
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(endpoint, headers=headers)
    try:
        with urlopen(request, timeout=1.5) as response:
            if not 200 <= int(response.status) < 300:
                return False
            body = json.loads(response.read().decode("utf-8"))
        models = {
            str(item.get("id"))
            for item in body.get("data", [])
            if isinstance(item, dict) and item.get("id")
        }
        required = {os.getenv("SOULTUNER_PLANNER_MODEL", PROFILE_SOULTUNER)}
        if os.getenv("SOULTUNER_DUAL_ROLE_MODELS", "0") == "1":
            required.add(os.getenv("SOULTUNER_CHAT_MODEL", DEFAULT_CHAT_MODEL))
        return required <= models
    except (HTTPError, URLError, OSError, TimeoutError, ValueError, json.JSONDecodeError):
        return False


def _mirror_planner_output(stream: Any, log_path: Path) -> None:
    """Copy child output to a bounded location and the Space runtime log."""

    try:
        with log_path.open("ab", buffering=0) as log_file:
            for raw_line in iter(stream.readline, b""):
                log_file.write(raw_line)
                line = raw_line.decode("utf-8", errors="replace").rstrip()
                if line:
                    print(f"[SoulTuner 35B] {line}", file=sys.stdout, flush=True)
    finally:
        stream.close()


def _stop_local_planner() -> None:
    global _planner_process, _planner_log_thread
    process = _planner_process
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
    _planner_process = None
    if _planner_log_thread is not None:
        _planner_log_thread.join(timeout=2)
        _planner_log_thread = None


def planner_runtime_status() -> dict[str, Any]:
    """Return the live local-endpoint state without exposing secrets or log text."""

    profile = os.getenv("SOULTUNER_MODEL_PROFILE", "").strip()
    if profile != PROFILE_SOULTUNER:
        return {"requested": False, "state": "disabled", "profile": profile or "demo-heuristic"}

    base_url = os.getenv("SOULTUNER_PLANNER_BASE_URL", "").strip() or DEFAULT_BASE_URL
    if not _uses_local_endpoint(base_url):
        return {
            "requested": True,
            "state": "external-endpoint",
            "profile": profile,
            "base_url": base_url,
        }
    if _endpoint_ready(base_url):
        return {
            "requested": True,
            "state": "ready",
            "profile": profile,
            "base_url": base_url,
            "pid": _planner_process.pid if _planner_process is not None else None,
        }

    process = _planner_process
    if process is None:
        state, returncode = "not-started", None
    else:
        returncode = process.poll()
        state = "starting" if returncode is None else "failed"
    return {
        "requested": True,
        "state": state,
        "profile": profile,
        "base_url": base_url,
        "pid": process.pid if process is not None else None,
        "returncode": returncode,
        "log": _planner_log_path.name if _planner_log_path is not None else None,
    }


def launch_local_planner_if_requested() -> dict[str, Any]:
    """Launch the local endpoint once and return a public startup summary.

    The child performs dependency checks, model downloads and vLLM startup in
    the background. Gradio can therefore bind its public port immediately while
    the first 72 GB base-model download is still in progress.
    """

    global _planner_process, _planner_log_path, _planner_log_thread
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

    # ModelScope's Gradio runtime starts ``python app.py`` directly, so the
    # public Space does not necessarily pass through start_space_amd.sh.  Set
    # the same dual-role defaults here: one 35B base copy serves prose, while
    # the named SoulTuner adapter remains the structure-only Planner.
    os.environ.setdefault("SOULTUNER_DUAL_ROLE_MODELS", "1")
    os.environ.setdefault("SOULTUNER_CHAT_MODEL", DEFAULT_CHAT_MODEL)
    os.environ.setdefault("SOULTUNER_CHAT_BASE_URL", base_url)

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
    _planner_process = subprocess.Popen(
        ["bash", str(script)],
        cwd=root,
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    if _planner_process.stdout is None:  # pragma: no cover - guaranteed by PIPE
        raise RuntimeError("local Planner stdout pipe was not created")
    _planner_log_path = log_path
    _planner_log_thread = threading.Thread(
        target=_mirror_planner_output,
        args=(_planner_process.stdout, log_path),
        name="soultuner-35b-log-mirror",
        daemon=True,
    )
    _planner_log_thread.start()
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
    if state == "ready":
        return "Planner 启动：`SoulTuner V4.2 35B 已就绪`  ·  本地 ROCm/vLLM 端点健康。"
    if state == "failed":
        return (
            "Planner 启动：`SoulTuner V4.2 35B 启动失败`  ·  "
            f"后台进程退出码 `{status.get('returncode')}`；请查看创空间运行日志。"
        )
    if state == "not-started":
        return "Planner 启动：`SoulTuner V4.2 35B 未启动`  ·  请重新部署当前创空间。"
    return (
        "Planner 启动：`SoulTuner V4.2 35B 正在后台初始化`  ·  "
        "首次启动需要下载约 72 GB 基座；端点就绪前请求会安全回退。"
    )


def live_startup_markdown() -> str:
    return startup_markdown(planner_runtime_status())
