"""Fail-closed readiness checks for the optional AMD 35B Space profile."""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from hardware import runtime_capabilities


def _probe_endpoint(base_url: str, timeout: float = 5.0) -> dict[str, Any]:
    base = base_url.rstrip("/")
    url = base if base.endswith("/models") else f"{base}/models"
    request = urllib.request.Request(url, method="GET")
    api_key = os.getenv("SOULTUNER_PLANNER_API_KEY", "").strip() or os.getenv("SOULTUNER_SERVE_API_KEY", "").strip()
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except (OSError, TimeoutError, ValueError, urllib.error.URLError) as exc:
        return {"ok": False, "url": url, "error": type(exc).__name__}
    models = body.get("data", []) if isinstance(body, dict) else []
    names = [str(item.get("id")) for item in models if isinstance(item, dict)]
    return {"ok": True, "url": url, "models": names}


def readiness_report(
    *,
    require_rocm: bool | None = None,
    probe_adapter: bool = True,
    probe_endpoint: bool = True,
) -> dict[str, Any]:
    """Return a JSON-serializable report without downloading or starting a model."""

    required = os.getenv("SOULTUNER_REQUIRE_ROCM", "0").strip() == "1" if require_rocm is None else require_rocm
    runtime = runtime_capabilities()
    is_rocm = runtime.get("accelerator") == "ROCm/HIP"
    findings: list[str] = []
    if required and not is_rocm:
        findings.append("ROCm/HIP GPU is required but was not detected")

    adapter_dir = os.getenv("SOULTUNER_ADAPTER_DIR", "").strip()
    if probe_adapter and adapter_dir:
        root = Path(adapter_dir)
        for name in ("adapter_config.json", "adapter_model.safetensors"):
            if not (root / name).is_file():
                findings.append(f"missing adapter file: {name}")

    endpoint_url = os.getenv("SOULTUNER_PLANNER_BASE_URL", "").strip()
    endpoint = None
    if probe_endpoint and endpoint_url:
        endpoint = _probe_endpoint(endpoint_url)
        if not endpoint["ok"]:
            findings.append("Planner endpoint is not healthy")

    return {
        "ready": not findings,
        "rocm_required": required,
        "runtime": runtime,
        "adapter_dir": adapter_dir or None,
        "endpoint": endpoint,
        "findings": findings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-adapter",
        action="store_true",
        help="Only check the runtime before model assets have been downloaded.",
    )
    parser.add_argument(
        "--skip-endpoint",
        action="store_true",
        help="Skip the HTTP probe while the endpoint has not started yet.",
    )
    args = parser.parse_args()
    report = readiness_report(
        probe_adapter=not args.skip_adapter,
        probe_endpoint=not args.skip_endpoint,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
