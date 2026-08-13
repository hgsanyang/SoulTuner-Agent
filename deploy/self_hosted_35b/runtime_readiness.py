"""Fail-closed checks for a self-hosted SoulTuner Planner endpoint."""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def runtime_capabilities() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"device": "CPU", "accelerator": "none", "torch": None}
    if not torch.cuda.is_available():
        return {"device": "CPU", "accelerator": "none", "torch": torch.__version__}
    hip = getattr(torch.version, "hip", None)
    return {
        "device": torch.cuda.get_device_name(0),
        "accelerator": "ROCm/HIP" if hip else "CUDA",
        "torch": torch.__version__,
        "hip": hip,
    }


def probe_models(base_url: str, timeout: float = 5.0) -> dict[str, Any]:
    url = base_url.rstrip("/")
    url = url if url.endswith("/models") else f"{url}/models"
    request = urllib.request.Request(url, method="GET")
    key = (
        os.getenv("SOULTUNER_PLANNER_API_KEY", "").strip()
        or os.getenv("SOULTUNER_SERVE_API_KEY", "").strip()
    )
    if key:
        request.add_header("Authorization", f"Bearer {key}")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except (OSError, TimeoutError, ValueError, urllib.error.URLError) as exc:
        return {"ok": False, "url": url, "error": type(exc).__name__}
    models = body.get("data", []) if isinstance(body, dict) else []
    return {
        "ok": isinstance(models, list),
        "url": url,
        "models": [str(item.get("id")) for item in models if isinstance(item, dict)],
    }


def readiness_report(*, probe_endpoint: bool = True) -> dict[str, Any]:
    findings: list[str] = []
    base_value = os.getenv("SOULTUNER_BASE_MODEL", "").strip()
    adapter_value = os.getenv("SOULTUNER_ADAPTER", "").strip()
    for root, names, label in (
        (Path(base_value) if base_value else None, ("config.json",), "base model"),
        (
            Path(adapter_value) if adapter_value else None,
            ("adapter_config.json", "adapter_model.safetensors"),
            "adapter",
        ),
    ):
        if root is None:
            findings.append(f"{label} path is not configured")
            continue
        for name in names:
            if not (root / name).is_file():
                findings.append(f"missing {label} file: {name}")
    endpoint = None
    if probe_endpoint:
        endpoint = probe_models(
            os.getenv("SOULTUNER_PLANNER_BASE_URL", "http://127.0.0.1:8000/v1")
        )
        if not endpoint["ok"]:
            findings.append("Planner endpoint is not healthy")
    return {
        "ready": not findings,
        "runtime": runtime_capabilities(),
        "endpoint": endpoint,
        "findings": findings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-endpoint", action="store_true")
    args = parser.parse_args()
    report = readiness_report(probe_endpoint=not args.skip_endpoint)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
