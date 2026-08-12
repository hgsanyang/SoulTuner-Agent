"""Runtime capability probe for CPU, NVIDIA CUDA and AMD ROCm hosts."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from typing import Any


def _rocminfo_name() -> str | None:
    executable = shutil.which("rocminfo")
    if not executable:
        return None
    try:
        result = subprocess.run(
            [executable],
            capture_output=True,
            check=False,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    for line in result.stdout.splitlines():
        if "Marketing Name:" in line:
            return line.split(":", 1)[1].strip()
    return None


def runtime_capabilities() -> dict[str, Any]:
    result: dict[str, Any] = {
        "device": "CPU",
        "accelerator": "none",
        "torch": "not installed",
        "hip": None,
        "cuda": None,
        "gpu_name": None,
        "dev_kfd": os.path.exists("/dev/kfd"),
    }
    try:
        import torch
    except ImportError:
        rocm_name = _rocminfo_name()
        if rocm_name:
            result.update(device="AMD GPU", accelerator="ROCm", gpu_name=rocm_name)
        return result

    result["torch"] = torch.__version__
    result["hip"] = getattr(torch.version, "hip", None)
    result["cuda"] = getattr(torch.version, "cuda", None)
    if torch.cuda.is_available():
        result["gpu_name"] = torch.cuda.get_device_name(0)
        if result["hip"]:
            result.update(device="AMD GPU", accelerator="ROCm/HIP")
        else:
            result.update(device="NVIDIA GPU", accelerator="CUDA")
    return result


def runtime_markdown() -> str:
    capabilities = runtime_capabilities()
    return (
        f"**运行设备：{capabilities['device']}**  ·  "
        f"加速栈：`{capabilities['accelerator']}`  ·  "
        f"GPU：`{capabilities['gpu_name'] or '未探测到'}`  ·  "
        f"PyTorch：`{capabilities['torch']}`  ·  HIP：`{capabilities['hip'] or 'N/A'}`"
    )


if __name__ == "__main__":
    print(json.dumps(runtime_capabilities(), ensure_ascii=False, indent=2))
