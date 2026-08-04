#!/usr/bin/env python
"""Refuse to run on CPU when the caller asked for GPU.

The failure this exists to prevent has no symptom. A container built with CUDA
wheels but started without a device reservation imports torch fine, reports
``cuda.is_available() == False``, and quietly extracts every vector on CPU. From
the outside that is indistinguishable from "ingestion is slow today" — for
hours, or until someone thinks to check.

Two ways in:

    MUSIC_REQUIRE_CUDA=1 python scripts/assert_cuda.py     # exit 1 if no GPU
    python scripts/assert_cuda.py --report                 # print, never fail

Only the first fails, and only when something explicitly asked for GPU. A CPU
deployment that never sets the flag is a legitimate configuration, not an error.
"""

from __future__ import annotations

import argparse
import os
import sys


def cuda_report() -> dict:
    """What torch actually sees. Never raises."""
    try:
        import torch
    except ImportError as exc:
        return {"ok": False, "reason": f"torch 未安装: {exc}", "torch": "", "devices": []}

    build = getattr(torch.version, "cuda", None)
    hip = getattr(torch.version, "hip", None)
    available = bool(torch.cuda.is_available())
    devices = []
    if available:
        try:
            devices = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        except Exception as exc:                      # driver present, device unusable
            return {"ok": False, "reason": f"设备枚举失败: {type(exc).__name__}: {exc}",
                    "torch": torch.__version__, "cuda_build": build, "devices": []}

    if not build and not hip:
        reason = (
            f"当前是 CPU 版 PyTorch（{torch.__version__}）。"
            "镜像需要用 GPU overlay 构建："
            "docker compose -f docker-compose.yml -f docker-compose.gpu.yml build"
        )
    elif not available:
        reason = (
            f"PyTorch 带 CUDA {build} 但看不到设备。"
            "多半是 compose 服务缺 deploy.resources.reservations.devices，"
            "或宿主机没装 NVIDIA Container Toolkit。"
        )
    else:
        reason = ""

    return {
        "ok": available,
        "reason": reason,
        "torch": torch.__version__,
        "cuda_build": build,
        "hip_build": hip,
        "device_count": len(devices),
        "devices": devices,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="store_true",
                        help="print the state and always exit 0")
    args = parser.parse_args()

    report = cuda_report()
    required = os.getenv("MUSIC_REQUIRE_CUDA", "").strip().lower() in {"1", "true", "yes"}

    if report["ok"]:
        print(f"CUDA OK: torch {report['torch']} (cu{report['cuda_build']}) "
              f"-> {report['device_count']} device(s): {', '.join(report['devices'])}")
        return 0

    line = f"CUDA 不可用: {report['reason']}"
    if args.report or not required:
        # Not an error unless GPU was asked for. A CPU deployment is valid.
        print(line + ("" if required else "  (未设置 MUSIC_REQUIRE_CUDA，按 CPU 运行)"))
        return 0

    print(f"FAIL: {line}", file=sys.stderr)
    print("      设置了 MUSIC_REQUIRE_CUDA=1 却拿不到 GPU —— 拒绝以 CPU 静默降级运行。",
          file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
