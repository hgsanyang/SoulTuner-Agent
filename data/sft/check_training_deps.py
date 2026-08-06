#!/usr/bin/env python
"""Record and verify the runtime a training run actually got.

Two different jobs, deliberately in one place:

* **Record** — torch, its HIP build, triton, ms-swift and the AMD-only extras go
  into the run record. Without them a score six weeks from now cannot be
  attributed to anything.
* **Verify** — a missing package fails here, cheaply, instead of thirty seconds
  into a billed instance. `flash-linear-attention` was learned this way: the 9B
  preflight died on `ImportError` after the model had been located but before a
  single step ran.

What this deliberately does **not** do is install or upgrade anything. The ROCm
PyTorch on the training box is a custom build; any `pip install -U` that touches
torch, triton or their CUDA-named packages silently replaces it with a PyPI
build that has no HIP support, and the failure then looks like "the GPU
disappeared". Reporting a missing dependency is the correct action; fixing it is
an operator decision.
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import re
import sys
from pathlib import Path

EXIT_OK = 0
EXIT_MISSING = 5

#: Packages whose exact version is pinned for AMD training.
PINNED_FILE_DEFAULT = Path(__file__).with_name("requirements-amd.txt")

#: Recorded but never pinned here — the box owns these and this file must not
#: express an opinion that could tempt anyone to "fix" them with pip.
RECORDED = ("torch", "ms-swift", "transformers", "peft", "trl", "accelerate", "triton")

_PIN = re.compile(r"^\s*([A-Za-z0-9._-]+)\s*==\s*([^\s#]+)")


def read_pins(path: Path) -> dict[str, str]:
    pins: dict[str, str] = {}
    if not path.is_file():
        return pins
    for line in path.read_text(encoding="utf-8").splitlines():
        match = _PIN.match(line)
        if match:
            pins[match.group(1).lower()] = match.group(2)
    return pins


def version_of(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def collect(pins: dict[str, str]) -> tuple[dict, list[str]]:
    report: dict = {"recorded": {}, "pinned": {}, "problems": []}
    problems: list[str] = []

    for name in RECORDED:
        report["recorded"][name] = version_of(name)

    try:
        import torch

        report["recorded"]["torch.version.hip"] = torch.version.hip
        report["recorded"]["cuda_available"] = bool(torch.cuda.is_available())
        report["recorded"]["device_count"] = torch.cuda.device_count()
        if not torch.version.hip:
            problems.append(
                "torch.version.hip is empty — this is not the ROCm build the "
                "training box is supposed to have"
            )
    except Exception as exc:  # noqa: BLE001 - importing torch is itself the check
        problems.append(f"torch is not importable: {type(exc).__name__}: {exc}")

    for name, expected in pins.items():
        found = version_of(name)
        report["pinned"][name] = {"expected": expected, "found": found}
        if found is None:
            problems.append(
                f"{name}=={expected} is required for AMD training and is not installed; "
                f"install it explicitly (see data/sft/requirements-amd.txt). "
                "This script will not install or upgrade anything."
            )
        elif found != expected:
            problems.append(f"{name} is {found}, expected {expected}")

    report["problems"] = problems
    return report, problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requirements", type=Path, default=PINNED_FILE_DEFAULT)
    parser.add_argument("--json", dest="json_out", type=Path)
    args = parser.parse_args(argv)

    report, problems = collect(read_pins(args.requirements))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    recorded = report["recorded"]
    print(
        "TRAINING DEPS: torch {torch} (hip {hip}) | triton {triton} | "
        "ms-swift {swift} | transformers {tf}".format(
            torch=recorded.get("torch"),
            hip=recorded.get("torch.version.hip"),
            triton=recorded.get("triton"),
            swift=recorded.get("ms-swift"),
            tf=recorded.get("transformers"),
        )
    )
    for name, state in report["pinned"].items():
        print(f"  pinned {name}: expected {state['expected']}, found {state['found']}")

    if problems:
        print("TRAINING DEPS FAIL:")
        for problem in problems:
            print(f"  - {problem}")
        return EXIT_MISSING

    print("TRAINING DEPS OK")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
