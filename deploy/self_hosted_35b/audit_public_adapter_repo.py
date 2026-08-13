"""Fail-closed audit for a public SoulTuner PEFT adapter directory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ALLOWED_FILES = {
    ".gitattributes",
    "adapter_config.json",
    "adapter_model.safetensors",
    "configuration.json",
    "LICENSE",
    "NOTICE",
    "README.md",
    "SHA256SUMS",
}
REQUIRED_FILES = {
    "adapter_config.json",
    "adapter_model.safetensors",
    "LICENSE",
    "NOTICE",
    "README.md",
    "SHA256SUMS",
}
CHECKSUM_FILES = {"adapter_config.json", "adapter_model.safetensors"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_checksums(path: Path) -> tuple[dict[str, str], list[str]]:
    checksums: dict[str, str] = {}
    findings: list[str] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        parts = line.split()
        if len(parts) != 2 or len(parts[0]) != 64:
            findings.append(f"invalid SHA256SUMS line {line_number}")
            continue
        digest, name = parts
        if name != Path(name).name or name in checksums:
            findings.append(f"unsafe or duplicate checksum name: {name}")
            continue
        checksums[name] = digest.casefold()
    return checksums, findings


def audit_release(
    root: Path,
    *,
    expected_adapter_sha256: str | None = None,
    expected_adapter_size: int | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    findings: list[str] = []
    if not root.is_dir():
        return {"ready_for_public": False, "root": str(root), "findings": ["not a directory"]}

    files = {path.name for path in root.iterdir() if path.is_file()}
    directories = {path.name for path in root.iterdir() if path.is_dir() and path.name != ".git"}
    missing = sorted(REQUIRED_FILES - files)
    unexpected = sorted(files - ALLOWED_FILES)
    if missing:
        findings.append(f"missing required files: {', '.join(missing)}")
    if unexpected:
        findings.append(f"unexpected files: {', '.join(unexpected)}")
    if directories:
        findings.append(f"unexpected directories: {', '.join(sorted(directories))}")

    config_path = root / "adapter_config.json"
    if config_path.is_file():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            if config.get("base_model_name_or_path") != "Qwen/Qwen3.6-35B-A3B":
                findings.append("adapter base_model_name_or_path is not the public model ID")
            if config.get("inference_mode") is not True:
                findings.append("adapter inference_mode must be true")
        except (OSError, ValueError, TypeError):
            findings.append("adapter_config.json is not valid JSON")

    checksum_path = root / "SHA256SUMS"
    verified: dict[str, str] = {}
    if checksum_path.is_file():
        checksums, checksum_findings = _read_checksums(checksum_path)
        findings.extend(checksum_findings)
        if set(checksums) != CHECKSUM_FILES:
            findings.append("SHA256SUMS must cover exactly adapter config and weights")
        for name, expected in checksums.items():
            candidate = root / name
            if candidate.is_file():
                actual = sha256(candidate)
                verified[name] = actual
                if actual != expected:
                    findings.append(f"checksum mismatch: {name}")

    adapter_path = root / "adapter_model.safetensors"
    if adapter_path.is_file():
        adapter_sha = verified.get(adapter_path.name) or sha256(adapter_path)
        if expected_adapter_sha256 and adapter_sha != expected_adapter_sha256.casefold():
            findings.append("adapter SHA-256 does not match checkpoint-450 release identity")
        if expected_adapter_size is not None and adapter_path.stat().st_size != expected_adapter_size:
            findings.append("adapter size does not match checkpoint-450 release identity")

    return {
        "ready_for_public": not findings,
        "root": str(root),
        "files": sorted(files),
        "verified_sha256": verified,
        "findings": findings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--expected-adapter-sha256")
    parser.add_argument("--expected-adapter-size", type=int)
    args = parser.parse_args()
    report = audit_release(
        args.model_dir,
        expected_adapter_sha256=args.expected_adapter_sha256,
        expected_adapter_size=args.expected_adapter_size,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ready_for_public"] else 8


if __name__ == "__main__":
    raise SystemExit(main())
