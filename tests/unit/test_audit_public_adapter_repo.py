from __future__ import annotations

import hashlib
import json
from pathlib import Path

from deploy.self_hosted_35b.audit_public_adapter_repo import audit_release


def _write_release(root: Path) -> tuple[str, int]:
    root.mkdir()
    adapter = root / "adapter_model.safetensors"
    adapter.write_bytes(b"safe-public-adapter")
    (root / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": "Qwen/Qwen3.6-35B-A3B",
                "inference_mode": True,
            }
        ),
        encoding="utf-8",
    )
    for name in ("LICENSE", "NOTICE", "README.md"):
        (root / name).write_text(name, encoding="utf-8")
    config_sha = hashlib.sha256((root / "adapter_config.json").read_bytes()).hexdigest()
    adapter_sha = hashlib.sha256(adapter.read_bytes()).hexdigest()
    (root / "SHA256SUMS").write_text(
        f"{config_sha}  adapter_config.json\n{adapter_sha}  adapter_model.safetensors\n",
        encoding="utf-8",
    )
    return adapter_sha, adapter.stat().st_size


def test_public_adapter_audit_accepts_clean_release(tmp_path: Path) -> None:
    root = tmp_path / "release"
    adapter_sha, adapter_size = _write_release(root)

    report = audit_release(
        root,
        expected_adapter_sha256=adapter_sha,
        expected_adapter_size=adapter_size,
    )

    assert report["ready_for_public"] is True
    assert report["findings"] == []


def test_public_adapter_audit_rejects_resume_state(tmp_path: Path) -> None:
    root = tmp_path / "release"
    _write_release(root)
    (root / "optimizer.pt").write_bytes(b"private-resume-state")

    report = audit_release(root)

    assert report["ready_for_public"] is False
    assert report["findings"] == ["unexpected files: optimizer.pt"]


def test_public_adapter_audit_rejects_checksum_drift(tmp_path: Path) -> None:
    root = tmp_path / "release"
    _write_release(root)
    (root / "adapter_model.safetensors").write_bytes(b"mutated")

    report = audit_release(root)

    assert report["ready_for_public"] is False
    assert report["findings"] == ["checksum mismatch: adapter_model.safetensors"]
