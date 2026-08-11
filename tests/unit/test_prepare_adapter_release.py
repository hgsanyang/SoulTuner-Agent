from __future__ import annotations

import json
from pathlib import Path

import pytest

from deploy.self_hosted_35b.prepare_adapter_release import prepare_release


def _checkpoint(path: Path) -> Path:
    path.mkdir()
    (path / "adapter_model.safetensors").write_bytes(b"safe-adapter")
    (path / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": "/private/training-host/model-cache/base",
                "inference_mode": False,
                "peft_type": "LORA",
            }
        ),
        encoding="utf-8",
    )
    (path / "optimizer.pt").write_bytes(b"must-not-be-released")
    return path


def test_prepare_release_sanitizes_base_and_excludes_resume_state(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "checkpoint-450")
    output = tmp_path / "release"

    released = prepare_release(checkpoint, output, "Qwen/Qwen3.6-35B-A3B")

    config = json.loads((output / "adapter_config.json").read_text(encoding="utf-8"))
    assert config["base_model_name_or_path"] == "Qwen/Qwen3.6-35B-A3B"
    assert config["inference_mode"] is True
    assert not (output / "optimizer.pt").exists()
    assert {path.name for path in released} == {
        "adapter_model.safetensors",
        "adapter_config.json",
        "SHA256SUMS",
    }


def test_prepare_release_never_writes_into_private_checkpoint(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "checkpoint-450")
    with pytest.raises(ValueError, match="different"):
        prepare_release(checkpoint, checkpoint, "Qwen/Qwen3.6-35B-A3B")
