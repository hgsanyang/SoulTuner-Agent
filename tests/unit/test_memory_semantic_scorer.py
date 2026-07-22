from pathlib import Path

from services.memory_semantic_scorer import _local_model_source, _resolve_device


def test_local_model_source_falls_back_to_model_name(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HF_HOME", str(tmp_path))

    assert _local_model_source("org/model") == "org/model"


def test_explicit_memory_device_is_preserved() -> None:
    assert _resolve_device("cpu") == "cpu"
