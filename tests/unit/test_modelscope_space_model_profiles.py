from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "deploy"
    / "modelscope_space"
    / "model_profiles.py"
)
SPEC = importlib.util.spec_from_file_location("soultuner_space_model_profiles", MODULE_PATH)
assert SPEC and SPEC.loader
profiles = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(profiles)


def test_4070_profile_is_one_dropdown_value(monkeypatch) -> None:
    monkeypatch.setenv("DASHSCOPE_API_KEY", "secret")
    monkeypatch.delenv("SOULTUNER_MODEL_PROFILE", raising=False)
    assert profiles.default_profile() == profiles.PROFILE_QWEN
    assert profiles.resolve_profile(profiles.PROFILE_QWEN)[0]["model"] == "qwen3.7-plus"


def test_35b_profile_uses_creation_space_endpoint(monkeypatch) -> None:
    monkeypatch.setenv("SOULTUNER_PLANNER_ENDPOINT", "https://example/v1/chat/completions")
    config, _ = profiles.resolve_profile(profiles.PROFILE_SOULTUNER)
    assert config["endpoint"] == "https://example/v1/chat/completions"
    assert config["model"] == "soultuner-v4.2-35b"


def test_unconfigured_profile_falls_back_without_secret(monkeypatch) -> None:
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    config, note = profiles.resolve_profile(profiles.PROFILE_QWEN)
    assert config is None
    assert "DASHSCOPE_API_KEY" in note
