import json

from config import settings as settings_module


def test_explicit_environment_wins_over_persisted_user_setting(monkeypatch, tmp_path):
    settings_path = tmp_path / "user_settings.json"
    settings_path.write_text(
        json.dumps({"dense_text_audio_backend": "muq"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(settings_module, "_USER_SETTINGS_FILE", settings_path)
    monkeypatch.setenv("DENSE_TEXT_AUDIO_BACKEND", "m2d")

    candidate = settings_module.GlobalSettings(
        dense_text_audio_backend="m2d",
    )
    applied = settings_module._load_user_overrides(candidate)

    assert candidate.dense_text_audio_backend == "m2d"
    assert "dense_text_audio_backend" not in applied


def test_persisted_user_setting_applies_without_environment_owner(monkeypatch, tmp_path):
    settings_path = tmp_path / "user_settings.json"
    settings_path.write_text(
        json.dumps({"dense_text_audio_backend": "muq"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(settings_module, "_USER_SETTINGS_FILE", settings_path)
    monkeypatch.delenv("DENSE_TEXT_AUDIO_BACKEND", raising=False)

    candidate = settings_module.GlobalSettings(
        dense_text_audio_backend="m2d",
    )
    applied = settings_module._load_user_overrides(candidate)

    assert candidate.dense_text_audio_backend == "muq"
    assert "dense_text_audio_backend" in applied
