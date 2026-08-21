from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from deploy.modelscope_space import open_audio_bootstrap


def _write_track(root: Path) -> tuple[Path, Path]:
    audio_root = root / "audio"
    track = audio_root / "83" / "4883.mp3"
    track.parent.mkdir(parents=True)
    track.write_bytes(b"original archive bytes")
    row = {
        "song_id": "sdd-4883",
        "audio_relpath": "83/4883.mp3",
        "audio_sha256": hashlib.sha256(track.read_bytes()).hexdigest(),
        "license_url": "https://creativecommons.org/licenses/by-nc-nd/2.5/",
        "source_url": "https://www.jamendo.com/track/4883",
        "attribution": "Pan by Tom La Meche",
    }
    catalog = root / "catalog.jsonl"
    catalog.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return catalog, audio_root


def test_verify_open_audio_accepts_original_licensed_bytes(tmp_path: Path) -> None:
    catalog, audio_root = _write_track(tmp_path)

    assert open_audio_bootstrap.verify_open_audio(catalog, audio_root) == 1


def test_verify_open_audio_rejects_path_traversal(tmp_path: Path) -> None:
    catalog, audio_root = _write_track(tmp_path)
    row = json.loads(catalog.read_text(encoding="utf-8"))
    row["audio_relpath"] = "../secret.mp3"
    catalog.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError):
        open_audio_bootstrap.verify_open_audio(catalog, audio_root)


def test_materialize_downloads_then_switches_catalog(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SOULTUNER_OPEN_AUDIO_DIR", str(tmp_path))
    monkeypatch.setenv("SOULTUNER_CATALOG_PATH", "")
    monkeypatch.setenv("SOULTUNER_AUDIO_ROOT", "")

    def fake_download(command: list[str], *, check: bool, timeout: int) -> None:
        assert command[:3] == ["modelscope", "download", open_audio_bootstrap.DEFAULT_DATASET_ID]
        assert check is True
        assert timeout == 300
        _write_track(tmp_path)

    monkeypatch.setattr(subprocess, "run", fake_download)

    status = open_audio_bootstrap.materialize_open_audio()

    assert status["state"] == "ready"
    assert status["tracks"] == 1
    assert Path(open_audio_bootstrap.os.environ["SOULTUNER_CATALOG_PATH"]) == tmp_path / "catalog.jsonl"


def test_materialize_failure_restores_safe_bundled_catalog(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("SOULTUNER_OPEN_AUDIO_DIR", str(tmp_path))
    monkeypatch.setenv("SOULTUNER_CATALOG_PATH", "")
    monkeypatch.setenv("SOULTUNER_AUDIO_ROOT", "")

    def fail_download(*_args: object, **_kwargs: object) -> None:
        raise subprocess.CalledProcessError(1, "modelscope")

    monkeypatch.setattr(subprocess, "run", fail_download)

    status = open_audio_bootstrap.materialize_open_audio()

    assert status == {"state": "fallback", "tracks": 0}
    assert Path(open_audio_bootstrap.os.environ["SOULTUNER_CATALOG_PATH"]).is_file()


def test_materialize_reuses_startup_verified_catalog_without_second_audio_hash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _write_track(tmp_path)
    monkeypatch.setenv("SOULTUNER_OPEN_AUDIO_DIR", str(tmp_path))
    monkeypatch.setenv("SOULTUNER_CATALOG_PATH", "")
    monkeypatch.setenv("SOULTUNER_AUDIO_ROOT", "")
    monkeypatch.setenv("SOULTUNER_OPEN_AUDIO_ALREADY_VERIFIED", "1")
    monkeypatch.setattr(
        open_audio_bootstrap,
        "verify_open_audio",
        lambda *_args: (_ for _ in ()).throw(AssertionError("duplicate hash pass")),
    )

    status = open_audio_bootstrap.materialize_open_audio()

    assert status["state"] == "ready"
    assert status["tracks"] == 1
