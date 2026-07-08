import json

import pytest

from services.online_audio_retention import retain_online_audio


def test_retain_online_audio_by_basename_marks_saved(tmp_path):
    root = tmp_path / "online_acquired"
    (root / "audio").mkdir(parents=True)
    (root / "metadata").mkdir(parents=True)
    basename = "Song A - Artist A"
    (root / "audio" / f"{basename}.mp3").write_bytes(b"audio")
    (root / "metadata" / f"{basename}_meta.json").write_text(
        json.dumps(
            {
                "musicId": 123,
                "musicName": "Song A",
                "artist": [["Artist A", 1]],
                "format": "mp3",
                "audio_retention": "temporary",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = retain_online_audio(root, file_basename=basename, ext="mp3")
    meta = json.loads((root / "metadata" / f"{basename}_meta.json").read_text(encoding="utf-8"))

    assert result["success"] is True
    assert result["audio_url"] == f"/static/online_audio/{basename}.mp3"
    assert meta["audio_retention"] == "saved"
    assert meta["audio_status"] == "cached"
    assert meta["retained_at"]


def test_retain_online_audio_reports_missing_audio(tmp_path):
    root = tmp_path / "online_acquired"
    (root / "metadata").mkdir(parents=True)
    basename = "Song B - Artist B"
    (root / "metadata" / f"{basename}_meta.json").write_text(
        json.dumps({"musicId": 456, "musicName": "Song B", "artist": [["Artist B", 2]], "format": "mp3"}),
        encoding="utf-8",
    )

    result = retain_online_audio(root, file_basename=basename, ext="mp3")

    assert result["success"] is False
    assert result["reason"] == "audio_missing"


def test_retain_online_audio_rejects_unsafe_basename(tmp_path):
    with pytest.raises(ValueError):
        retain_online_audio(tmp_path, file_basename="../escape", ext="mp3")
