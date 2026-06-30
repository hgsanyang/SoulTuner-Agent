from __future__ import annotations

import json
from pathlib import Path

from data.pipeline.yt_dlp_manual_flywheel import (
    build_records,
    stage_records,
    summarise,
)


def _write_info(path: Path, **overrides):
    payload = {
        "id": "abc123",
        "track": "Demo Song",
        "artists": ["Demo Artist"],
        "album": "Demo Album",
        "duration": 123,
        "webpage_url": "https://example.test/watch?v=abc123",
        "extractor_key": "Youtube",
        "tags": ["Demo", "Indie"],
        "description": "Provided to YouTube by Example.",
    }
    payload.update(overrides)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_build_records_uses_info_json_metadata(tmp_path):
    raw = tmp_path / "downloads"
    processed = tmp_path / "processed_audio"
    raw.mkdir()
    audio = raw / "0001 - fallback title - fallback artist.mp3"
    audio.write_bytes(b"fake-audio")
    _write_info(audio.with_suffix(".info.json"))
    audio.with_suffix(".jpg").write_bytes(b"fake-cover")

    records = build_records(raw, processed)

    assert len(records) == 1
    record = records[0]
    assert record.title == "Demo Song"
    assert record.artist == "Demo Artist"
    assert record.album == "Demo Album"
    assert record.duration == 123000
    assert record.music_id == "yt_abc123"
    assert record.status == "stage_needed"
    assert record.target_basename == "Demo Song - Demo Artist"


def test_stage_records_writes_processed_audio_layout(tmp_path):
    raw = tmp_path / "downloads"
    processed = tmp_path / "processed_audio"
    raw.mkdir()
    audio = raw / "0001 - Demo Song - Demo Artist.mp3"
    audio.write_bytes(b"fake-audio")
    _write_info(audio.with_suffix(".info.json"))
    audio.with_suffix(".jpg").write_bytes(b"fake-cover")
    records = build_records(raw, processed)

    staged = stage_records(records)

    assert len(staged) == 1
    song = staged[0]
    assert Path(song.audio_path).exists()
    assert Path(song.metadata_path).exists()
    assert Path(song.lrc_path).exists()
    metadata = json.loads(Path(song.metadata_path).read_text(encoding="utf-8"))
    assert metadata["source"] == "yt_dlp_manual"
    assert metadata["dataset"] == "yt_dlp_manual"
    assert metadata["musicName"] == "Demo Song"
    assert metadata["artist"] == [["Demo Artist", 0]]


def test_existing_processed_song_is_skipped_by_default(tmp_path):
    raw = tmp_path / "downloads"
    processed = tmp_path / "processed_audio"
    raw.mkdir()
    (processed / "audio").mkdir(parents=True)
    audio = raw / "0001 - Demo Song - Demo Artist.mp3"
    audio.write_bytes(b"fake-audio")
    _write_info(audio.with_suffix(".info.json"))
    (processed / "audio" / "Demo Song - Demo Artist.mp3").write_bytes(b"already")

    records = build_records(raw, processed)
    staged = stage_records(records)

    assert summarise(records)["by_status"] == {"already_processed": 1}
    assert staged == []


def test_stage_records_dedupes_same_music_id_preferring_mp3(tmp_path):
    raw = tmp_path / "downloads"
    processed = tmp_path / "processed_audio"
    raw.mkdir()
    webm = raw / "0001 - Demo Song - Demo Artist.webm"
    mp3 = raw / "0002 - Demo Song - Demo Artist.mp3"
    webm.write_bytes(b"fake-webm")
    mp3.write_bytes(b"fake-mp3")
    _write_info(webm.with_suffix(".info.json"), id="same-id")
    _write_info(mp3.with_suffix(".info.json"), id="same-id")

    records = build_records(raw, processed)
    staged = stage_records(records)

    assert len(staged) == 1
    assert staged[0].music_id == "yt_same-id"
    assert staged[0].audio_path.endswith(".mp3")
