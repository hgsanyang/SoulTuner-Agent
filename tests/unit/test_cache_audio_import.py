from __future__ import annotations

import struct
import wave
from pathlib import Path
from types import SimpleNamespace

from services.audio_format import MetadataCandidate
from services.cache_audio_import import (
    build_existing_digest_index,
    choose_preferred_cache_entries,
    plan_cache_audio,
    publish_cache_audio,
    remove_published_files,
    sha256_file,
)


def _wav_bytes(path: Path) -> bytes:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(8000)
        handle.writeframes(b"".join(struct.pack("<h", 0) for _ in range(1600)))
    return path.read_bytes()


def _entry(path: Path, *, song_id: str = "42", quality: str = "320", size: int | None = None):
    return SimpleNamespace(
        path=str(path),
        song_id=song_id,
        quality=quality,
        bytes=int(size if size is not None else path.stat().st_size),
        title="Quiet Test",
        artist="Test Artist",
        album="Test Album",
        duration_ms=200,
        release_year=2024,
        album_id="album-1",
        cover_url="https://example.invalid/cover.jpg",
        aliases=["Alias"],
    )


def _patch_audio_validation(monkeypatch) -> None:
    monkeypatch.setattr("services.audio_decoder.shutil.which", lambda name: name)
    monkeypatch.setattr(
        "services.audio_format.probe",
        lambda _path: {
            "codec": "pcm_s16le",
            "sample_rate": 8000,
            "channels": 1,
            "bitrate": 0,
            "duration_ms": 200,
            "format_name": "wav",
        },
    )
    monkeypatch.setattr(
        "services.cache_audio_import.read_metadata",
        lambda _path: MetadataCandidate(
            container="wav", codec="pcm_s16le", sample_rate=8000,
            channels=1, duration_ms=200,
        ),
    )


def test_choose_preferred_cache_entry_skips_partial_and_lower_quality(tmp_path: Path) -> None:
    low = tmp_path / "42-128-a.uc"
    high = tmp_path / "42-999-b.uc"
    partial = tmp_path / "43-999-c.uc!"
    for path in (low, high, partial):
        path.write_bytes(b"x")

    chosen, skipped = choose_preferred_cache_entries([
        _entry(low, quality="128"),
        _entry(high, quality="999"),
        _entry(partial, song_id="43", quality="999"),
    ])

    assert [Path(item.path).name for item in chosen] == [high.name]
    assert {item["reason"] for item in skipped} == {"lower_quality_copy", "partial_download"}


def test_digest_index_hashes_only_candidate_sizes(tmp_path: Path) -> None:
    audio = tmp_path / "audio"
    audio.mkdir()
    wanted = audio / "wanted.mp3"
    ignored = audio / "ignored.flac"
    wanted.write_bytes(b"a" * 100)
    ignored.write_bytes(b"b" * 200)

    index = build_existing_digest_index([audio], candidate_sizes={100})

    assert index == {sha256_file(wanted): str(wanted)}


def test_plan_decodes_to_real_container_and_detects_hash_duplicate(tmp_path: Path, monkeypatch) -> None:
    _patch_audio_validation(monkeypatch)
    raw_wav = _wav_bytes(tmp_path / "source.wav")
    cache = tmp_path / "42-320-a.uc"
    cache.write_bytes(bytes(byte ^ 0xA3 for byte in raw_wav))
    existing = tmp_path / "existing.wav"
    existing.write_bytes(raw_wav)

    plan = plan_cache_audio(
        _entry(cache),
        tmp_path / "stage",
        existing_digests={sha256_file(existing): str(existing)},
    )

    assert plan.state == "duplicate_exact_hash"
    assert plan.container == "wav"
    assert plan.reason == str(existing)


def test_publish_writes_library_layout_and_is_reversible(tmp_path: Path, monkeypatch) -> None:
    _patch_audio_validation(monkeypatch)
    raw_wav = _wav_bytes(tmp_path / "source.wav")
    cache = tmp_path / "42-320-a.uc"
    cache.write_bytes(bytes(byte ^ 0xA3 for byte in raw_wav))
    plan = plan_cache_audio(_entry(cache), tmp_path / "stage")

    published = publish_cache_audio(
        plan,
        processed_root=tmp_path / "processed_audio",
        lyrics="[00:00.00]test",
        run_id="run-1",
    )

    assert Path(published.record["audio_path"]).exists()
    assert published.record["audio_url"].endswith(".wav")
    assert published.record["catalog_tier"] == "library"
    assert published.record["cache_import_run_id"] == "run-1"
    assert published.record["release_year"] == 2024
    assert remove_published_files(published.created_files) == len(published.created_files)
    assert all(not Path(path).exists() for path in published.created_files)


def test_same_hash_in_one_batch_is_not_published_twice(tmp_path: Path, monkeypatch) -> None:
    _patch_audio_validation(monkeypatch)
    raw_wav = _wav_bytes(tmp_path / "source.wav")
    seen: set[str] = set()
    plans = []
    for song_id in ("1", "2"):
        cache = tmp_path / f"{song_id}-320-a.uc"
        cache.write_bytes(bytes(byte ^ 0xA3 for byte in raw_wav))
        plans.append(plan_cache_audio(_entry(cache, song_id=song_id), tmp_path / f"stage-{song_id}", seen_digests=seen))

    assert plans[0].state == "ready"
    assert plans[1].state == "duplicate_within_batch"
