from __future__ import annotations

import hashlib
import json

from tools.data.song_describer_pipeline import hash_file, write_jsonl
from tools.data.song_describer_public_bundle import (
    _probe_one,
    build_public_bundle,
    jamendo_cover_url,
    jpeg_dimensions,
    placeholder_svg,
)


def _jpeg(width: int = 600, height: int = 600, suffix: bytes = b"") -> bytes:
    # Minimal structure sufficient for the deliberately metadata-only JPEG probe.
    return (
        b"\xff\xd8"
        + b"\xff\xc0"
        + b"\x00\x11"
        + b"\x08"
        + height.to_bytes(2, "big")
        + width.to_bytes(2, "big")
        + b"\x03\x01\x11\x00\x02\x11\x00\x03\x11\x00"
        + b"\xff\xd9"
        + suffix
        + b"x" * 600
    )


def _row(tmp_path, *, track_id="100", album_id="200"):
    audio = tmp_path / "audio" / "00" / f"{track_id}.mp3"
    audio.parent.mkdir(parents=True, exist_ok=True)
    audio.write_bytes(b"original-audio-" + track_id.encode())
    return {
        "song_id": f"sdd-{track_id}",
        "track_id": track_id,
        "album_id": album_id,
        "title": "Quiet Rain",
        "artist": "Example Artist",
        "album": "Night Rooms",
        "source_url": f"https://www.jamendo.com/track/{track_id}",
        "audio_relpath": f"00/{track_id}.mp3",
        "audio_available": True,
        "audio_sha256": hash_file(audio),
        "license_url": "http://creativecommons.org/licenses/by-nc-sa/3.0/",
        "attribution": "Quiet Rain by Example Artist",
    }


def test_jamendo_cover_url_uses_official_documented_endpoint(tmp_path):
    row = _row(tmp_path)

    assert jamendo_cover_url(row) == (
        "https://usercontent.jamendo.com/?type=album&id=200&width=600&trackid=100"
    )


def test_jpeg_dimensions_does_not_decode_or_rewrite():
    payload = _jpeg(600, 500)

    assert jpeg_dimensions(payload) == (600, 500)


def test_placeholder_is_stable_and_escaped(tmp_path):
    row = _row(tmp_path)
    row["title"] = "Rain & <Night>"

    first = placeholder_svg(row)
    second = placeholder_svg(dict(row))

    assert first == second
    assert b"Rain &amp; &lt;Night&gt;" in first
    assert b"SoulTuner" in first


def test_default_remote_cover_becomes_local_placeholder(tmp_path):
    row = _row(tmp_path)
    default = _jpeg(suffix=b"default")

    result = _probe_one(
        row,
        cover_root=tmp_path / "covers",
        default_cover_sha256=hashlib.sha256(default).hexdigest(),
        refresh=False,
        fetcher=lambda _url: (default, "image/jpeg"),
    )

    assert result["cover_status"] == "placeholder"
    assert result["placeholder_reason"] == "jamendo_default_image"
    assert result["display_cover_url"].endswith("sdd-100.svg")
    assert result["official_cover_redistributable_in_bundle"] is False


def test_official_cover_records_remote_provenance_but_no_redistribution(tmp_path):
    row = _row(tmp_path)
    default = _jpeg(suffix=b"default")
    official = _jpeg(suffix=b"official")

    result = _probe_one(
        row,
        cover_root=tmp_path / "covers",
        default_cover_sha256=hashlib.sha256(default).hexdigest(),
        refresh=False,
        fetcher=lambda _url: (official, "image/jpeg"),
    )

    assert result["cover_status"] == "official_remote"
    assert result["display_cover_url"].startswith("https://usercontent.jamendo.com/")
    assert result["remote_cover_width"] == 600
    assert result["remote_cover_height"] == 600
    assert result["remote_cover_sha256"] == hashlib.sha256(official).hexdigest()
    assert result["official_cover_redistributable_in_bundle"] is False
    assert "provided by Jamendo" in result["cover_attribution"]


def test_bundle_hardlinks_audio_and_only_packages_generated_cover(tmp_path):
    cache = tmp_path / "cache"
    row = _row(cache)
    audio_manifest = cache / "artifacts" / "song_describer_full.jsonl"
    write_jsonl(audio_manifest, [row])
    placeholder = placeholder_svg(row)
    placeholder_path = cache / "covers" / "placeholders" / "sdd-100.svg"
    placeholder_path.parent.mkdir(parents=True)
    placeholder_path.write_bytes(placeholder)
    cover_row = {
        "song_id": "sdd-100",
        "cover_status": "official_remote",
        "display_cover_url": jamendo_cover_url(row),
        "fallback_cover_relpath": "covers/placeholders/sdd-100.svg",
        "fallback_cover_sha256": hashlib.sha256(placeholder).hexdigest(),
        "cover_attribution": "Example Artist — Night Rooms; provided by Jamendo",
        "source_page_url": row["source_url"],
        "cover_rights_note": "remote-only",
    }
    cover_manifest = cache / "covers" / "cover_manifest_full.jsonl"
    write_jsonl(cover_manifest, [cover_row])

    result = build_public_bundle(audio_manifest, cover_manifest, cache, tmp_path / "bundle", mode="hardlink")

    bundled_audio = tmp_path / "bundle" / "audio" / "00" / "100.mp3"
    bundled_placeholder = tmp_path / "bundle" / "covers" / "placeholders" / "sdd-100.svg"
    assert bundled_audio.read_bytes() == b"original-audio-100"
    assert bundled_placeholder.read_bytes() == placeholder
    catalog = json.loads((tmp_path / "bundle" / "catalog.jsonl").read_text(encoding="utf-8"))
    assert catalog["cover_url"].startswith("https://usercontent.jamendo.com/")
    assert catalog["official_cover_packaged"] is False
    assert result["official_cover_bytes_packaged"] == 0
    assert result["audio_license_coverage"] == 1

