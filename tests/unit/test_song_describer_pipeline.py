from __future__ import annotations

import hashlib
import json
import zipfile

import pytest

from tools.data.song_describer_pipeline import (
    RemoteFile,
    build_audit,
    build_manifest_rows,
    download_file,
    extract_selected_audio,
    hash_file,
    parse_license_url,
    read_licenses,
    write_jsonl,
)
from tools.data.queue_song_describer import load_ingest_songs, manifest_row_to_ingest_song


def _write_fixture(root):
    metadata = root / "metadata"
    metadata.mkdir()
    (metadata / "song_describer.csv").write_text(
        "caption_id,track_id,caption,is_valid_subset,familiarity,artist_id,album_id,path,duration\n"
        '2,100,"Second caption, with comma",True,3,7,8,00/100.mp3,120.0\n'
        "1,100,First caption,True,2,7,8,00/100.mp3,120.0\n"
        "3,200,Unvalidated only,,1,9,10,00/200.mp3,60.0\n",
        encoding="utf-8",
    )
    (metadata / "audio_metadata.tsv").write_text(
        "track_0000100\tartist_000007\talbum_000008\tQuiet Rain\tExample Artist\tNight Rooms\t2020-01-02\thttp://www.jamendo.com/track/100\n"
        "track_0000200\tartist_000009\talbum_000010\tOther\tOther Artist\tOther Album\t2021-01-02\thttp://www.jamendo.com/track/200\n",
        encoding="utf-8",
    )
    (metadata / "song_describer_14_04_23.mtg-jamendo.tsv").write_text(
        "track_0000100\tartist_000007\talbum_000008\t00/100.mp3\t120.0\tgenre---ambient\tinstrument---piano\tmood/theme---calm\n"
        "track_0000200\tartist_000009\talbum_000010\t00/200.mp3\t60.0\tgenre---rock\n",
        encoding="utf-8",
    )
    (metadata / "audio_licenses.txt").write_text(
        "00/100.mp3\n"
        "Quiet Rain by Example Artist from Jamendo: http://www.jamendo.com/track/100\n"
        "Available under a Creative Commons Attribution-Non-Commercial-Share-Alike license: http://creativecommons.org/licenses/by-nc-sa/3.0/\n"
        "--\n"
        "00/200.mp3\n"
        "Other by Other Artist from Jamendo: http://www.jamendo.com/track/200\n"
        "Available under a Creative Commons Attribution license: http://creativecommons.org/licenses/by/3.0/\n"
        "--\n",
        encoding="utf-8",
    )
    return metadata


def test_parse_license_url_preserves_restrictions_and_jurisdiction():
    licence = parse_license_url("http://creativecommons.org/licenses/by-nc-nd/2.0/fr/")

    assert licence["id"] == "CC-BY-NC-ND-2.0-FR"
    assert licence["noncommercial_only"] is True
    assert licence["no_derivatives"] is True
    assert licence["share_alike"] is False
    assert licence["commercial_demo_allowed"] is False
    assert licence["transformations_allowed"] is False


def test_parse_license_url_supports_licence_art_libre():
    licence = parse_license_url("http://artlibre.org/licence/lal/")

    assert licence["id"] == "LICENCE-ART-LIBRE"
    assert licence["commercial_demo_allowed"] is True
    assert licence["transformations_allowed"] is True
    assert licence["share_alike"] is True


def test_read_licenses_keeps_verbatim_attribution(tmp_path):
    metadata = _write_fixture(tmp_path)

    rows = read_licenses(metadata / "audio_licenses.txt")

    assert rows["100"]["audio_relpath"] == "00/100.mp3"
    assert rows["100"]["attribution_text"].startswith("Quiet Rain by Example Artist")
    assert rows["100"]["id"] == "CC-BY-NC-SA-3.0"


def test_validated_manifest_joins_captions_metadata_license_and_audio_sha(tmp_path):
    metadata = _write_fixture(tmp_path)
    audio_root = tmp_path / "audio"
    (audio_root / "00").mkdir(parents=True)
    audio = audio_root / "00" / "100.mp3"
    audio.write_bytes(b"test-audio")

    rows = build_manifest_rows(metadata, audio_root=audio_root, subset="validated")

    assert len(rows) == 1
    row = rows[0]
    assert row["song_id"] == "sdd-100"
    assert row["title"] == "Quiet Rain"
    assert row["artist"] == "Example Artist"
    assert row["genres"] == ["ambient"]
    assert row["instruments"] == ["piano"]
    assert row["moods_themes"] == ["calm"]
    assert [caption["caption_id"] for caption in row["captions"]] == ["1", "2"]
    assert row["audio_available"] is True
    assert row["audio_sha256"] == hash_file(audio)
    assert row["attribution"].startswith("Quiet Rain by Example Artist")
    assert row["license_url"] == "http://creativecommons.org/licenses/by-nc-sa/3.0/"
    assert row["source_url"] == "http://www.jamendo.com/track/100"
    assert row["usage_policy"]["planner_training_allowed"] is False
    assert row["usage_policy"]["audio_delivery_mode"] == "original_archive_bytes_only"
    assert row["usage_policy"]["muq_feature_generation_only"] is True
    assert row["usage_policy"]["muq_rewrites_audio"] is False


def test_manifest_and_audit_are_bound_by_sha256(tmp_path):
    metadata = _write_fixture(tmp_path)
    rows = build_manifest_rows(metadata, audio_root=tmp_path / "audio", subset="validated")
    manifest = tmp_path / "manifest.jsonl"

    write_jsonl(manifest, rows)
    audit = build_audit(rows, manifest_path=manifest)

    assert audit["track_count"] == 1
    assert audit["caption_count"] == 2
    assert audit["audio_missing"] == 1
    assert audit["noncommercial_only_tracks"] == 1
    assert audit["unknown_or_missing_licenses"] == 0
    assert audit["provenance_coverage"] == {"attribution": 1, "license_url": 1, "source_url": 1}
    assert audit["manifest_sha256"] == hash_file(manifest)
    assert json.loads(manifest.read_text(encoding="utf-8"))["track_id"] == "100"


def test_extract_selected_audio_rejects_path_traversal(tmp_path):
    archive_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../escape.mp3", b"nope")

    with pytest.raises(ValueError, match="unsafe path"):
        extract_selected_audio(archive_path, tmp_path / "audio", ["00/100.mp3"])


def test_extract_selected_audio_only_materialises_wanted_tracks(tmp_path):
    archive_path = tmp_path / "audio.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("audio/00/100.mp3", b"wanted")
        archive.writestr("audio/00/200.mp3", b"not-wanted")

    count = extract_selected_audio(archive_path, tmp_path / "audio", ["00/100.mp3"])

    assert count == 1
    assert (tmp_path / "audio" / "00" / "100.mp3").read_bytes() == b"wanted"
    assert not (tmp_path / "audio" / "00" / "200.mp3").exists()


def test_download_file_resumes_clean_early_eof_until_checksum_matches(tmp_path, monkeypatch):
    payload = b"abcdefghijklmnopqrstuvwxyz"
    calls: list[int] = []

    class Response:
        def __init__(self, body: bytes, status: int):
            self.body = body
            self.status = status

        def getcode(self):
            return self.status

        def read(self, _size=-1):
            body, self.body = self.body, b""
            return body

        def close(self):
            return None

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

    def fake_urlopen(request, timeout):
        assert timeout == 120
        range_header = request.get_header("Range")
        offset = int(range_header.removeprefix("bytes=").removesuffix("-")) if range_header else 0
        calls.append(offset)
        return Response(payload[offset : offset + 5], 206 if offset else 200)

    monkeypatch.setattr("tools.data.song_describer_pipeline.urlopen", fake_urlopen)
    monkeypatch.setattr("tools.data.song_describer_pipeline.time.sleep", lambda _seconds: None)
    remote = RemoteFile(
        name="chunked.bin",
        size=len(payload),
        checksum_type="sha256",
        checksum=hashlib.sha256(payload).hexdigest(),
        url="https://example.test/chunked.bin",
    )

    result = download_file(remote, tmp_path / remote.name)

    assert result.read_bytes() == payload
    assert calls == [0, 5, 10, 15, 20, 25]


def test_no_derivatives_track_is_original_bytes_only(tmp_path):
    metadata = _write_fixture(tmp_path)
    # Make the validated fixture track ND and prove extraction is a byte copy.
    licence_path = metadata / "audio_licenses.txt"
    licence_path.write_text(
        licence_path.read_text(encoding="utf-8").replace(
            "by-nc-sa/3.0/", "by-nc-nd/3.0/"
        ),
        encoding="utf-8",
    )
    original = b"original-mp3-container-bytes"
    archive_path = tmp_path / "audio.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("audio/00/100.2min.mp3", original)
    audio_root = tmp_path / "audio"

    extract_selected_audio(archive_path, audio_root, ["00/100.mp3"])
    rows = build_manifest_rows(metadata, audio_root=audio_root, subset="validated")

    assert (audio_root / "00" / "100.mp3").read_bytes() == original
    assert rows[0]["usage_policy"]["no_derivatives"] is True
    assert rows[0]["usage_policy"]["audio_transforms_allowed"] is False
    assert rows[0]["usage_policy"]["audio_delivery_mode"] == "original_archive_bytes_only"


def test_manifest_row_maps_to_existing_ingest_contract(tmp_path):
    metadata = _write_fixture(tmp_path)
    audio_root = tmp_path / "audio"
    (audio_root / "00").mkdir(parents=True)
    (audio_root / "00" / "100.mp3").write_bytes(b"real-enough-for-contract-test")
    row = build_manifest_rows(metadata, audio_root=audio_root, subset="validated")[0]

    song = manifest_row_to_ingest_song(row, tmp_path)

    assert song["song_id"] == "sdd-100"
    assert song["source_id"] == "100"
    assert song["audio_url"] == "/static/mtg_audio/00/100.mp3"
    assert song["audio_path"].endswith("100.mp3")
    assert song["file_basename"] == "100"
    assert song["audio_retention"] == "saved"
    assert song["requested_by"].startswith("explicit")
    assert song["tagging_mode"] == "deferred"
    assert song["audio_license"]["id"] == "CC-BY-NC-SA-3.0"


def test_public_catalog_cover_is_preserved_in_ingest_contract(tmp_path):
    metadata = _write_fixture(tmp_path)
    audio_root = tmp_path / "audio"
    (audio_root / "00").mkdir(parents=True)
    (audio_root / "00" / "100.mp3").write_bytes(b"real-enough-for-contract-test")
    row = build_manifest_rows(metadata, audio_root=audio_root, subset="validated")[0]
    row["cover_url"] = "https://usercontent.jamendo.com/cover.jpg"

    song = manifest_row_to_ingest_song(row, tmp_path)

    assert song["cover_url"] == "https://usercontent.jamendo.com/cover.jpg"


def test_load_ingest_songs_fails_closed_on_audio_tampering(tmp_path):
    metadata = _write_fixture(tmp_path)
    audio_root = tmp_path / "audio"
    (audio_root / "00").mkdir(parents=True)
    audio = audio_root / "00" / "100.mp3"
    audio.write_bytes(b"original")
    rows = build_manifest_rows(metadata, audio_root=audio_root, subset="validated")
    manifest = tmp_path / "manifest.jsonl"
    write_jsonl(manifest, rows)
    audio.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_ingest_songs(manifest, tmp_path)
