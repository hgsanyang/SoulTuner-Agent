"""A file is what its bytes say, not what its name says.

The failure being prevented is quiet: a blob named ``.mp3`` that is really
something else passes a suffix check, reaches the feature extractor as
high-entropy noise, and produces a vector that mis-places the track forever. A
failed import is loud; a poisoned embedding is not.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from services.audio_format import (
    BROWSER_NATIVE,
    CONTAINERS,
    MetadataCandidate,
    browser_can_play,
    detect_container,
    detect_file,
    is_lossless,
    mime_for,
    suffix_for,
    suffix_matches_content,
)

# Minimal real headers.
MP3_ID3 = b"ID3\x03\x00\x00\x00\x00\x00\x00" + b"\x00" * 16
MP3_SYNC = b"\xff\xfb\x90\x64" + b"\x00" * 16
FLAC = b"fLaC\x00\x00\x00\x22" + b"\x00" * 16
OGG_VORBIS = b"OggS\x00\x02" + b"\x00" * 20 + b"\x01vorbis" + b"\x00" * 16
OGG_OPUS = b"OggS\x00\x02" + b"\x00" * 20 + b"OpusHead" + b"\x00" * 16
WAV = b"RIFF\x24\x08\x00\x00WAVEfmt " + b"\x00" * 16
M4A = b"\x00\x00\x00\x20ftypM4A " + b"\x00" * 16
AIFF = b"FORM\x00\x00\x00\x00AIFF" + b"\x00" * 16


@pytest.mark.parametrize("header,expected", [
    (MP3_ID3, "mp3"),
    (MP3_SYNC, "mp3"),
    (FLAC, "flac"),
    (OGG_VORBIS, "ogg"),
    (OGG_OPUS, "opus"),
    (WAV, "wav"),
    (M4A, "m4a"),
    (AIFF, "aiff"),
])
def test_containers_are_identified_from_their_headers(header, expected):
    assert detect_container(header) == expected


def test_opus_and_vorbis_are_distinguished_inside_the_ogg_container():
    """They share a container but need different suffixes and MIME types, so
    stopping at "ogg" would serve every Opus file under the wrong name."""
    assert detect_container(OGG_OPUS) == "opus"
    assert detect_container(OGG_VORBIS) == "ogg"
    assert suffix_for("opus") != suffix_for("ogg")


def test_a_flac_is_never_mistaken_for_an_mp3():
    """The MPEG frame sync is 11 set bits — weak enough that a container with a
    real signature has to be checked first."""
    tricky = b"fLaC\x00\x00\x00\x22\xff\xfb\x90\x64" + b"\x00" * 16
    assert detect_container(tricky) == "flac"


@pytest.mark.parametrize("junk", [
    b"", b"\x00", b"\x00\x00\x00", b"not audio at all", b"\x89PNG\r\n\x1a\n",
    b"PK\x03\x04" + b"\x00" * 16,
])
def test_non_audio_is_not_forced_into_a_format(junk):
    """Returning a guess here is how a blob reaches the embedder."""
    assert detect_container(junk) == ""


def test_a_reserved_mpeg_layer_is_not_accepted():
    """0xFF followed by three set bits is common in binary data; the version and
    layer fields are what make it actually an MPEG frame."""
    assert detect_container(b"\xff\xe8\x00\x00" + b"\x00" * 16) == ""   # layer 00
    assert detect_container(b"\xff\xea\x00\x00" + b"\x00" * 16) == ""   # version 01


# ---- serving ----------------------------------------------------------------

def test_every_known_container_has_a_suffix_and_a_mime():
    for container in CONTAINERS:
        assert suffix_for(container).startswith(".")
        assert "/" in mime_for(container)


def test_an_unknown_container_yields_no_mime_rather_than_a_plausible_wrong_one():
    """Serving FLAC as audio/mpeg makes some browsers refuse it outright."""
    assert mime_for("nonsense") == ""
    assert suffix_for("nonsense") == ""


def test_flac_is_not_transcoded_for_compatibility():
    """Chrome, Firefox and Edge have played FLAC for years. Converting it to MP3
    in the name of compatibility would discard the lossless data for nothing."""
    assert browser_can_play("flac") is True
    assert is_lossless("flac") is True


def test_lossless_flags_are_right():
    assert is_lossless("wav") and is_lossless("aiff")
    assert not is_lossless("mp3") and not is_lossless("ogg")


def test_browser_native_set_is_a_subset_of_known_containers():
    assert BROWSER_NATIVE <= set(CONTAINERS)


# ---- files on disk ----------------------------------------------------------

def test_detect_file_reads_the_header(tmp_path):
    path = tmp_path / "whatever.bin"
    path.write_bytes(FLAC)
    assert detect_file(path) == "flac"


def test_a_missing_file_is_unknown_not_an_exception(tmp_path):
    assert detect_file(tmp_path / "nope.mp3") == ""


def test_a_mislabelled_extension_is_caught(tmp_path):
    """The exact case that matters: FLAC bytes under an .mp3 name."""
    path = tmp_path / "song.mp3"
    path.write_bytes(FLAC)
    assert detect_file(path) == "flac"
    assert suffix_matches_content(path) is False


def test_a_correctly_named_file_passes(tmp_path):
    path = tmp_path / "song.flac"
    path.write_bytes(FLAC)
    assert suffix_matches_content(path) is True


def test_an_unrecognisable_file_never_reports_a_matching_suffix(tmp_path):
    path = tmp_path / "song.mp3"
    path.write_bytes(b"this is not audio")
    assert suffix_matches_content(path) is False


# ---- metadata candidate -----------------------------------------------------

def test_setting_a_field_records_where_it_came_from():
    candidate = MetadataCandidate()
    candidate.set("title", "海阔天空", source="embedded_tag", confidence=0.9)
    assert candidate.value("title") == "海阔天空"
    assert candidate.fields["title"].source == "embedded_tag"
    assert candidate.fields["title"].confidence == 0.9


def test_empty_values_are_not_recorded():
    """An empty tag must not outrank a later source that actually has a value."""
    candidate = MetadataCandidate()
    for empty in ("", None, [], 0):
        candidate.set("title", empty, source="embedded_tag")
    assert candidate.fields == {}
    assert candidate.value("title", "fallback") == "fallback"


def test_read_metadata_survives_a_file_that_is_not_audio(tmp_path):
    from services.audio_format import read_metadata

    path = tmp_path / "broken.mp3"
    path.write_bytes(b"definitely not audio")
    candidate = read_metadata(path)
    assert candidate.container == ""
    assert candidate.fields == {}


def test_read_metadata_survives_an_empty_file(tmp_path):
    from services.audio_format import read_metadata

    path = tmp_path / "empty.mp3"
    path.write_bytes(b"")
    assert read_metadata(path).container == ""


def test_timestamped_lyrics_are_kept_in_both_forms():
    """The player wants timings; search wants plain text. Keeping only one
    means re-deriving the other later, badly."""
    from services.audio_format import _strip_timestamps

    lrc = "[00:00.00]第一行\n[00:05.00]第二行"
    assert _strip_timestamps(lrc) == "第一行\n第二行"


def test_artist_splitting_handles_the_separators_that_actually_occur():
    from services.audio_format import _split_artists

    assert _split_artists("A/B") == ["A", "B"]
    assert _split_artists("A、B") == ["A", "B"]
    assert _split_artists("A; B") == ["A", "B"]
    assert _split_artists("单人") == ["单人"]
    assert _split_artists("") == []


# ---- real files, produced by ffmpeg ----------------------------------------
#
# Everything above feeds hand-built byte strings to detect_container. That is
# useful for pinning the ordering rules, and it is also how a real bug survived:
# the Opus fixture put OpusHead inside the first 64 bytes, while a real Ogg page
# header is 27 bytes plus a segment table and pushes it to offset 28. detect_file
# read 16 bytes, so every real Opus file was detected as plain Ogg — wrong
# suffix, wrong MIME — and the suite stayed green.
#
# These tests read files ffmpeg actually encoded, through the on-disk path.

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "audio"


def _fixture(name: str) -> Path:
    path = FIXTURES / name
    if not path.exists():
        pytest.skip(f"fixture missing: {name}")
    return path


@pytest.mark.parametrize("name,expected", [
    ("tone.mp3", "mp3"),
    ("tone.flac", "flac"),
    ("tone.ogg", "ogg"),
    ("tone.opus", "opus"),
    ("tone.wav", "wav"),
    ("tone.m4a", "m4a"),
])
def test_real_encoded_files_are_detected_through_the_disk_path(name, expected):
    assert detect_file(_fixture(name)) == expected


def test_a_real_opus_file_is_not_reported_as_plain_ogg():
    """The regression this file exists for. OpusHead is at byte 28 in real
    output, so any header window shorter than that silently loses the codec."""
    path = _fixture("tone.opus")
    assert b"OpusHead" not in path.read_bytes()[:16]     # the trap
    assert detect_file(path) == "opus"                    # and it still works
    assert suffix_for("opus") == ".opus"
    assert mime_for("opus") == "audio/opus"


def test_every_real_fixture_has_a_matching_suffix():
    for path in sorted(FIXTURES.glob("tone.*")):
        assert suffix_matches_content(path), f"{path.name} 内容与扩展名不符"


def test_real_files_probe_to_a_plausible_duration():
    """ffprobe is optional on the host; when present it must agree with the
    one-second tone these fixtures encode."""
    from services.audio_format import probe

    stream = probe(_fixture("tone.flac"))
    if not stream:
        pytest.skip("ffprobe unavailable on this host")
    assert 800 <= stream["duration_ms"] <= 1200
    assert stream["sample_rate"] > 0


def test_read_metadata_reports_real_container_facts():
    from services.audio_format import read_metadata

    candidate = read_metadata(_fixture("tone.flac"))
    assert candidate.container == "flac"
    assert candidate.lossless is True


def test_tags_written_by_mutagen_are_read_back(tmp_path):
    """Proves the tag reader actually reads tags, rather than merely not
    crashing on a file that has none."""
    mutagen = pytest.importorskip("mutagen")
    from mutagen.flac import FLAC, Picture

    src = _fixture("tone.flac")
    work = tmp_path / "tagged.flac"
    work.write_bytes(src.read_bytes())

    audio = FLAC(str(work))
    audio["title"] = "海阔天空"
    audio["artist"] = "Beyond"
    audio["album"] = "乐与怒"
    audio["date"] = "1993"
    audio["lyrics"] = "[00:00.00]今天我"
    picture = Picture()
    picture.type = 3
    picture.mime = "image/jpeg"
    picture.data = b"\xff\xd8\xff\xe0" + b"cover-bytes" * 8
    audio.add_picture(picture)
    audio.save()

    from services.audio_format import read_metadata

    candidate = read_metadata(work)
    assert candidate.value("title") == "海阔天空"
    assert candidate.value("artists") == ["Beyond"]
    assert candidate.value("album") == "乐与怒"
    assert str(candidate.value("year")).startswith("1993")
    assert candidate.cover_bytes.startswith(b"\xff\xd8\xff")
    # a timestamped line has to survive in both forms
    assert candidate.synced_lyrics.startswith("[00:00.00]")
    assert candidate.plain_lyrics == "今天我"
    assert candidate.fields["title"].source == "embedded_tag"


def test_id3_tags_on_a_real_mp3_are_read_back(tmp_path):
    pytest.importorskip("mutagen")
    from mutagen.id3 import APIC, ID3, TALB, TIT2, TPE1

    src = _fixture("tone.mp3")
    work = tmp_path / "tagged.mp3"
    work.write_bytes(src.read_bytes())

    tags = ID3()
    tags.add(TIT2(encoding=3, text="Yellow"))
    tags.add(TPE1(encoding=3, text="Coldplay"))
    tags.add(TALB(encoding=3, text="Parachutes"))
    tags.add(APIC(encoding=3, mime="image/jpeg", type=3, desc="",
                  data=b"\xff\xd8\xff\xe0" + b"jpegdata" * 4))
    tags.save(str(work))

    from services.audio_format import read_metadata

    candidate = read_metadata(work)
    assert candidate.value("title") == "Yellow"
    assert candidate.value("artists") == ["Coldplay"]
    assert candidate.value("album") == "Parachutes"
    assert candidate.cover_bytes.startswith(b"\xff\xd8\xff")


# ---- process_audio must not destroy what it routes -------------------------
#
# It used to name every output .mp3 and re-encode anything that was not already
# MP3. Fed a FLAC that meant: lossy re-encode, then store the result under a
# name claiming it is an MP3 — the same defect found in the live catalogue,
# except produced systematically at the entry point.

def test_a_flac_survives_routing_as_a_flac(tmp_path):
    from services.audio_decoder import process_audio

    src = tmp_path / "song.flac"
    src.write_bytes(_fixture("tone.flac").read_bytes())
    out = tmp_path / "out"
    result = process_audio(src, out)

    assert result.output_path.suffix == ".flac"
    assert result.container == "flac"
    assert result.mime_type == "audio/flac"
    assert result.lossless is True
    assert result.transcoded is False
    # byte-identical: preserved, not re-encoded
    assert result.output_path.read_bytes() == src.read_bytes()


def test_an_mp3_stays_an_mp3(tmp_path):
    from services.audio_decoder import process_audio

    src = tmp_path / "song.mp3"
    src.write_bytes(_fixture("tone.mp3").read_bytes())
    result = process_audio(src, tmp_path / "out")
    assert result.output_path.suffix == ".mp3"
    assert result.container == "mp3"
    assert result.lossless is False


def test_a_mislabelled_flac_is_written_under_its_real_extension(tmp_path):
    """The production defect, at the routing layer: bytes are FLAC, name says
    mp3. The output must follow the bytes."""
    from services.audio_decoder import process_audio

    src = tmp_path / "liar.mp3"
    src.write_bytes(_fixture("tone.flac").read_bytes())
    result = process_audio(src, tmp_path / "out")
    assert result.output_path.suffix == ".flac"
    assert result.mime_type == "audio/flac"


def test_asking_for_mp3_playback_is_explicit_not_implicit(tmp_path):
    """A caller that needs a derived MP3 says so. It is never imposed."""
    from services.audio_decoder import process_audio

    src = tmp_path / "song.flac"
    src.write_bytes(_fixture("tone.flac").read_bytes())
    try:
        result = process_audio(src, tmp_path / "out", playback_format="mp3")
    except Exception as exc:                 # ffmpeg absent on this host
        pytest.skip(f"transcode unavailable: {type(exc).__name__}")
    assert result.output_path.suffix == ".mp3"
    assert result.transcoded is True


@pytest.mark.parametrize("name", ["tone.ogg", "tone.opus", "tone.wav", "tone.m4a"])
def test_every_real_format_keeps_its_own_container(tmp_path, name):
    from services.audio_decoder import process_audio

    src = tmp_path / name
    src.write_bytes(_fixture(name).read_bytes())
    result = process_audio(src, tmp_path / "out")
    assert result.output_path.suffix == Path(name).suffix
    assert result.transcoded is False
