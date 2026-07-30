"""A file is what its bytes say, not what its name says.

The failure being prevented is quiet: a blob named ``.mp3`` that is really
something else passes a suffix check, reaches the feature extractor as
high-entropy noise, and produces a vector that mis-places the track forever. A
failed import is loud; a poisoned embedding is not.
"""

from __future__ import annotations

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
