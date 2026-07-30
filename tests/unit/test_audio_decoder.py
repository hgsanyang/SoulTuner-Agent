"""Routing behaviour for services.audio_decoder.

The interesting cases are the negative ones: a lying suffix must fail loudly
rather than reach the feature extractor, and a recognised-but-refused container
must be distinguishable from a genuinely unknown one.
"""

from __future__ import annotations

import shutil
import struct
import wave
from pathlib import Path

import pytest

from services.audio_decoder import (
    AudioDecodeError,
    FfmpegTranscodeDecoder,
    Mp3PassthroughDecoder,
    ProtectedContainerDecoder,
    ProtectedContainerError,
    UnsupportedAudioFormatException,
    process_audio,
    resolve_decoder,
    supported_suffixes,
)

# Minimal ID3v2.3 header (10 bytes, zero-length tag body) + one MPEG frame sync.
_FAKE_MP3 = b"ID3\x03\x00\x00\x00\x00\x00\x00" + b"\xff\xfb\x90\x00" + b"\x00" * 256


def _write_wav(path: Path, seconds: float = 0.2, rate: int = 8000) -> Path:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"".join(struct.pack("<h", 0) for _ in range(int(rate * seconds))))
    return path


def test_mp3_routes_to_passthrough_and_copies(tmp_path: Path) -> None:
    src = tmp_path / "track.mp3"
    src.write_bytes(_FAKE_MP3)
    out = tmp_path / "out"

    result = process_audio(src, out)

    assert result.decoder == Mp3PassthroughDecoder.name
    assert result.transcoded is False
    assert result.output_path == out / "track.mp3"
    assert result.output_path.read_bytes() == _FAKE_MP3


def test_mp3_in_place_does_not_truncate_itself(tmp_path: Path) -> None:
    src = tmp_path / "track.mp3"
    src.write_bytes(_FAKE_MP3)

    result = process_audio(src)

    assert result.output_path.resolve() == src.resolve()
    assert src.read_bytes() == _FAKE_MP3


def test_existing_output_is_reused_unless_overwrite(tmp_path: Path) -> None:
    src = tmp_path / "track.mp3"
    src.write_bytes(_FAKE_MP3)
    out = tmp_path / "out"
    out.mkdir()
    (out / "track.mp3").write_bytes(b"stale")

    reused = process_audio(src, out)
    assert reused.reused is True
    assert (out / "track.mp3").read_bytes() == b"stale"

    rebuilt = process_audio(src, out, overwrite=True)
    assert rebuilt.reused is False
    assert (out / "track.mp3").read_bytes() == _FAKE_MP3


@pytest.mark.parametrize("suffix", [".uc", ".uc!", ".ncm", ".mflac", ".kgm"])
def test_protected_containers_refuse_with_a_reason(tmp_path: Path, suffix: str) -> None:
    src = tmp_path / f"cached{suffix}"
    src.write_bytes(b"\x00" * 64)

    assert isinstance(resolve_decoder(src), ProtectedContainerDecoder)
    with pytest.raises(ProtectedContainerError) as excinfo:
        process_audio(src, tmp_path / "out")

    assert suffix in str(excinfo.value)
    # Callers that only care "can I use this file" catch the broader class.
    assert isinstance(excinfo.value, UnsupportedAudioFormatException)
    assert not (tmp_path / "out" / "cached.mp3").exists()


def test_protected_suffixes_are_not_advertised_as_supported() -> None:
    advertised = supported_suffixes()
    assert ".mp3" in advertised and ".flac" in advertised
    assert advertised.isdisjoint({".uc", ".uc!", ".ncm"})


def test_encrypted_blob_wearing_an_mp3_name_fails_loudly(tmp_path: Path) -> None:
    """The poisoned-embedding case: suffix says mp3, bytes say otherwise."""
    src = tmp_path / "liar.mp3"
    src.write_bytes(b"CTENFDAM" + b"\xde\xad\xbe\xef" * 16)

    with pytest.raises(AudioDecodeError) as excinfo:
        process_audio(src, tmp_path / "out")

    assert "frame sync" in str(excinfo.value)
    assert not (tmp_path / "out" / "liar.mp3").exists()


def test_header_fallback_routes_extensionless_file(tmp_path: Path) -> None:
    src = tmp_path / "no_extension_here"
    src.write_bytes(b"fLaC" + b"\x00" * 32)

    assert isinstance(resolve_decoder(src), FfmpegTranscodeDecoder)


def test_unknown_format_raises_unsupported(tmp_path: Path) -> None:
    src = tmp_path / "notes.xyz"
    src.write_bytes(b"plain text, not audio")

    with pytest.raises(UnsupportedAudioFormatException):
        process_audio(src, tmp_path / "out")


def test_missing_input_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        process_audio(tmp_path / "gone.mp3")


def test_directory_input_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "a_directory.mp3"
    target.mkdir()

    with pytest.raises(AudioDecodeError):
        process_audio(target)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")
def test_wav_is_transcoded_to_mp3(tmp_path: Path) -> None:
    src = _write_wav(tmp_path / "tone.wav")
    out = tmp_path / "out"

    result = process_audio(src, out)

    assert result.decoder == FfmpegTranscodeDecoder.name
    assert result.transcoded is True
    assert result.output_path == out / "tone.mp3"
    assert result.output_path.stat().st_size > 0
    assert result.output_path.read_bytes()[:3] in (b"ID3", b"\xff\xfb")


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")
def test_corrupt_transcode_input_leaves_no_partial_output(tmp_path: Path) -> None:
    src = tmp_path / "broken.flac"
    src.write_bytes(b"fLaC" + b"\x00" * 128)
    out = tmp_path / "out"

    with pytest.raises(AudioDecodeError):
        process_audio(src, out)

    assert not (out / "broken.mp3").exists()
