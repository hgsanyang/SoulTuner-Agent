"""Routing, native-format preservation and cache decode behaviour."""

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
    UcCacheDecoder,
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


@pytest.mark.parametrize("suffix", [".ncm", ".mflac", ".kgm"])
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
    # .uc and .uc! are now decoded, not protected — only .ncm stays protected.
    assert ".uc" in advertised and ".uc!" in advertised
    assert advertised.isdisjoint({".ncm"})


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


def test_wav_is_preserved_by_default(tmp_path: Path) -> None:
    src = _write_wav(tmp_path / "tone.wav")
    out = tmp_path / "out"

    result = process_audio(src, out)

    assert result.decoder == FfmpegTranscodeDecoder.name
    assert result.transcoded is False
    assert result.output_path == out / "tone.wav"
    assert result.output_path.read_bytes() == src.read_bytes()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")
def test_wav_only_transcodes_when_playback_copy_is_requested(tmp_path: Path) -> None:
    src = _write_wav(tmp_path / "tone.wav")
    result = process_audio(src, tmp_path / "out", playback_format="mp3")

    assert result.transcoded is True
    assert result.output_path == tmp_path / "out" / "tone.mp3"
    assert result.output_path.read_bytes()[:3] in (b"ID3", b"\xff\xfb")


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")
def test_corrupt_transcode_input_leaves_no_partial_output(tmp_path: Path) -> None:
    src = tmp_path / "broken.flac"
    src.write_bytes(b"fLaC" + b"\x00" * 128)
    out = tmp_path / "out"

    with pytest.raises(AudioDecodeError):
        process_audio(src, out, playback_format="mp3")

    assert not (out / "broken.mp3").exists()


# ── UcCacheDecoder tests ──────────────────────────────────────────────────


def test_uc_routes_to_uc_cache_decoder(tmp_path: Path) -> None:
    """Both .uc and .uc! suffixes resolve to UcCacheDecoder."""
    for suffix in (".uc", ".uc!"):
        src = tmp_path / f"song{suffix}"
        src.write_bytes(b"\x00" * 64)
        assert isinstance(resolve_decoder(src), UcCacheDecoder)


def test_uc_xor_roundtrip(tmp_path: Path) -> None:
    """XOR with 0xA3 is self-inverse: encoding then decoding gives back the original."""
    original = b"Hello, this is a test payload for .uc decoding!"
    encoded = bytearray([b ^ 0xA3 for b in original])

    src = tmp_path / "cache.uc"
    src.write_bytes(encoded)

    dst = tmp_path / "out" / "cache.mp3"
    (tmp_path / "out").mkdir()
    decoder = UcCacheDecoder()
    result = decoder.decode(str(src), str(dst))

    assert result is True
    assert dst.read_bytes() == original


def test_uc_empty_file(tmp_path: Path) -> None:
    """An empty cache entry is rejected and leaves no output."""
    src = tmp_path / "empty.uc"
    src.write_bytes(b"")

    dst = tmp_path / "out" / "empty.mp3"
    (tmp_path / "out").mkdir()
    decoder = UcCacheDecoder()
    with pytest.raises(AudioDecodeError, match="decoded payload is empty"):
        decoder.decode(str(src), str(dst))
    assert not dst.exists()


def test_uc_all_byte_values(tmp_path: Path) -> None:
    """Every possible byte value is XOR-ed correctly."""
    all_bytes = bytes(range(256))
    expected = bytearray([b ^ 0xA3 for b in all_bytes])

    # Encode: XOR 0xA3 on all_bytes to create the .uc file
    encoded = bytearray([b ^ 0xA3 for b in all_bytes])
    src = tmp_path / "all_bytes.uc"
    src.write_bytes(encoded)

    dst = tmp_path / "out" / "all_bytes.mp3"
    (tmp_path / "out").mkdir()
    decoder = UcCacheDecoder()
    decoder.decode(str(src), str(dst))

    # Double XOR should give back the original
    assert dst.read_bytes() == all_bytes


def test_uc_missing_input_raises(tmp_path: Path) -> None:
    """Decoding a non-existent cache entry reports the missing input."""
    decoder = UcCacheDecoder()
    with pytest.raises(FileNotFoundError):
        decoder.decode(str(tmp_path / "missing.uc"), str(tmp_path / "out.mp3"))


@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="ffprobe not on PATH")
def test_uc_process_audio_integration_preserves_underlying_wav(tmp_path: Path) -> None:
    """The final suffix comes from decoded bytes, not from the cache suffix."""
    source_audio = _write_wav(tmp_path / "source.wav")
    payload = source_audio.read_bytes()
    encoded = bytes(byte ^ 0xA3 for byte in payload)

    src = tmp_path / "track.uc"
    src.write_bytes(encoded)
    out = tmp_path / "out"

    result = process_audio(src, out)

    assert result.decoder == UcCacheDecoder.name
    assert result.source_suffix == ".uc"
    assert result.output_path == out / "track.wav"
    assert result.container == "wav"
    assert result.transcoded is False
    assert result.output_path.read_bytes() == payload


def test_uc_large_payload_is_decoded_without_changing_length(tmp_path: Path) -> None:
    original = bytes(range(256)) * 16384  # 4 MiB, larger than one decode chunk.
    src = tmp_path / "large.uc"
    src.write_bytes(original.translate(bytes(byte ^ 0xA3 for byte in range(256))))
    dst = tmp_path / "large.decoded"

    UcCacheDecoder().decode(str(src), str(dst))

    assert dst.stat().st_size == len(original)
    assert dst.read_bytes() == original


def test_uc_decode_failure_removes_partial_output(tmp_path: Path, monkeypatch) -> None:
    src = tmp_path / "broken.uc"
    src.write_bytes(b"x" * 1024)
    dst = tmp_path / "out" / "broken.decoded"

    def fail_replace(_source, _target):
        raise OSError("disk full")

    monkeypatch.setattr("services.audio_decoder.os.replace", fail_replace)
    with pytest.raises(AudioDecodeError, match="disk full"):
        UcCacheDecoder().decode(str(src), str(dst))

    assert not dst.exists()
    assert list(dst.parent.glob("*.part")) == []


@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="ffprobe not on PATH")
def test_uc_invalid_decoded_payload_is_not_published(tmp_path: Path) -> None:
    payload = b"not an audio stream" * 32
    src = tmp_path / "bad.uc"
    src.write_bytes(bytes(byte ^ 0xA3 for byte in payload))

    with pytest.raises(AudioDecodeError, match="recognised audio container"):
        process_audio(src, tmp_path / "out")

    assert list((tmp_path / "out").glob("bad.*")) == []
