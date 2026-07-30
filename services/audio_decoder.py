"""Unified audio decode routing: one entry point, many container formats.

Callers (API handlers, ingest scripts, embedder backfills) should only ever need
``process_audio(path)`` and get back a standard MP3 they can hand to librosa /
MuQ / CLaMP3. Deciding *how* a container becomes an MP3 — straight copy, ffmpeg
transcode, or "this pipeline does not open that" — belongs here rather than in
the ``suffix.lower() in SUPPORTED_AUDIO`` checks currently duplicated across
``data/pipeline/local_download_flywheel.py``, ``yt_dlp_manual_flywheel.py``,
``retrieval/data_flywheel.py`` and ``ingest_to_neo4j.py``.

Scope, per the "不做 DRM 绕过" constraint in CLAUDE.md: access-controlled
cache/download containers are *recognised but never unwrapped*. They resolve to
:class:`ProtectedContainerDecoder`, which raises with the reason and points at
the entitled acquisition path. Recognising them still matters — a file named
``.mp3`` that is really an encrypted blob otherwise reaches the feature
extractor as high-entropy noise, and a poisoned embedding is far harder to
notice than a failed ingest.

Failures raise; decoders never return ``False`` to mean "it broke". A bare
boolean erases the reason, and the reason is what the caller needs to decide
between retrying, skipping, or telling the user the track is unavailable.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Final

logger = logging.getLogger(__name__)

#: Every decoder converges on this container.
TARGET_SUFFIX: Final = ".mp3"

_TRANSCODE_BITRATE: Final = "320k"
_FFMPEG_TIMEOUT_S: Final = 600
_HEADER_BYTES: Final = 16


class AudioDecodeError(RuntimeError):
    """Decoding was attempted and failed: bad bytes, transcoder error, or I/O."""


class UnsupportedAudioFormatException(AudioDecodeError):
    """No registered decoder claims this container."""


class ProtectedContainerError(UnsupportedAudioFormatException):
    """Container is recognised, but it is access-controlled and not unwrapped here.

    Distinct from :class:`UnsupportedAudioFormatException` so callers can tell
    "we have no idea what this is" from "we know exactly what this is and the
    answer is no" — the two deserve different user-facing messages and different
    HTTP statuses.
    """


@dataclass(frozen=True, slots=True)
class DecodeResult:
    """What ``process_audio`` produced, and how."""

    output_path: Path
    decoder: str
    source_suffix: str
    transcoded: bool
    reused: bool = False


class BaseAudioDecoder(ABC):
    """Strategy interface: turn one container into a standard MP3 on disk.

    Subclasses declare what they claim two ways. ``suffixes`` is the fast path.
    ``magic`` is the fallback for files with no extension or a lying one, which
    is common for anything that has been through a download cache.
    """

    name: ClassVar[str] = "base"
    suffixes: ClassVar[frozenset[str]] = frozenset()
    magic: ClassVar[tuple[bytes, ...]] = ()

    @classmethod
    def claims_suffix(cls, suffix: str) -> bool:
        return suffix.lower() in cls.suffixes

    @classmethod
    def claims_header(cls, header: bytes) -> bool:
        return any(header.startswith(prefix) for prefix in cls.magic)

    @abstractmethod
    def decode(self, input_path: str, output_path: str) -> bool:
        """Write a standard MP3 to ``output_path``.

        Returns ``True`` when the output is on disk and non-empty. Raises
        :class:`AudioDecodeError` (or a subclass) on any failure — see the module
        docstring on why this is not signalled by a return value.
        """


class Mp3PassthroughDecoder(BaseAudioDecoder):
    """Already the target container: verify the header, then copy."""

    name = "mp3-passthrough"
    suffixes = frozenset({".mp3"})
    # ID3v2 tag, or a raw MPEG audio frame sync (11 set bits) for tagless files.
    magic = (b"ID3", b"\xff\xfb", b"\xff\xfa", b"\xff\xf3", b"\xff\xf2")

    def decode(self, input_path: str, output_path: str) -> bool:
        src, dst = _validated_paths(input_path, output_path)
        header = _read_header(src)
        if not self.claims_header(header):
            raise AudioDecodeError(
                f"{src.name} is named {TARGET_SUFFIX} but carries neither an ID3 tag "
                f"nor an MPEG frame sync (first bytes: {header[:4].hex() or 'empty'}); "
                "refusing to pass it through as audio"
            )
        if src.resolve() == dst.resolve():
            return True
        _copy(src, dst)
        return True


class FfmpegTranscodeDecoder(BaseAudioDecoder):
    """Plain (unencrypted) containers ffmpeg can read directly."""

    name = "ffmpeg-transcode"
    suffixes = frozenset({".flac", ".wav", ".m4a", ".aac", ".ogg", ".opus", ".webm"})
    magic = (b"fLaC", b"RIFF", b"OggS")

    def decode(self, input_path: str, output_path: str) -> bool:
        src, dst = _validated_paths(input_path, output_path)
        binary = shutil.which("ffmpeg")
        if binary is None:
            raise AudioDecodeError(
                f"ffmpeg not found on PATH; required to transcode {src.suffix} "
                "(the backend image installs it, local runs may not have it)"
            )
        # -vn / -map 0:a:0 drop embedded cover art, which otherwise becomes a
        # video stream and makes libmp3lame fail on an ostensibly audio-only file.
        cmd = [
            binary, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-i", str(src),
            "-vn", "-map", "0:a:0",
            "-c:a", "libmp3lame", "-b:a", _TRANSCODE_BITRATE,
            str(dst),
        ]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=_FFMPEG_TIMEOUT_S, check=False,
            )
        except subprocess.TimeoutExpired as exc:
            dst.unlink(missing_ok=True)
            raise AudioDecodeError(
                f"ffmpeg timed out after {_FFMPEG_TIMEOUT_S}s on {src.name}"
            ) from exc
        except OSError as exc:
            raise AudioDecodeError(f"cannot launch ffmpeg: {exc}") from exc

        if proc.returncode != 0:
            dst.unlink(missing_ok=True)
            detail = (proc.stderr or "").strip().splitlines()
            raise AudioDecodeError(
                f"ffmpeg failed on {src.name} (exit {proc.returncode}): "
                f"{detail[-1] if detail else 'no stderr'}"
            )
        if not dst.exists() or dst.stat().st_size == 0:
            dst.unlink(missing_ok=True)
            raise AudioDecodeError(
                f"ffmpeg reported success but wrote no audio for {src.name}"
            )
        return True


class ProtectedContainerDecoder(BaseAudioDecoder):
    """Recognises access-controlled containers and refuses them, with a reason.

    This class deliberately contains no key derivation, no cipher, and no
    byte-level unwrapping of any kind. Its whole job is to make the router fail
    informatively instead of either shrugging (``UnsupportedAudioFormat`` on a
    format we clearly recognise) or, worse, letting the blob through to the
    feature extractor.

    Per CLAUDE.md the project takes audio only from responses the account is
    actually entitled to and reports an honest failure otherwise. Registered
    ahead of the general decoders so a later ``register_decoder`` call cannot
    quietly take these suffixes over.
    """

    name = "protected-container"
    suffixes = frozenset({
        ".ncm", ".uc", ".uc!",
        ".mflac", ".mgg", ".qmc0", ".qmc3", ".qmcflac", ".qmcogg",
        ".kgm", ".kgma", ".vpr", ".xm", ".bkcmp3", ".tm0",
    })
    magic = (b"CTENFDAM",)

    def decode(self, input_path: str, output_path: str) -> bool:
        src = Path(input_path)
        raise ProtectedContainerError(
            f"{src.name}: '{src.suffix.lower() or 'unknown'}' is an access-controlled "
            "audio container and this pipeline does not unwrap it. Obtain the track "
            "through an entitled source (see services/online_audio_retention.py and "
            "codex_doc/LEGAL_AUDIO_SOURCES_FOR_AMD_DEPLOYMENT.md), then re-ingest the "
            "resulting standard file."
        )


# Order matters only for overlapping claims; the protected set stays first.
_REGISTRY: list[type[BaseAudioDecoder]] = [
    ProtectedContainerDecoder,
    Mp3PassthroughDecoder,
    FfmpegTranscodeDecoder,
]


def register_decoder(decoder_cls: type[BaseAudioDecoder]) -> None:
    """Append a decoder to the routing table.

    Appended, not prepended: existing claims win, so a new decoder cannot take
    over a suffix that is already routed.
    """
    if not issubclass(decoder_cls, BaseAudioDecoder):
        raise TypeError(f"{decoder_cls!r} is not a BaseAudioDecoder")
    if decoder_cls not in _REGISTRY:
        _REGISTRY.append(decoder_cls)


def supported_suffixes() -> frozenset[str]:
    """Suffixes that route to a decoder able to produce audio."""
    return frozenset().union(*(
        cls.suffixes for cls in _REGISTRY
        if not issubclass(cls, ProtectedContainerDecoder)
    ))


def resolve_decoder(path: str | Path) -> BaseAudioDecoder:
    """Pick a decoder by suffix, falling back to the leading magic bytes.

    Raises :class:`UnsupportedAudioFormatException` when nothing claims the file,
    :class:`FileNotFoundError` when the header fallback is needed and the file is
    absent.
    """
    src = Path(path)
    suffix = src.suffix.lower()
    for decoder_cls in _REGISTRY:
        if decoder_cls.claims_suffix(suffix):
            return decoder_cls()

    header = _read_header(_existing_file(src))
    for decoder_cls in _REGISTRY:
        if decoder_cls.claims_header(header):
            logger.info(
                "%s: suffix %r unknown, routed to %s by header %s",
                src.name, suffix or "<none>", decoder_cls.name, header[:4].hex(),
            )
            return decoder_cls()

    raise UnsupportedAudioFormatException(
        f"{src.name}: no decoder claims suffix {suffix or '<none>'} "
        f"or header {header[:4].hex() or 'empty'}; "
        f"routable audio suffixes are {sorted(supported_suffixes())}"
    )


def process_audio(
    file: str | Path,
    output_dir: str | Path | None = None,
    *,
    overwrite: bool = False,
) -> DecodeResult:
    """Produce a standard MP3 for ``file`` and report which strategy did it.

    The single entry point outward-facing code should use. ``output_dir``
    defaults to the input's own directory. With ``overwrite=False`` an existing
    target is reused rather than rebuilt, which is what makes this cheap to call
    from an idempotent ingest pass.
    """
    src = _existing_file(file)
    decoder = resolve_decoder(src)
    suffix = src.suffix.lower()

    target_dir = Path(output_dir) if output_dir is not None else src.parent
    _ensure_writable_dir(target_dir)
    dst = target_dir / f"{src.stem}{TARGET_SUFFIX}"

    same_file = dst.exists() and src.resolve() == dst.resolve()
    if dst.exists() and not overwrite and not same_file:
        logger.debug("%s: reusing existing %s", src.name, dst)
        return DecodeResult(dst, decoder.name, suffix, transcoded=False, reused=True)

    decoder.decode(str(src), str(dst))
    return DecodeResult(
        output_path=dst,
        decoder=decoder.name,
        source_suffix=suffix,
        transcoded=not isinstance(decoder, Mp3PassthroughDecoder),
    )


def _existing_file(file: str | Path) -> Path:
    src = Path(file)
    if not src.exists():
        raise FileNotFoundError(f"audio input does not exist: {src}")
    if not src.is_file():
        raise AudioDecodeError(f"audio input is not a regular file: {src}")
    return src


def _read_header(path: Path, size: int = _HEADER_BYTES) -> bytes:
    try:
        with open(path, "rb") as handle:
            return handle.read(size)
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise AudioDecodeError(f"cannot read {path}: {exc}") from exc


def _ensure_writable_dir(directory: Path) -> None:
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise AudioDecodeError(
            f"cannot create output directory {directory}: {exc}"
        ) from exc
    if not os.access(directory, os.W_OK):
        raise AudioDecodeError(f"output directory is not writable: {directory}")


def _validated_paths(input_path: str, output_path: str) -> tuple[Path, Path]:
    src = _existing_file(input_path)
    dst = Path(output_path)
    _ensure_writable_dir(dst.parent)
    return src, dst


def _copy(src: Path, dst: Path) -> None:
    try:
        shutil.copyfile(src, dst)
    except OSError as exc:
        raise AudioDecodeError(f"cannot write {dst}: {exc}") from exc
