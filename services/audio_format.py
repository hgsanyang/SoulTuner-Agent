"""What a file actually is, as opposed to what its name says.

Three jobs, all of which the catalogue already needs and none of which involve
converting anything:

* :func:`detect_container` — read the leading bytes and name the container.
* :func:`mime_for` / :func:`suffix_for` — serve it with headers a browser
  believes, and store it under an extension that matches its contents.
* :func:`read_metadata` — pull embedded tags, cover art and lyrics *before* any
  transcode, because a transcode is where they get lost.

Why this is not a detail: the library already holds 17 FLAC files alongside 1883
MP3s, and the ingest path copies files through with ``shutil.copy2`` rather than
transcoding. So multi-format is the existing reality, not a future feature. A
layer that assumes ``.mp3`` would be introducing the bottleneck, not removing it.

The failure this prevents is specific and quiet: a file whose extension says
``.mp3`` but whose bytes are something else reaches the feature extractor as
high-entropy noise. The ingest succeeds, a vector gets written, and the track is
then permanently mis-placed in the similarity space. A wrong embedding is far
harder to notice than a failed import.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

logger = logging.getLogger(__name__)

_HEADER_BYTES: Final = 16

#: container -> (canonical suffix, mime, lossless)
CONTAINERS: Final[dict[str, tuple[str, str, bool]]] = {
    "mp3": (".mp3", "audio/mpeg", False),
    "flac": (".flac", "audio/flac", True),
    "ogg": (".ogg", "audio/ogg", False),
    "opus": (".opus", "audio/opus", False),
    "wav": (".wav", "audio/wav", True),
    "m4a": (".m4a", "audio/mp4", False),
    "aiff": (".aiff", "audio/aiff", True),
    "wma": (".wma", "audio/x-ms-wma", False),
}

#: Formats every current browser can play from a plain <audio> element.
#: FLAC is here deliberately: Chrome, Firefox and Edge have supported it for
#: years, so transcoding it to MP3 for "compatibility" would discard the
#: lossless data for nothing. Safari is the holdout, hence the derived-playback
#: path existing at all.
BROWSER_NATIVE: Final[frozenset[str]] = frozenset({"mp3", "flac", "ogg", "opus", "wav", "m4a"})


@dataclass
class MetadataField:
    """One value plus where it came from, so a later merge can rank sources."""

    value: Any
    source: str = "unknown"
    confidence: float = 0.0


@dataclass
class MetadataCandidate:
    """Everything readable from the file itself, before any conversion."""

    container: str = ""
    codec: str = ""
    sample_rate: int = 0
    channels: int = 0
    bitrate: int = 0
    duration_ms: int = 0
    lossless: bool = False
    fields: dict[str, MetadataField] = field(default_factory=dict)
    cover_bytes: bytes = b""
    plain_lyrics: str = ""
    synced_lyrics: str = ""

    def value(self, name: str, default: Any = "") -> Any:
        entry = self.fields.get(name)
        return entry.value if entry is not None else default

    def set(self, name: str, value: Any, *, source: str, confidence: float = 0.9) -> None:
        if value in (None, "", [], 0):
            return
        self.fields[name] = MetadataField(value=value, source=source, confidence=confidence)


def detect_container(data: bytes) -> str:
    """Name the container from its leading bytes, or "" when unrecognised.

    Order matters. An MP3 frame sync is only two bits shy of arbitrary, so every
    container with a distinctive signature is checked first — otherwise a FLAC
    file whose header happens to contain 0xFF 0xFB would be called an MP3.
    """
    if len(data) < 4:
        return ""
    if data[:4] == b"fLaC":
        return "flac"
    if data[:4] == b"OggS":
        # Opus and Vorbis share the Ogg container; the codec name sits in the
        # first packet. Reading it matters because they need different suffixes.
        head = data[:64]
        if b"OpusHead" in head:
            return "opus"
        return "ogg"
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return "wav"
    if data[:4] == b"FORM" and data[8:12] in (b"AIFF", b"AIFC"):
        return "aiff"
    if data[4:8] == b"ftyp":
        return "m4a"
    if data[:3] == b"ID3":
        return "mp3"
    if data[:4] == b"0&\xb2u" or data[:4] == b"\x30\x26\xb2\x75":
        return "wma"
    # MPEG frame sync: 11 set bits, then a layer/version that is not "reserved".
    if data[0] == 0xFF and (data[1] & 0xE0) == 0xE0:
        version = (data[1] >> 3) & 0x03
        layer = (data[1] >> 1) & 0x03
        if version != 0x01 and layer != 0x00:
            return "mp3"
    return ""


def detect_file(path: str | Path) -> str:
    """Container of a file on disk, or "" if unreadable/unrecognised."""
    try:
        with open(path, "rb") as handle:
            return detect_container(handle.read(_HEADER_BYTES))
    except OSError as exc:
        logger.warning("[audio-format] 读取文件头失败 %s: %s", path, type(exc).__name__)
        return ""


def suffix_for(container: str) -> str:
    return CONTAINERS.get(container, ("", "", False))[0]


def mime_for(container: str) -> str:
    """MIME to serve this container as. Empty when unknown — never guessed.

    Serving FLAC as ``audio/mpeg`` makes some browsers refuse it outright, so an
    honest empty result that the caller handles beats a plausible wrong one.
    """
    return CONTAINERS.get(container, ("", "", False))[1]


def is_lossless(container: str) -> bool:
    return CONTAINERS.get(container, ("", "", False))[2]


def browser_can_play(container: str) -> bool:
    """Whether a derived playback copy is needed at all."""
    return container in BROWSER_NATIVE


def suffix_matches_content(path: str | Path) -> bool:
    """True when the extension agrees with the bytes.

    A mismatch is the case worth catching: it is the shape that lets a non-audio
    blob through a suffix check and into the embedder.
    """
    container = detect_file(path)
    if not container:
        return False
    return Path(path).suffix.lower() == suffix_for(container)


def probe(path: str | Path, *, timeout: int = 60) -> dict[str, Any]:
    """ffprobe the stream. Returns {} when ffprobe is absent or the file is bad.

    Empty means "could not verify", which callers must treat as a failure to
    validate rather than as a valid file with unknown properties.
    """
    import json
    import shutil

    if not shutil.which("ffprobe"):
        return {}
    try:
        completed = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "a:0",
                "-show_entries",
                "stream=codec_name,sample_rate,channels,bit_rate:format=duration,format_name",
                "-of", "json", str(path),
            ],
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        logger.warning("[audio-format] ffprobe 失败 %s: %s", path, type(exc).__name__)
        return {}
    if completed.returncode != 0:
        return {}
    try:
        payload = json.loads(completed.stdout or b"{}")
    except json.JSONDecodeError:
        return {}
    streams = payload.get("streams") or [{}]
    stream = streams[0] if streams else {}
    fmt = payload.get("format") or {}
    duration = fmt.get("duration")
    return {
        "codec": str(stream.get("codec_name") or ""),
        "sample_rate": int(stream.get("sample_rate") or 0),
        "channels": int(stream.get("channels") or 0),
        "bitrate": int(stream.get("bit_rate") or 0),
        "duration_ms": int(float(duration) * 1000) if duration else 0,
        "format_name": str(fmt.get("format_name") or ""),
    }


# ------------------------------------------------------------------ tags ----

_TEXT_KEYS = {
    "title": ("TIT2", "\xa9nam", "title", "TITLE", "INAM"),
    "album": ("TALB", "\xa9alb", "album", "ALBUM", "IPRD"),
    "album_artist": ("TPE2", "aART", "albumartist", "ALBUMARTIST"),
    "year": ("TDRC", "TYER", "\xa9day", "date", "DATE", "ICRD"),
    "track_number": ("TRCK", "trkn", "tracknumber", "TRACKNUMBER"),
    "disc_number": ("TPOS", "disk", "discnumber", "DISCNUMBER"),
}
_ARTIST_KEYS = ("TPE1", "\xa9ART", "artist", "ARTIST", "IART")
_GENRE_KEYS = ("TCON", "\xa9gen", "genre", "GENRE", "IGNR")
_LYRIC_KEYS = ("USLT", "\xa9lyr", "lyrics", "LYRICS", "UNSYNCEDLYRICS")

EMBEDDED = "embedded_tag"


def read_metadata(path: str | Path) -> MetadataCandidate:
    """Read container facts and embedded tags. Never raises on a bad file.

    Runs before any transcode on purpose: cover art and lyrics live in the
    container, and a conversion is exactly where they stop existing.
    """
    src = Path(path)
    container = detect_file(src)
    candidate = MetadataCandidate(container=container, lossless=is_lossless(container))

    stream = probe(src)
    candidate.codec = stream.get("codec", "")
    candidate.sample_rate = stream.get("sample_rate", 0)
    candidate.channels = stream.get("channels", 0)
    candidate.bitrate = stream.get("bitrate", 0)
    candidate.duration_ms = stream.get("duration_ms", 0)

    try:
        import mutagen
    except ImportError:
        return candidate

    try:
        tags = mutagen.File(str(src))
    except Exception as exc:
        logger.warning("[audio-format] 读标签失败 %s: %s", src.name, type(exc).__name__)
        return candidate
    if tags is None:
        return candidate

    if getattr(tags, "info", None) is not None:
        length = getattr(tags.info, "length", 0) or 0
        if length and not candidate.duration_ms:
            candidate.duration_ms = int(length * 1000)
        if not candidate.sample_rate:
            candidate.sample_rate = int(getattr(tags.info, "sample_rate", 0) or 0)
        if not candidate.channels:
            candidate.channels = int(getattr(tags.info, "channels", 0) or 0)

    def first(keys: tuple[str, ...]) -> str:
        for key in keys:
            if key not in tags:
                continue
            raw = tags[key]
            values = raw if isinstance(raw, list) else [raw]
            for value in values:
                text = _stringify(value)
                if text:
                    return text
        return ""

    for name, keys in _TEXT_KEYS.items():
        candidate.set(name, first(keys), source=EMBEDDED)

    artists = _split_artists(first(_ARTIST_KEYS))
    if artists:
        candidate.set("artists", artists, source=EMBEDDED)
        candidate.set("artist", "、".join(artists), source=EMBEDDED)
    genres = _split_artists(first(_GENRE_KEYS))
    if genres:
        candidate.set("genres", genres, source=EMBEDDED)

    candidate.cover_bytes = _extract_cover(tags)
    lyrics = first(_LYRIC_KEYS)
    if lyrics:
        # A timestamped line means a synced LRC; keeping both is deliberate,
        # since the player wants timings and search wants plain text.
        if "[" in lyrics and "]" in lyrics:
            candidate.synced_lyrics = lyrics
            candidate.plain_lyrics = _strip_timestamps(lyrics)
        else:
            candidate.plain_lyrics = lyrics
    return candidate


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        parts = [_stringify(v) for v in value]
        return "/".join(p for p in parts if p)
    text = getattr(value, "text", None)
    if text is not None:
        return _stringify(text)
    if isinstance(value, (int, float)):
        return str(value)
    try:
        return str(value).strip()
    except Exception:
        return ""


def _split_artists(text: str) -> list[str]:
    if not text:
        return []
    parts = [p.strip() for p in text.replace("；", ";").replace("、", "/").replace(";", "/").replace(",", "/").split("/")]
    return [p for p in parts if p]


def _strip_timestamps(lrc: str) -> str:
    import re

    lines = [re.sub(r"\[[^\]]*\]", "", line).strip() for line in lrc.splitlines()]
    return "\n".join(line for line in lines if line)


def _extract_cover(tags: Any) -> bytes:
    """Embedded cover art across the tag formats we actually see."""
    try:
        pictures = getattr(tags, "pictures", None)      # FLAC
        if pictures:
            return bytes(pictures[0].data)
        for key in tags.keys():                          # ID3 APIC:*
            if str(key).startswith("APIC"):
                return bytes(tags[key].data)
        if "covr" in tags:                               # MP4
            covr = tags["covr"]
            return bytes(covr[0] if isinstance(covr, list) else covr)
    except Exception as exc:
        logger.debug("[audio-format] 封面提取跳过: %s", type(exc).__name__)
    return b""
