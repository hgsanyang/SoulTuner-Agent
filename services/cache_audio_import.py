"""Prepare decoded cache audio for the existing catalog ingestion worker.

The module is deliberately split into planning and publishing. Planning decodes
into a temporary directory, identifies the real container, verifies the audio
stream, reads embedded metadata, and checks duplicates. Publishing is the only
step that writes durable files, and it returns the exact paths needed for a
reversible import run.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

from services.audio_decoder import DecodeResult, process_audio
from services.audio_format import MetadataCandidate, read_metadata
SUPPORTED_AUDIO_SUFFIXES = frozenset(
    {".mp3", ".flac", ".wav", ".m4a", ".aac", ".ogg", ".opus", ".webm"}
)


@dataclass(slots=True)
class CacheImportPlan:
    source_path: str
    song_id: str
    title: str
    artist: str
    album: str
    duration_ms: int
    quality: str
    state: str
    release_year: int = 0
    album_id: str = ""
    cover_url: str = ""
    aliases: list[str] = field(default_factory=list)
    reason: str = ""
    decoded_path: str = ""
    container: str = ""
    mime_type: str = ""
    lossless: bool = False
    sha256: str = ""
    embedded_fields: dict[str, dict[str, Any]] = field(default_factory=dict)
    embedded_lyrics: str = ""
    cover_path: str = ""


@dataclass(slots=True)
class PublishedCacheSong:
    record: dict[str, Any]
    created_files: list[str]
    plan: CacheImportPlan


def sha256_file(path: str | Path, *, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def choose_preferred_cache_entries(entries: Iterable[Any]) -> tuple[list[Any], list[dict[str, str]]]:
    """Keep one completed cache file per song id, preferring quality then size."""
    selected: dict[str, Any] = {}
    skipped: list[dict[str, str]] = []
    for entry in entries:
        path = Path(str(getattr(entry, "path", "")))
        song_id = str(getattr(entry, "song_id", "")).strip()
        if not song_id:
            skipped.append({"path": str(path), "reason": "missing_song_id"})
            continue
        if path.suffix.lower() == ".uc!":
            skipped.append({"path": str(path), "reason": "partial_download"})
            continue
        current = selected.get(song_id)
        rank = (_quality_rank(getattr(entry, "quality", "")), int(getattr(entry, "bytes", 0)))
        current_rank = (
            _quality_rank(getattr(current, "quality", "")),
            int(getattr(current, "bytes", 0)),
        ) if current is not None else (-1, -1)
        if rank > current_rank:
            if current is not None:
                skipped.append({"path": str(getattr(current, "path", "")), "reason": "lower_quality_copy"})
            selected[song_id] = entry
        else:
            skipped.append({"path": str(path), "reason": "lower_quality_copy"})
    return list(selected.values()), skipped


def build_existing_digest_index(
    roots: Iterable[str | Path],
    *,
    candidate_sizes: set[int] | None = None,
) -> dict[str, str]:
    """Hash only relevant library files; decoded cache bytes keep the same size."""
    found: dict[str, str] = {}
    for root_value in roots:
        root = Path(root_value)
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_AUDIO_SUFFIXES:
                continue
            try:
                size = path.stat().st_size
            except OSError:
                continue
            if candidate_sizes and size not in candidate_sizes:
                continue
            try:
                found.setdefault(sha256_file(path), str(path))
            except OSError:
                continue
    return found


def plan_cache_audio(
    entry: Any,
    temporary_dir: str | Path,
    *,
    existing_digests: Mapping[str, str] | None = None,
    seen_digests: set[str] | None = None,
) -> CacheImportPlan:
    """Decode and inspect one cache entry without writing durable catalog files."""
    plan = CacheImportPlan(
        source_path=str(getattr(entry, "path", "")),
        song_id=str(getattr(entry, "song_id", "")),
        title=str(getattr(entry, "title", "") or "").strip(),
        artist=str(getattr(entry, "artist", "") or "").strip(),
        album=str(getattr(entry, "album", "") or "").strip(),
        duration_ms=int(getattr(entry, "duration_ms", 0) or 0),
        quality=str(getattr(entry, "quality", "") or ""),
        state="planning",
        release_year=int(getattr(entry, "release_year", 0) or 0),
        album_id=str(getattr(entry, "album_id", "") or ""),
        cover_url=str(getattr(entry, "cover_url", "") or ""),
        aliases=list(getattr(entry, "aliases", []) or []),
    )
    decoded: DecodeResult = process_audio(plan.source_path, temporary_dir, overwrite=True)
    embedded = read_metadata(decoded.output_path)
    _merge_embedded_metadata(plan, embedded)
    plan.decoded_path = str(decoded.output_path)
    plan.container = decoded.container
    plan.mime_type = decoded.mime_type
    plan.lossless = decoded.lossless
    plan.sha256 = sha256_file(decoded.output_path)
    plan.embedded_fields = {
        key: {"value": value.value, "source": value.source, "confidence": value.confidence}
        for key, value in embedded.fields.items()
    }
    plan.embedded_lyrics = embedded.synced_lyrics or embedded.plain_lyrics
    if embedded.cover_bytes:
        cover_suffix = ".png" if embedded.cover_bytes.startswith(b"\x89PNG") else ".jpg"
        cover_path = Path(temporary_dir) / f"{Path(plan.source_path).stem}_cover{cover_suffix}"
        cover_path.write_bytes(embedded.cover_bytes)
        plan.cover_path = str(cover_path)

    if not plan.title or not plan.artist:
        plan.state = "metadata_missing"
        plan.reason = "title_or_artist_missing_after_metadata_merge"
    elif existing_digests and plan.sha256 in existing_digests:
        plan.state = "duplicate_exact_hash"
        plan.reason = str(existing_digests[plan.sha256])
    elif seen_digests is not None and plan.sha256 in seen_digests:
        plan.state = "duplicate_within_batch"
        plan.reason = plan.sha256
    else:
        plan.state = "ready"
        if seen_digests is not None:
            seen_digests.add(plan.sha256)
    return plan


def publish_cache_audio(
    plan: CacheImportPlan,
    *,
    processed_root: str | Path,
    lyrics: str = "",
    run_id: str,
) -> PublishedCacheSong:
    """Move one ready plan into the durable library layout atomically."""
    if plan.state != "ready":
        raise ValueError(f"cannot publish cache plan in state {plan.state}")
    source = Path(plan.decoded_path)
    if not source.exists():
        raise FileNotFoundError(f"decoded audio missing: {source}")

    root = Path(processed_root)
    audio_dir = root / "audio"
    metadata_dir = root / "metadata"
    lyric_dir = root / "lyrics"
    cover_dir = root / "covers"
    for directory in (audio_dir, metadata_dir, lyric_dir, cover_dir):
        directory.mkdir(parents=True, exist_ok=True)

    basename = _safe_basename(f"{plan.title} - {plan.artist}")
    suffix = source.suffix.lower()
    audio_path = _collision_safe_path(audio_dir / f"{basename}{suffix}", plan)
    basename = audio_path.stem
    metadata_path = metadata_dir / f"{basename}_meta.json"
    lyric_path = lyric_dir / f"{basename}.lrc"
    cover_source = Path(plan.cover_path) if plan.cover_path else None
    cover_path = cover_dir / f"{basename}_cover{cover_source.suffix if cover_source else '.jpg'}"
    created: list[str] = []

    try:
        os.replace(source, audio_path)
        created.append(str(audio_path))

        lyric_text = str(lyrics or "").strip()
        if not lyric_text:
            lyric_text = plan.embedded_lyrics
        if lyric_text:
            _atomic_write_text(lyric_path, lyric_text)
            created.append(str(lyric_path))

        if cover_source is not None and cover_source.exists():
            shutil.copyfile(cover_source, cover_path)
            created.append(str(cover_path))

        release_year = plan.release_year or _year_from_value(_embedded_value(plan, "year"))
        artists = [part.strip() for part in re.split(r"[、,/;&]+", plan.artist) if part.strip()]
        metadata = {
            "musicId": int(plan.song_id) if plan.song_id.isdigit() else plan.song_id,
            "musicName": plan.title,
            "artist": [[name, 0] for name in artists] or [[plan.artist, 0]],
            "album": plan.album or "Unknown",
            "duration": plan.duration_ms,
            "format": plan.container,
            "mime_type": plan.mime_type,
            "lossless": plan.lossless,
            "source": "netease_cache",
            "source_platform": "netease",
            "source_id": plan.song_id,
            "source_audio_sha256": plan.sha256,
            "source_quality": plan.quality,
            "raw_source_path": plan.source_path,
            "metadata_source": "netease_api+embedded_audio",
            "metadata_fields": plan.embedded_fields,
            "release_year": release_year or None,
            "album_id": plan.album_id,
            "aliases": plan.aliases,
            "cache_import_run_id": run_id,
            "imported_at": datetime.now().isoformat(),
        }
        _atomic_write_text(metadata_path, json.dumps(metadata, ensure_ascii=False, indent=2))
        created.append(str(metadata_path))

        record = {
            "title": plan.title,
            "artist": plan.artist,
            "album": plan.album or "Unknown",
            "duration": plan.duration_ms,
            "song_id": plan.song_id,
            "source_id": plan.song_id,
            "source": "netease_cache",
            "platform": "netease",
            "metadata_source": "netease_api+embedded_audio",
            "release_year": release_year or None,
            "album_id": plan.album_id,
            "ext": plan.container,
            "audio_url": f"/static/audio/{audio_path.name}",
            "audio_path": str(audio_path),
            "lrc_url": f"/static/lyrics/{lyric_path.name}" if lyric_path.exists() else "",
            "lrc_path": str(lyric_path) if lyric_path.exists() else "",
            "cover_url": (
                f"/static/covers/{cover_path.name}" if cover_path.exists() else plan.cover_url
            ),
            "file_basename": basename,
            "audio_retention": "saved",
            "catalog_tier": "library",
            "requested_by": "explicit_cache_import",
            "cache_import_run_id": run_id,
        }
        return PublishedCacheSong(record=record, created_files=created, plan=plan)
    except Exception:
        for value in reversed(created):
            Path(value).unlink(missing_ok=True)
        raise


def remove_published_files(paths: Iterable[str | Path]) -> int:
    removed = 0
    for value in paths:
        path = Path(value)
        if path.is_file():
            path.unlink()
            removed += 1
    return removed


def plan_as_dict(plan: CacheImportPlan) -> dict[str, Any]:
    return asdict(plan)


def _merge_embedded_metadata(plan: CacheImportPlan, embedded: MetadataCandidate) -> None:
    def value(name: str) -> Any:
        field = embedded.fields.get(name)
        return field.value if field is not None else None

    plan.title = plan.title or str(value("title") or "").strip()
    plan.artist = plan.artist or str(value("artist") or "").strip()
    plan.album = plan.album or str(value("album") or "").strip()
    plan.duration_ms = plan.duration_ms or int(embedded.duration_ms or 0)


def _embedded_value(plan: CacheImportPlan, name: str) -> Any:
    field = plan.embedded_fields.get(name) or {}
    return field.get("value")


def _quality_rank(value: Any) -> int:
    text = str(value or "").strip()
    if text == "999":
        return 1_000_000
    try:
        return int(text)
    except ValueError:
        return -1


def _year_from_value(value: Any) -> int:
    match = re.search(r"(?:19|20)\d{2}", str(value or ""))
    return int(match.group(0)) if match else 0


def _safe_basename(value: str) -> str:
    cleaned = re.sub(r"[\\/:*?\"<>|]+", "_", str(value or ""))
    return " ".join(cleaned.split()).strip(" .") or "Unknown"


def _collision_safe_path(target: Path, plan: CacheImportPlan) -> Path:
    if not target.exists():
        return target
    try:
        if sha256_file(target) == plan.sha256:
            raise FileExistsError(f"exact audio already exists: {target}")
    except OSError:
        pass
    suffix = f" [{plan.song_id}]" if plan.song_id else f" [{plan.sha256[:8]}]"
    return target.with_name(f"{target.stem}{suffix}{target.suffix}")


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)
