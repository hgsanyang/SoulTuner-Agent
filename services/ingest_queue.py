"""Filesystem-backed queue that decouples API requests from GPU ingestion."""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Mapping

QUEUE_ROOT = Path(os.getenv("MUSIC_INGEST_QUEUE_DIR", "data/ingest_queue"))
PENDING_DIR = QUEUE_ROOT / "pending"
PROCESSING_DIR = QUEUE_ROOT / "processing"
DONE_DIR = QUEUE_ROOT / "done"
FAILED_DIR = QUEUE_ROOT / "failed"
PLACEHOLDER_TITLES = {"new song", "unknown", "untitled", "test", "song"}


def _ensure_dirs() -> None:
    for directory in (PENDING_DIR, PROCESSING_DIR, DONE_DIR, FAILED_DIR):
        directory.mkdir(parents=True, exist_ok=True)


class IngestQueueValidationError(ValueError):
    """Raised when a song cannot safely enter the offline enrichment queue."""


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _has_audio_pointer(song: Mapping[str, Any]) -> bool:
    return bool(_clean_text(song.get("audio_url")) or _clean_text(song.get("file_basename")))


def validate_songs_for_queue(songs: list[dict[str, Any]] | Any) -> list[dict[str, Any]]:
    """Validate and normalize queued songs before the GPU worker sees them."""
    if not isinstance(songs, list) or not songs:
        raise IngestQueueValidationError("queue job must contain at least one song")

    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(songs):
        if not isinstance(raw, Mapping):
            raise IngestQueueValidationError(f"song[{index}] must be an object")
        song = dict(raw)
        title = _clean_text(song.get("title") or song.get("name"))
        artist = _clean_text(song.get("artist"))
        if not title:
            raise IngestQueueValidationError(f"song[{index}] missing title")
        if title.casefold() in PLACEHOLDER_TITLES:
            raise IngestQueueValidationError(f"song[{index}] has placeholder title: {title}")
        if not artist:
            raise IngestQueueValidationError(f"song[{index}] missing artist")
        if not _has_audio_pointer(song):
            raise IngestQueueValidationError(f"song[{index}] missing audio_url or file_basename")
        song["title"] = title
        song["artist"] = artist
        normalized.append(song)
    return normalized


def enqueue_songs(songs: list[dict[str, Any]]) -> str:
    """Atomically enqueue songs for the offline enrichment worker."""
    _ensure_dirs()
    normalized_songs = validate_songs_for_queue(songs)
    job_id = f"{int(time.time())}-{uuid.uuid4().hex[:10]}"
    target = PENDING_DIR / f"{job_id}.json"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(
        json.dumps({"job_id": job_id, "songs": normalized_songs}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(target)
    return job_id


def _load_job(path: Path, status: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        payload = {}
    validation_error = ""
    try:
        validate_songs_for_queue(payload.get("songs") or [])
        valid = True
    except IngestQueueValidationError as exc:
        validation_error = str(exc)
        valid = False
    stat = path.stat()
    return {
        "job_id": payload.get("job_id") or path.stem,
        "status": status,
        "songs": payload.get("songs") or [],
        "song_count": len(payload.get("songs") or []),
        "error": payload.get("error", ""),
        "valid": valid,
        "validation_error": validation_error,
        "updated_at": int(stat.st_mtime * 1000),
        "file": path.name,
    }


def list_jobs(limit: int = 50) -> list[dict[str, Any]]:
    """Return recent queue jobs across states for UI observability."""
    _ensure_dirs()
    rows: list[dict[str, Any]] = []
    for status, directory in (
        ("processing", PROCESSING_DIR),
        ("pending", PENDING_DIR),
        ("failed", FAILED_DIR),
        ("done", DONE_DIR),
    ):
        for path in directory.glob("*.json"):
            rows.append(_load_job(path, status))
    rows.sort(key=lambda row: row.get("updated_at", 0), reverse=True)
    return rows[: max(1, int(limit))]


def retry_failed_job(job_id: str) -> bool:
    """Move a failed job back to pending for the worker."""
    _ensure_dirs()
    clean_id = str(job_id or "").strip()
    if not clean_id:
        return False
    failed_path = FAILED_DIR / f"{clean_id}.json"
    if not failed_path.exists():
        return False
    payload = json.loads(failed_path.read_text(encoding="utf-8"))
    payload.pop("error", None)
    payload["retried_at"] = int(time.time() * 1000)
    failed_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    failed_path.replace(PENDING_DIR / failed_path.name)
    return True


def claim_next_job() -> tuple[Path, dict[str, Any]] | None:
    """Move one pending job to processing and return its payload."""
    _ensure_dirs()
    for pending in sorted(PENDING_DIR.glob("*.json")):
        processing = PROCESSING_DIR / pending.name
        try:
            pending.replace(processing)
        except (FileNotFoundError, PermissionError):
            continue
        payload = json.loads(processing.read_text(encoding="utf-8"))
        try:
            payload["songs"] = validate_songs_for_queue(payload.get("songs") or [])
        except IngestQueueValidationError as exc:
            fail_job(processing, str(exc))
            continue
        return processing, payload
    return None


def complete_job(job_path: Path) -> None:
    _ensure_dirs()
    job_path.replace(DONE_DIR / job_path.name)


def fail_job(job_path: Path, error: str) -> None:
    _ensure_dirs()
    payload = json.loads(job_path.read_text(encoding="utf-8"))
    payload["error"] = error[:1000]
    job_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    job_path.replace(FAILED_DIR / job_path.name)
