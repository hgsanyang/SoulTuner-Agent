"""Filesystem-backed queue that decouples API requests from GPU ingestion."""

from __future__ import annotations

import json
import hashlib
import os
import re
import time
import uuid
import sqlite3
import threading
from functools import wraps
from contextlib import closing
from pathlib import Path
from typing import Any, Mapping

QUEUE_ROOT = Path(os.getenv("MUSIC_INGEST_QUEUE_DIR", "data/ingest_queue"))
PENDING_DIR = QUEUE_ROOT / "pending"
PROCESSING_DIR = QUEUE_ROOT / "processing"
DONE_DIR = QUEUE_ROOT / "done"
FAILED_DIR = QUEUE_ROOT / "failed"
PLACEHOLDER_TITLES = {"new song", "unknown", "untitled", "test", "song"}
_LOCK = threading.RLock()
_LOCAL = threading.local()
LEASE_SECONDS = 120
MAX_ATTEMPTS = 3


def serialized(function):
    """Cross-process serialization of short filesystem state transitions."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        with _LOCK:
            if getattr(_LOCAL, "locked", False):
                return function(*args, **kwargs)
            _ensure_dirs()
            with closing(sqlite3.connect(PENDING_DIR.parent / "queue-lock.sqlite3", timeout=30)) as db, db:
                db.execute("BEGIN IMMEDIATE")
                _LOCAL.locked = True
                try:
                    return function(*args, **kwargs)
                finally:
                    _LOCAL.locked = False
    return wrapped


def _write_job(path, payload):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


@serialized
def heartbeat(job_path: Path) -> bool:
    if job_path.resolve().parent != PROCESSING_DIR.resolve() or not job_path.exists():
        return False
    payload = json.loads(job_path.read_text(encoding="utf-8"))
    payload["lease_until"] = time.time() + LEASE_SECONDS
    _write_job(job_path, payload)
    return True


@serialized
def checkpoint_stage(job_path: Path, index: int, key: str, result: Mapping[str, Any]) -> None:
    """Checkpoint model output under the current claim, before catalog writes."""
    if job_path.resolve().parent != PROCESSING_DIR.resolve() or not job_path.exists():
        raise FileNotFoundError("ingest claim no longer owned")
    payload = json.loads(job_path.read_text(encoding="utf-8"))
    if not 0 <= index < len(payload["songs"]):
        raise ValueError("invalid song checkpoint index")
    payload.setdefault("song_stages", {}).setdefault(str(index), {})[key] = dict(result)
    _write_job(job_path, payload)


@serialized
def checkpoint_song(job_path: Path, index: int, result: Mapping[str, Any]) -> None:
    """Persist finished-song progress under the current fenced claim.

    A crash between external writes and this checkpoint can replay that song;
    downstream upserts must remain idempotent. This is not exactly-once delivery.
    """
    if job_path.resolve().parent != PROCESSING_DIR.resolve() or not job_path.exists():
        raise FileNotFoundError("ingest claim no longer owned")
    payload = json.loads(job_path.read_text(encoding="utf-8"))
    if not 0 <= index < len(payload["songs"]):
        raise ValueError("invalid song checkpoint index")
    payload.setdefault("completed_songs", {})[str(index)] = dict(result)
    _write_job(job_path, payload)


@serialized
def recover_expired_jobs() -> int:
    recovered = 0
    for path in PROCESSING_DIR.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("invalid job")
            if float(payload.get("lease_until", 0)) > time.time():
                continue
            if int(payload.get("attempt", 0)) >= MAX_ATTEMPTS:
                fail_job(path, "lease expired; retry budget exhausted")
            else:
                job_id = str(payload.get("job_id") or path.stem)
                if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", job_id):
                    raise ValueError("invalid job id")
                path.replace(PENDING_DIR / f"{job_id}.json")
            recovered += 1
        except (ValueError, OSError, TypeError):
            fail_job(path, "invalid processing job")
    return recovered


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


@serialized
def enqueue_songs(songs: list[dict[str, Any]]) -> str:
    """Atomically enqueue songs for the offline enrichment worker."""
    _ensure_dirs()
    normalized_songs = validate_songs_for_queue(songs)
    fingerprint = hashlib.sha256(json.dumps(normalized_songs, ensure_ascii=False,
                                            sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    for directory in (PENDING_DIR, PROCESSING_DIR):
        for active in directory.glob("*.json"):
            try:
                existing = json.loads(active.read_text(encoding="utf-8"))
                if isinstance(existing, dict) and existing.get("fingerprint") == fingerprint:
                    return str(existing["job_id"])
            except (ValueError, OSError, KeyError):
                continue
    job_id = f"{int(time.time())}-{uuid.uuid4().hex[:10]}"
    target = PENDING_DIR / f"{job_id}.json"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(
        json.dumps({"job_id": job_id, "songs": normalized_songs, "fingerprint": fingerprint}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(target)
    return job_id


def _load_job(path: Path, status: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        payload = {}
    if not isinstance(payload, dict):
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
        "result": payload.get("result") or {},
        "valid": valid,
        "validation_error": validation_error,
        "updated_at": int(stat.st_mtime * 1000),
        "file": path.name,
    }


@serialized
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


@serialized
def retry_failed_job(job_id: str) -> bool:
    """Move a failed job back to pending for the worker."""
    _ensure_dirs()
    clean_id = str(job_id or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", clean_id):
        return False
    failed_path = FAILED_DIR / f"{clean_id}.json"
    if failed_path.resolve().parent != FAILED_DIR.resolve():
        return False
    if not failed_path.exists():
        return False
    try:
        payload = json.loads(failed_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return False
        validate_songs_for_queue(payload.get("songs"))
    except (ValueError, OSError):
        return False
    payload.pop("error", None)
    payload["attempt"] = 0
    payload["retried_at"] = int(time.time() * 1000)
    _write_job(failed_path, payload)
    failed_path.replace(PENDING_DIR / failed_path.name)
    return True


@serialized
def claim_next_job() -> tuple[Path, dict[str, Any]] | None:
    """Move one pending job to processing and return its payload."""
    _ensure_dirs()
    recover_expired_jobs()
    for pending in sorted(PENDING_DIR.glob("*.json")):
        processing = PROCESSING_DIR / f"{pending.stem}.{uuid.uuid4().hex}.json"
        try:
            pending.replace(processing)
        except (FileNotFoundError, PermissionError):
            continue
        try:
            payload = json.loads(processing.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise IngestQueueValidationError("job must be an object")
            payload["songs"] = validate_songs_for_queue(payload.get("songs") or [])
            payload["attempt"] = int(payload.get("attempt", 0)) + 1
            payload["lease_until"] = time.time() + LEASE_SECONDS
            payload.setdefault("job_id", pending.stem)
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", str(payload["job_id"])):
                raise IngestQueueValidationError("invalid job id")
            _write_job(processing, payload)
        except (ValueError, OSError, TypeError) as exc:
            fail_job(processing, str(exc))
            continue
        return processing, payload
    return None


@serialized
def complete_job(job_path: Path, result: Mapping[str, Any] | None = None) -> None:
    """Move a job to done and persist its enrichment result for auditability."""
    _ensure_dirs()
    if job_path.resolve().parent != PROCESSING_DIR.resolve() or not job_path.exists():
        raise FileNotFoundError("ingest claim no longer owned")
    if result is not None:
        payload = json.loads(job_path.read_text(encoding="utf-8"))
        payload["result"] = dict(result)
        payload["completed_at"] = int(time.time() * 1000)
        _write_job(job_path, payload)
    job_path.replace(DONE_DIR / job_path.name)


@serialized
def fail_job(job_path: Path, error: str) -> None:
    _ensure_dirs()
    if job_path.resolve().parent != PROCESSING_DIR.resolve() or not job_path.exists():
        raise FileNotFoundError("ingest claim no longer owned")
    try:
        payload = json.loads(job_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            payload = {}
    except (ValueError, OSError):
        payload = {}
    payload.setdefault("job_id", job_path.stem.split(".")[0])
    payload["error"] = error[:1000]
    _write_job(job_path, payload)
    job_id = str(payload["job_id"])
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", job_id):
        job_id = uuid.uuid4().hex
    job_path.replace(FAILED_DIR / f"{job_id}.json")
