"""Filesystem-backed queue that decouples API requests from GPU ingestion."""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

QUEUE_ROOT = Path(os.getenv("MUSIC_INGEST_QUEUE_DIR", "data/ingest_queue"))
PENDING_DIR = QUEUE_ROOT / "pending"
PROCESSING_DIR = QUEUE_ROOT / "processing"
DONE_DIR = QUEUE_ROOT / "done"
FAILED_DIR = QUEUE_ROOT / "failed"


def _ensure_dirs() -> None:
    for directory in (PENDING_DIR, PROCESSING_DIR, DONE_DIR, FAILED_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def enqueue_songs(songs: list[dict[str, Any]]) -> str:
    """Atomically enqueue songs for the offline enrichment worker."""
    _ensure_dirs()
    job_id = f"{int(time.time())}-{uuid.uuid4().hex[:10]}"
    target = PENDING_DIR / f"{job_id}.json"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(
        json.dumps({"job_id": job_id, "songs": songs}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(target)
    return job_id


def claim_next_job() -> tuple[Path, dict[str, Any]] | None:
    """Move one pending job to processing and return its payload."""
    _ensure_dirs()
    for pending in sorted(PENDING_DIR.glob("*.json")):
        processing = PROCESSING_DIR / pending.name
        try:
            pending.replace(processing)
        except (FileNotFoundError, PermissionError):
            continue
        return processing, json.loads(processing.read_text(encoding="utf-8"))
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
