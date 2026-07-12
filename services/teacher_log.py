"""Teacher-data logging for later local-model distillation.

The logger is safely on by default and writes JSONL rows under
``data/teacher/``.  Text fields are hashed/redacted unless
``TEACHER_LOG_STORE_TEXT=1`` is explicitly enabled for private local SFT
collection.  Set ``TEACHER_LOG=0`` to disable collection entirely.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

from services.runtime_mode import side_effects_disabled
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEACHER_DIR_ENV = "TEACHER_LOG_DIR"
ALLOW_FAST_ENV = "TEACHER_LOG_ALLOW_FAST"


def teacher_log_enabled() -> bool:
    return not side_effects_disabled() and os.getenv("TEACHER_LOG", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def teacher_log_stores_text() -> bool:
    return os.getenv("TEACHER_LOG_STORE_TEXT", "").strip().lower() in {"1", "true", "yes", "on"}


def teacher_log_allows_fast() -> bool:
    return os.getenv(ALLOW_FAST_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _redact(value: Any) -> Any:
    if teacher_log_stores_text():
        return value
    if isinstance(value, str):
        return {
            "sha256": _sha256_text(value),
            "chars": len(value),
        }
    if isinstance(value, dict):
        return {str(key): _redact(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    return value


def _metadata_marks_fast(metadata: dict[str, Any]) -> bool:
    quality_mode = str(metadata.get("planner_quality_mode") or metadata.get("quality_mode") or "").strip().lower()
    teacher_quality = str(metadata.get("teacher_quality") or "").strip().lower()
    return quality_mode == "fast" or teacher_quality == "fast"


def _teacher_dir() -> Path:
    configured = os.getenv(TEACHER_DIR_ENV, "").strip()
    root = Path(configured) if configured else PROJECT_ROOT / "data" / "teacher"
    root.mkdir(parents=True, exist_ok=True)
    return root


def log_teacher_example(
    kind: str,
    *,
    inputs: dict[str, Any],
    output: Any,
    metadata: dict[str, Any] | None = None,
) -> Path | None:
    """Append one teacher example if enabled.

    Returns the written path, or ``None`` when disabled.  Logging is best-effort:
    callers should not fail recommendation flow because a local log file cannot
    be written.
    """
    if not teacher_log_enabled():
        return None
    raw_metadata = _jsonable(metadata or {})
    if isinstance(raw_metadata, dict) and _metadata_marks_fast(raw_metadata) and not teacher_log_allows_fast():
        return None
    safe_kind = "".join(ch for ch in str(kind).lower() if ch.isalnum() or ch in {"_", "-"})
    if not safe_kind:
        safe_kind = "teacher"
    row = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "kind": safe_kind,
        "text_mode": "full" if teacher_log_stores_text() else "hashed",
        "inputs": _redact(_jsonable(inputs)),
        "output": _redact(_jsonable(output)),
        "metadata": raw_metadata,
    }
    path = _teacher_dir() / f"{safe_kind}.jsonl"
    try:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        return path
    except Exception:
        return None
