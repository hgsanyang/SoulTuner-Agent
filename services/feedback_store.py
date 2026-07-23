"""Canonical feedback event store (SQLite + WAL).

Why SQLite is now the source of truth
-------------------------------------
JSONL append was fine for a replay dump but is a poor system of record: a
half-written line after a crash corrupts the tail, concurrent writers interleave,
and "the latest exposure for this id" costs a full-file scan on every feedback
call. Exposures are now written twice by design (provisional before the songs
stream, final after the graph finishes), which makes last-write-wins semantics
load-bearing — exactly what an upsert gives us and a log does not.

So: SQLite in WAL mode is canonical; JSONL keeps being written as an export /
training snapshot so every existing replay and eval script still works.

Schema evolution is handled by keeping the whole event as JSON in `payload` and
only promoting the columns we filter on. A new field never needs a migration to
be retained.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from contextlib import closing
import threading
import time
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()

SCHEMA_VERSION = 1


def store_path() -> Path:
    """Same directory the JSONL snapshots use — one env var, no drift."""
    from services.feedback_logger import _feedback_dir

    return _feedback_dir() / "feedback.sqlite3"


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False, timeout=10.0)
    conn.row_factory = sqlite3.Row
    # WAL: readers never block the writer, and a crash truncates the WAL instead
    # of corrupting the main db. NORMAL is the right durability/throughput trade
    # for telemetry — we would rather lose the last event than stall a request.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=10000")
    # The DDL is re-run on every connect on purpose: measured at ~0.1ms of the
    # ~5ms write, so memoising it would only buy staleness bugs when the file is
    # removed underneath us. 5ms is noise against a multi-second recommendation.
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS exposures (
            exposure_id TEXT PRIMARY KEY,
            ts INTEGER NOT NULL,
            user_id TEXT,
            intent_type TEXT,
            policy_version TEXT,
            provisional INTEGER NOT NULL DEFAULT 0,
            item_count INTEGER NOT NULL DEFAULT 0,
            payload TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_exposures_ts ON exposures(ts);

        CREATE TABLE IF NOT EXISTS song_feedback (
            song_feedback_id TEXT PRIMARY KEY,
            ts INTEGER NOT NULL,
            exposure_id TEXT,
            music_id TEXT,
            user_id TEXT,
            context_fit TEXT,
            payload TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_song_feedback_exposure ON song_feedback(exposure_id);

        CREATE TABLE IF NOT EXISTS slate_feedback (
            slate_feedback_id TEXT PRIMARY KEY,
            ts INTEGER NOT NULL,
            exposure_id TEXT,
            user_id TEXT,
            rating TEXT,
            payload TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_slate_feedback_exposure ON slate_feedback(exposure_id);

        CREATE TABLE IF NOT EXISTS user_events (
            event_id TEXT PRIMARY KEY,
            ts INTEGER NOT NULL,
            exposure_id TEXT,
            user_id TEXT,
            event_type TEXT,
            payload TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_user_events_exposure ON user_events(exposure_id);
        """
    )
    conn.commit()
    return conn


def get_connection() -> sqlite3.Connection:
    """Open a fresh configured connection. The CALLER closes it.

    Deliberately not cached: a long-lived handle keeps the file locked (Windows
    cannot even delete the directory) and goes stale as soon as the configured
    feedback dir changes. Telemetry write volume is tiny, so open/close per
    operation is the simpler and safer trade.
    """
    return _connect(store_path())


def reset_connection() -> None:
    """No-op kept for callers/tests; connections are no longer cached."""
    return None


def _now_ms() -> int:
    return int(time.time() * 1000)


def _writes_disabled() -> bool:
    from services.runtime_mode import side_effects_disabled

    return side_effects_disabled()


def upsert_exposure(payload: dict[str, Any]) -> None:
    """Insert or REPLACE an exposure.

    Provisional and final writes share an exposure_id on purpose; the final one
    must win. A log cannot express that without a scan, an upsert can.
    """
    if _writes_disabled():
        return
    with _LOCK, closing(get_connection()) as conn:
        conn.execute(
            """
            INSERT INTO exposures (exposure_id, ts, user_id, intent_type, policy_version,
                                   provisional, item_count, payload)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(exposure_id) DO UPDATE SET
                ts=excluded.ts, user_id=excluded.user_id, intent_type=excluded.intent_type,
                policy_version=excluded.policy_version, provisional=excluded.provisional,
                item_count=excluded.item_count, payload=excluded.payload
            """,
            (
                str(payload.get("exposure_id") or ""),
                int(payload.get("ts") or _now_ms()),
                str(payload.get("user_id") or ""),
                str(payload.get("intent_type") or ""),
                str(payload.get("policy_version") or ""),
                1 if payload.get("provisional") else 0,
                int(payload.get("count") or len(payload.get("items") or [])),
                json.dumps(payload, ensure_ascii=False),
            ),
        )
        conn.commit()


def get_exposure(exposure_id: str) -> dict[str, Any] | None:
    exposure_id = str(exposure_id or "").strip()
    if not exposure_id:
        return None
    with closing(get_connection()) as conn:
        row = conn.execute(
            "SELECT payload FROM exposures WHERE exposure_id = ?", (exposure_id,)
        ).fetchone()
    if row is None:
        return None
    try:
        return json.loads(row["payload"])
    except Exception:
        return None


def _insert_event(table: str, id_column: str, id_value: str, payload: dict[str, Any],
                  columns: dict[str, Any]) -> None:
    if _writes_disabled():
        return
    if not id_value:
        # An empty primary key is silently catastrophic here: INSERT OR REPLACE
        # makes every such row overwrite the previous one, so the table keeps
        # exactly one event and the loss is invisible until you count rows.
        # Retain the event under a synthetic id and say loudly that a writer is
        # sending the wrong key name.
        id_value = str(uuid.uuid4())
        logger.warning("[feedback] %s written without %s; using synthetic id %s",
                       table, id_column, id_value)
        payload = {**payload, id_column: id_value}
    cols = [id_column, "ts", *columns.keys(), "payload"]
    values = [id_value, int(payload.get("ts") or _now_ms()), *columns.values(),
              json.dumps(payload, ensure_ascii=False)]
    placeholders = ", ".join("?" for _ in cols)
    with _LOCK, closing(get_connection()) as conn:
        conn.execute(
            f"INSERT OR REPLACE INTO {table} ({', '.join(cols)}) VALUES ({placeholders})",
            values,
        )
        conn.commit()


def insert_song_feedback(payload: dict[str, Any]) -> None:
    _insert_event(
        "song_feedback", "song_feedback_id", str(payload.get("song_feedback_id") or ""), payload,
        {
            "exposure_id": str(payload.get("exposure_id") or ""),
            "music_id": str(payload.get("music_id") or ""),
            "user_id": str(payload.get("user_id") or ""),
            "context_fit": str(payload.get("context_fit") or "") or None,
        },
    )


def insert_slate_feedback(payload: dict[str, Any]) -> None:
    _insert_event(
        "slate_feedback", "slate_feedback_id", str(payload.get("slate_feedback_id") or ""), payload,
        {
            "exposure_id": str(payload.get("exposure_id") or ""),
            "user_id": str(payload.get("user_id") or ""),
            "rating": str(payload.get("rating") or "") or None,
        },
    )


def insert_user_event(payload: dict[str, Any]) -> None:
    _insert_event(
        "user_events", "event_id", str(payload.get("event_id") or ""), payload,
        {
            "exposure_id": str(payload.get("exposure_id") or ""),
            "user_id": str(payload.get("user_id") or ""),
            "event_type": str(payload.get("event_type") or ""),
        },
    )


def export_jsonl(table: str, out_path: Path, *, since_ms: int | None = None) -> int:
    """Dump a table back to JSONL — training snapshots stay reproducible."""
    sql = f"SELECT payload FROM {table}"
    args: Iterable[Any] = ()
    if since_ms is not None:
        sql += " WHERE ts >= ?"
        args = (int(since_ms),)
    sql += " ORDER BY ts ASC"
    with closing(get_connection()) as conn:
        rows = conn.execute(sql, tuple(args)).fetchall()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as sink:
        for row in rows:
            sink.write(row["payload"] + "\n")
    return len(rows)


def counts() -> dict[str, int]:
    out: dict[str, int] = {}
    with closing(get_connection()) as conn:
        for table in ("exposures", "song_feedback", "slate_feedback", "user_events"):
            out[table] = int(conn.execute(f"SELECT COUNT(*) AS c FROM {table}").fetchone()["c"])
    return out
