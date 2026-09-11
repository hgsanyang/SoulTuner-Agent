"""Append-only SQLite ledger for auditable user memory.

Neo4j remains the recommendation hot path. This ledger preserves provenance,
expiry, deletion tombstones, and user isolation without mutating raw events.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
import uuid
from contextlib import closing
from pathlib import Path
from typing import Any, Callable, Iterable

from services.memory_models import MemoryLayer, MemoryRecord, MemoryStatus
from services.runtime_mode import side_effects_disabled


SCHEMA = """
CREATE TABLE IF NOT EXISTS memory_records (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    record_id TEXT NOT NULL UNIQUE,
    user_id TEXT NOT NULL,
    layer TEXT NOT NULL,
    kind TEXT NOT NULL,
    source TEXT NOT NULL,
    evidence_id TEXT NOT NULL,
    confidence REAL NOT NULL,
    created_at INTEGER NOT NULL,
    valid_from INTEGER NOT NULL,
    expires_at INTEGER,
    status TEXT NOT NULL,
    memory_key TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    why_used TEXT NOT NULL,
    target_record_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_memory_user_layer_time
ON memory_records(user_id, layer, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_memory_user_key
ON memory_records(user_id, memory_key, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_memory_target
ON memory_records(user_id, target_record_id);
CREATE TABLE IF NOT EXISTS memory_deletions (
    user_id TEXT NOT NULL,
    record_id TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'pending',
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    PRIMARY KEY(user_id, record_id)
);
CREATE TABLE IF NOT EXISTS memory_bulk_deletions (
    user_id TEXT PRIMARY KEY,
    payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS memory_preference_writes (
    operation_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'pending',
    created_at INTEGER NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_memory_pending_write_owner
ON memory_preference_writes(user_id) WHERE state='pending';
CREATE TABLE IF NOT EXISTS memory_projection_jobs (
    record_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'pending'
);
CREATE INDEX IF NOT EXISTS idx_memory_projection_pending_owner ON memory_projection_jobs(user_id,state);
"""

PREFERENCE_CONFLICT_FIELDS = {
    "add_genres": "avoid_genres",
    "avoid_genres": "add_genres",
    "add_moods": "avoid_moods",
    "avoid_moods": "add_moods",
    "add_scenarios": "avoid_scenarios",
    "avoid_scenarios": "add_scenarios",
    "add_artists": "avoid_artists",
    "avoid_artists": "add_artists",
}


def default_memory_db_path() -> Path:
    configured = os.getenv("MEMORY_EVENT_DB", "").strip()
    return Path(configured) if configured else Path("data") / "memory" / "memory_v2.sqlite3"


class MemoryEventStore:
    def __init__(
        self,
        path: str | Path | None = None,
        *,
        clock_ms: Callable[[], int] | None = None,
        id_factory: Callable[[], str] | None = None,
    ):
        self.path = Path(path) if path is not None else default_memory_db_path()
        self._clock_ms = clock_ms or (lambda: int(time.time() * 1000))
        self._id_factory = id_factory or (lambda: str(uuid.uuid4()))

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.executescript(SCHEMA)
        return connection

    def append(
        self,
        *,
        user_id: str,
        layer: MemoryLayer,
        kind: str,
        source: str,
        evidence_id: str,
        payload: dict[str, Any],
        confidence: float = 1.0,
        expires_at: int | None = None,
        memory_key: str = "",
        why_used: str = "",
        status: MemoryStatus = MemoryStatus.ACTIVE,
        target_record_id: str | None = None,
        now_ms: int | None = None,
        record_id: str | None = None,
        enqueue_projection: bool = False,
    ) -> MemoryRecord | None:
        if side_effects_disabled():
            return None
        user_id = str(user_id or "").strip()
        if not user_id:
            raise ValueError("user_id is required")
        now = int(now_ms if now_ms is not None else self._clock_ms())
        retry_record_id = record_id is not None
        record_id = str(record_id or self._id_factory())
        payload_data = dict(payload or {})
        if layer != MemoryLayer.RAW_EVENT and memory_key and status == MemoryStatus.ACTIVE:
            prior = next(
                (
                    item for item in self.effective_records(user_id=user_id, now_ms=now, limit=1000)
                    if item.layer == layer and item.memory_key == memory_key
                ),
                None,
            )
            payload_data.setdefault(
                "canonical_memory_id",
                str(prior.payload.get("canonical_memory_id") or prior.record_id) if prior else record_id,
            )
        record = MemoryRecord(
            record_id=record_id,
            user_id=user_id,
            layer=layer,
            kind=str(kind or "memory"),
            source=str(source or "unknown"),
            evidence_id=str(evidence_id or ""),
            confidence=max(0.0, min(1.0, float(confidence))),
            created_at=now,
            valid_from=now,
            expires_at=expires_at,
            status=status,
            memory_key=str(memory_key or ""),
            payload=payload_data,
            why_used=str(why_used or ""),
            target_record_id=target_record_id,
        )
        data = record.model_dump()
        with closing(self._connect()) as connection:
            cursor = connection.execute(
                f"""INSERT {'OR IGNORE' if retry_record_id else ''} INTO memory_records(
                    record_id,user_id,layer,kind,source,evidence_id,confidence,
                    created_at,valid_from,expires_at,status,memory_key,payload_json,
                    why_used,target_record_id
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    data["record_id"], data["user_id"], data["layer"], data["kind"],
                    data["source"], data["evidence_id"], data["confidence"],
                    data["created_at"], data["valid_from"], data["expires_at"],
                    data["status"], data["memory_key"],
                    json.dumps(data["payload"], ensure_ascii=False, separators=(",", ":")),
                    data["why_used"], data["target_record_id"],
                ),
            )
            if retry_record_id and cursor.rowcount == 0:
                existing = self._decode(connection.execute(
                    "SELECT * FROM memory_records WHERE record_id=?", (record_id,),
                ).fetchone())
                def comparable(value):
                    return {k: v for k, v in value.items() if k != "canonical_memory_id"}
                if (existing.user_id != user_id or existing.layer != layer or existing.memory_key != memory_key
                        or comparable(existing.payload) != comparable(payload_data)):
                    raise RuntimeError("memory_record_identity_conflict")
                return existing
            opposite = PREFERENCE_CONFLICT_FIELDS.get(str(payload_data.get("field") or ""))
            effective_now = int(self._clock_ms())
            if (layer == MemoryLayer.EXPLICIT and status == MemoryStatus.ACTIVE and opposite
                    and now <= effective_now and (expires_at is None or expires_at > effective_now)):
                inverse_key = f"preference:{opposite}:{str(payload_data.get('value') or '').strip().casefold()}"
                connection.execute("""
                    INSERT INTO memory_records (
                        record_id,user_id,layer,kind,source,evidence_id,confidence,
                        created_at,valid_from,expires_at,status,memory_key,payload_json,why_used,target_record_id
                    )
                    SELECT 'correction-' || lower(hex(randomblob(16))),r.user_id,r.layer,
                        'tombstone','user_correction',?,1.0,?,?,NULL,'deleted',r.memory_key,'{}',
                        'Explicit preference correction',r.record_id
                    FROM memory_records r WHERE r.user_id=? AND r.layer='L1'
                      AND r.memory_key=? AND r.status='active'
                      AND r.seq<(SELECT seq FROM memory_records WHERE record_id=?)
                      AND NOT EXISTS (SELECT 1 FROM memory_records d
                        WHERE d.user_id=r.user_id AND d.target_record_id=r.record_id
                          AND d.status IN ('deleted','superseded'))
                """, (record_id, now, now, user_id, inverse_key, record_id))
            if enqueue_projection:
                if layer != MemoryLayer.INFERRED or status != MemoryStatus.ACTIVE:
                    raise ValueError("only_active_inferred_projection")
                connection.execute(
                    "INSERT INTO memory_projection_jobs(record_id,user_id) VALUES (?,?)",
                    (record_id, user_id),
                )
            connection.commit()
        return record

    def pending_projection_ids(self, *, user_id: str, limit: int = 20) -> list[str]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT j.record_id FROM memory_projection_jobs j JOIN memory_records r ON r.record_id=j.record_id "
                "WHERE j.user_id=? AND j.state='pending' ORDER BY r.seq LIMIT ?",
                (user_id, max(0, min(int(limit), 100))),
            ).fetchall()
        return [row[0] for row in rows]

    def projection_sequence(self, *, user_id: str, record_id: str) -> int:
        with closing(self._connect()) as connection:
            row = connection.execute("SELECT seq FROM memory_records WHERE user_id=? AND record_id=?",
                                     (user_id, record_id)).fetchone()
        if row is None:
            raise RuntimeError("projection_record_not_found")
        return int(row[0])

    def projection_pending(self, *, user_id: str, record_id: str) -> bool:
        with closing(self._connect()) as connection:
            return connection.execute(
                "SELECT 1 FROM memory_projection_jobs WHERE user_id=? AND record_id=? AND state='pending'",
                (user_id, record_id),
            ).fetchone() is not None

    def finish_projection(self, *, user_id: str, record_id: str) -> None:
        if side_effects_disabled():
            return
        with closing(self._connect()) as connection:
            connection.execute("UPDATE memory_projection_jobs SET state='completed' WHERE user_id=? AND record_id=?",
                               (user_id, record_id))
            connection.commit()

    def stage_preference_write(self, *, user_id: str, preferences: dict[str, Any]) -> dict[str, Any]:
        if side_effects_disabled():
            raise RuntimeError("memory_write_disabled")
        payload = json.dumps(preferences, ensure_ascii=False, sort_keys=True)
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT * FROM memory_preference_writes WHERE user_id=? AND state='pending'", (user_id,),
            ).fetchone()
            if existing:
                if existing["payload_json"] != payload:
                    raise RuntimeError("previous_preference_write_pending")
                return dict(existing)
            operation = {"operation_id": str(uuid.uuid4()), "user_id": user_id,
                         "payload_json": payload, "created_at": int(self._clock_ms()), "state": "pending"}
            connection.execute(
                "INSERT INTO memory_preference_writes(operation_id,user_id,payload_json,created_at) VALUES (?,?,?,?)",
                (operation["operation_id"], user_id, payload, operation["created_at"]),
            )
            connection.commit()
            return operation

    def pending_preference_write(self, *, user_id: str) -> dict[str, Any] | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM memory_preference_writes WHERE user_id=? AND state='pending'", (user_id,),
            ).fetchone()
        return dict(row) if row else None

    def finish_preference_write(self, *, user_id: str, operation_id: str) -> None:
        if side_effects_disabled():
            return
        with closing(self._connect()) as connection:
            connection.execute(
                "UPDATE memory_preference_writes SET state='completed' WHERE user_id=? AND operation_id=?",
                (user_id, operation_id),
            )
            connection.commit()

    def tombstone(
        self,
        *,
        user_id: str,
        target_record_id: str,
        source: str = "user_delete",
        evidence_id: str = "",
    ) -> MemoryRecord | None:
        target = self.get(user_id=user_id, record_id=target_record_id)
        if target is None:
            return None
        existing = self._rows(
            "WHERE user_id=? AND target_record_id=? AND status='deleted' ORDER BY seq DESC LIMIT 1",
            (user_id, target_record_id),
        )
        if existing:
            return existing[0]
        return self.append(
            user_id=user_id,
            layer=target.layer,
            kind="tombstone",
            source=source,
            evidence_id=evidence_id,
            payload={"deleted_record_id": target_record_id},
            memory_key=target.memory_key,
            status=MemoryStatus.DELETED,
            target_record_id=target_record_id,
            why_used="User requested deletion",
        )

    def tombstone_layer(self, *, user_id: str, layer: MemoryLayer, through_seq: int | None = None) -> int:
        """Atomically invalidate all active records in a layer, without a UI limit."""
        if side_effects_disabled():
            return 0
        now = self._clock_ms()
        with closing(self._connect()) as connection:
            cursor = connection.execute("""
                INSERT INTO memory_records (
                    record_id,user_id,layer,kind,source,evidence_id,confidence,
                    created_at,valid_from,expires_at,status,memory_key,payload_json,
                    why_used,target_record_id
                )
                SELECT 'forget-' || lower(hex(randomblob(16))), r.user_id,r.layer,
                    'tombstone','user_delete','',1.0,?,?,NULL,?,r.memory_key,'{}',
                    'User requested layer deletion',r.record_id
                FROM memory_records r
                WHERE r.user_id=? AND r.layer=? AND r.status=? AND (? IS NULL OR r.seq<=?)
                AND NOT EXISTS (
                    SELECT 1 FROM memory_records d WHERE d.user_id=r.user_id
                    AND d.target_record_id=r.record_id AND d.status IN (?,?)
                )
            """, (now, now, MemoryStatus.DELETED.value, user_id, layer.value,
                  MemoryStatus.ACTIVE.value, through_seq, through_seq,
                  MemoryStatus.DELETED.value, MemoryStatus.SUPERSEDED.value))
            connection.commit()
            return cursor.rowcount

    def bulk_deletion_snapshot(self, *, user_id: str) -> dict[str, Any] | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT payload_json FROM memory_bulk_deletions WHERE user_id=?", (user_id,),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def layer_watermark(self, *, user_id: str, layer: MemoryLayer) -> int:
        with closing(self._connect()) as connection:
            return int(connection.execute(
                "SELECT coalesce(max(seq),0) FROM memory_records WHERE user_id=? AND layer=?",
                (user_id, layer.value),
            ).fetchone()[0])

    def is_effective_record(self, *, user_id: str, record_id: str) -> bool:
        now = int(self._clock_ms())
        with closing(self._connect()) as connection:
            return connection.execute("""
                SELECT 1 FROM memory_records r WHERE r.user_id=? AND r.record_id=?
                  AND r.status='active' AND r.valid_from<=?
                  AND (r.expires_at IS NULL OR r.expires_at>?)
                  AND NOT EXISTS (SELECT 1 FROM memory_records d WHERE d.user_id=r.user_id
                    AND d.target_record_id=r.record_id AND d.status IN ('deleted','superseded'))
            """, (user_id, record_id, now, now)).fetchone() is not None

    def record_ids_after(self, *, user_id: str, through_seq: int) -> set[str]:
        with closing(self._connect()) as connection:
            return {row[0] for row in connection.execute(
                "SELECT record_id FROM memory_records WHERE user_id=? AND seq>?", (user_id, through_seq),
            )}

    def stage_bulk_deletion(self, *, user_id: str, snapshot: dict[str, Any]) -> dict[str, Any]:
        if side_effects_disabled():
            raise RuntimeError("bulk_deletion_disabled")
        with closing(self._connect()) as connection:
            connection.execute(
                "INSERT OR IGNORE INTO memory_bulk_deletions(user_id,payload_json) VALUES (?,?)",
                (user_id, json.dumps(snapshot, ensure_ascii=False)),
            )
            connection.commit()
            row = connection.execute(
                "SELECT payload_json FROM memory_bulk_deletions WHERE user_id=?", (user_id,),
            ).fetchone()
        return json.loads(row[0])

    def finish_bulk_deletion(self, *, user_id: str, operation_id: str) -> None:
        if side_effects_disabled():
            return
        with closing(self._connect()) as connection:
            connection.execute(
                "DELETE FROM memory_bulk_deletions WHERE user_id=? AND json_extract(payload_json,'$.operation_id')=?",
                (user_id, operation_id),
            )
            connection.commit()

    def supersede(
        self,
        *,
        user_id: str,
        target_record_id: str,
        source: str = "memory_link",
        evidence_id: str = "",
        superseded_by: str = "",
    ) -> MemoryRecord | None:
        """Mark a record superseded (a newer memory replaced it).

        Like tombstone but status=SUPERSEDED: the target drops out of the
        effective view while its history stays in the append-only ledger.
        """
        target = self.get(user_id=user_id, record_id=target_record_id)
        if target is None:
            return None
        return self.append(
            user_id=user_id,
            layer=target.layer,
            kind="supersede",
            source=source,
            evidence_id=evidence_id,
            payload={"superseded_record_id": target_record_id, "superseded_by": superseded_by},
            memory_key=target.memory_key,
            status=MemoryStatus.SUPERSEDED,
            target_record_id=target_record_id,
            why_used="Superseded by a newer, linked memory",
        )

    def resolve_effective_record_id(self, *, user_id: str, canonical_memory_id: str) -> str | None:
        """Map a canonical memory id to its current effective record id, or None."""
        target = str(canonical_memory_id or "").strip()
        if not target:
            return None
        for record in self.effective_records(user_id=user_id, limit=1000):
            canonical = str(record.payload.get("canonical_memory_id") or record.record_id)
            if canonical == target or record.record_id == target:
                return record.record_id
        return None

    def get(self, *, user_id: str, record_id: str) -> MemoryRecord | None:
        rows = self._rows("WHERE user_id = ? AND record_id = ?", (user_id, record_id))
        return rows[0] if rows else None

    def stage_deletion(self, *, user_id: str, record_id: str) -> bool:
        """Persist only an owner-verified intent; a crash leaves it retryable."""
        if side_effects_disabled():
            return False
        now = int(self._clock_ms())
        with closing(self._connect()) as connection:
            connection.execute(
                """INSERT OR IGNORE INTO memory_deletions(user_id,record_id,created_at,updated_at)
                   SELECT user_id,record_id,?,? FROM memory_records
                   WHERE user_id=? AND record_id=? AND status='active'""",
                (now, now, user_id, record_id),
            )
            connection.commit()
            return connection.execute(
                "SELECT 1 FROM memory_deletions WHERE user_id=? AND record_id=?",
                (user_id, record_id),
            ).fetchone() is not None

    def tombstone_lineage(self, *, user_id: str, record_id: str) -> bool:
        """Retire target and older same-key versions, never later replacements."""
        if side_effects_disabled():
            return False
        now = int(self._clock_ms())
        with closing(self._connect()) as connection:
            connection.execute("""
                INSERT INTO memory_records (
                    record_id,user_id,layer,kind,source,evidence_id,confidence,
                    created_at,valid_from,expires_at,status,memory_key,payload_json,
                    why_used,target_record_id
                )
                SELECT 'forget-' || lower(hex(randomblob(16))),r.user_id,r.layer,
                    'tombstone','user_delete','',1.0,?,?,NULL,'deleted',r.memory_key,'{}',
                    'User requested recording lineage deletion',r.record_id
                FROM memory_records r JOIN memory_records target
                  ON target.user_id=r.user_id AND target.record_id=?
                WHERE r.user_id=? AND r.status='active' AND r.seq<=target.seq
                  AND (r.record_id=target.record_id OR
                       (target.memory_key<>'' AND r.memory_key=target.memory_key AND r.layer=target.layer))
                  AND NOT EXISTS (
                    SELECT 1 FROM memory_records d WHERE d.user_id=r.user_id
                    AND d.target_record_id=r.record_id AND d.status='deleted')
            """, (now, now, record_id, user_id))
            connection.commit()
            return connection.execute(
                "SELECT 1 FROM memory_records WHERE user_id=? AND target_record_id=? AND status='deleted' LIMIT 1",
                (user_id, record_id),
            ).fetchone() is not None

    def has_newer_memory_version(self, *, user_id: str, record_id: str) -> bool:
        with closing(self._connect()) as connection:
            return connection.execute("""
                SELECT 1 FROM memory_records r JOIN memory_records target
                  ON target.user_id=r.user_id AND target.record_id=?
                WHERE r.user_id=? AND target.memory_key<>'' AND r.memory_key=target.memory_key
                  AND r.layer=target.layer AND r.seq>target.seq AND r.status='active'
                  AND NOT EXISTS (SELECT 1 FROM memory_records d
                    WHERE d.user_id=r.user_id AND d.target_record_id=r.record_id
                    AND d.status IN ('deleted','superseded')) LIMIT 1
            """, (record_id, user_id)).fetchone() is not None

    def finish_deletion(self, *, user_id: str, record_id: str) -> None:
        if side_effects_disabled():
            return
        with closing(self._connect()) as connection:
            connection.execute(
                "UPDATE memory_deletions SET state='completed',updated_at=? WHERE user_id=? AND record_id=?",
                (int(self._clock_ms()), user_id, record_id),
            )
            connection.commit()

    def pending_deletions(self, *, user_id: str, limit: int = 100) -> list[str]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT record_id FROM memory_deletions WHERE user_id=? AND state='pending' "
                "ORDER BY created_at LIMIT ?", (user_id, max(1, min(int(limit), 1000))),
            ).fetchall()
        return [row[0] for row in rows]

    def deletion_completed(self, *, user_id: str, record_id: str) -> bool:
        with closing(self._connect()) as connection:
            return connection.execute(
                "SELECT 1 FROM memory_deletions WHERE user_id=? AND record_id=? AND state='completed'",
                (user_id, record_id),
            ).fetchone() is not None

    def pending_deletion_count(self, *, user_id: str) -> int:
        with closing(self._connect()) as connection:
            return int(connection.execute(
                "SELECT count(*) FROM memory_deletions WHERE user_id=? AND state='pending'", (user_id,),
            ).fetchone()[0])

    def list_records(
        self,
        *,
        user_id: str,
        layers: Iterable[MemoryLayer] | None = None,
        limit: int = 200,
    ) -> list[MemoryRecord]:
        params: list[Any] = [user_id]
        clause = "WHERE user_id = ?"
        values = [layer.value for layer in (layers or [])]
        if values:
            clause += " AND layer IN (" + ",".join("?" for _ in values) + ")"
            params.extend(values)
        clause += " ORDER BY seq DESC LIMIT ?"
        params.append(max(1, min(int(limit), 10000)))
        return self._rows(clause, tuple(params))

    def _invalidated_ids(self, user_id: str) -> set[str]:
        """Deletion applies across layers and read windows, never just recent rows."""
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT DISTINCT target_record_id FROM memory_records "
                "WHERE user_id = ? AND status IN (?, ?) AND target_record_id IS NOT NULL",
                (user_id, MemoryStatus.DELETED.value, MemoryStatus.SUPERSEDED.value),
            ).fetchall()
        return {row[0] for row in rows}

    def effective_records(self, *, user_id: str, now_ms: int | None = None, limit: int = 200) -> list[MemoryRecord]:
        now = int(now_ms if now_ms is not None else self._clock_ms())
        limit = max(0, min(int(limit), 10000))
        if not limit:
            return []
        eligible = """WHERE r.user_id = ? AND r.status = 'active'
            AND r.valid_from <= ? AND (r.expires_at IS NULL OR r.expires_at > ?)
            AND NOT EXISTS (
                SELECT 1 FROM memory_records d
                WHERE d.user_id = r.user_id AND d.target_record_id = r.record_id
                  AND d.status IN ('deleted', 'superseded'))"""
        blocked_keys: set[str] = set()
        effective: list[MemoryRecord] = []
        # Both passes share a snapshot. Expired rows, tombstones and repeated
        # versions must not exhaust a raw-row window before usable memories.
        with closing(self._connect()) as connection:
            connection.execute("BEGIN")
            for raw in connection.execute(
                "SELECT r.* FROM memory_records r " + eligible + " AND r.layer = 'L1'",
                (user_id, now, now),
            ):
                row = self._decode(raw)
                if row.memory_key:
                    blocked_keys.add(row.memory_key)
                field = str(row.payload.get("field") or "")
                value = str(row.payload.get("value") or "").strip()
                conflict_field = PREFERENCE_CONFLICT_FIELDS.get(field)
                if conflict_field and value:
                    blocked_keys.add(f"preference:{conflict_field}:{value.casefold()}")
            newest_by_key: set[tuple[MemoryLayer, str]] = set()
            # Explicit preferences get candidate admission priority, not an
            # unconditional prompt injection; relevance scoring still follows.
            for raw in connection.execute(
                "SELECT r.* FROM memory_records r " + eligible
                + " ORDER BY CASE WHEN r.layer = 'L1' THEN 0 ELSE 1 END, r.seq DESC",
                (user_id, now, now),
            ):
                row = self._decode(raw)
                if row.layer == MemoryLayer.INFERRED and row.memory_key in blocked_keys:
                    continue
                if row.memory_key:
                    identity = (row.layer, row.memory_key)
                    if identity in newest_by_key:
                        continue
                    newest_by_key.add(identity)
                effective.append(row)
                if len(effective) >= limit:
                    break
        return effective

    def recent_evidence(self, *, user_id: str, limit: int = 40) -> list[MemoryRecord]:
        """Return bounded user-originated L0 evidence for consolidation."""
        now = self._clock_ms()
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """SELECT r.* FROM memory_records r
                WHERE r.user_id = ? AND r.layer = 'L0' AND r.status = 'active'
                  AND r.source IN ('user_action', 'user_statement', 'slate_feedback')
                  AND r.valid_from <= ? AND (r.expires_at IS NULL OR r.expires_at > ?)
                  AND NOT EXISTS (
                      SELECT 1 FROM memory_records d
                      WHERE d.user_id = r.user_id AND d.target_record_id = r.record_id
                        AND d.status IN ('deleted', 'superseded'))
                ORDER BY r.seq DESC LIMIT ?""",
                (user_id, now, now, max(0, min(int(limit), 200))),
            ).fetchall()
        return [self._decode(row) for row in rows]

    def pending_evidence_count(self, *, user_id: str, limit: int = 200) -> int:
        """Count evidence newer than the latest consolidation audit marker."""
        # Find the watermark independently of the output window. Otherwise many
        # newer bookkeeping rows hide it and already-consolidated events recur.
        now = int(self._clock_ms())
        with closing(self._connect()) as connection:
            row = connection.execute(
                """SELECT count(*) FROM (
                    SELECT r.record_id FROM memory_records r
                    WHERE r.user_id = ? AND r.layer = 'L0' AND r.status = 'active'
                      AND r.source IN ('user_action','user_statement','slate_feedback')
                      AND r.valid_from <= ? AND (r.expires_at IS NULL OR r.expires_at > ?)
                      AND r.created_at > coalesce((
                          SELECT max(a.created_at) FROM memory_records a
                          WHERE a.user_id = r.user_id AND a.kind = 'consolidation_audit'
                            AND a.created_at <= ?), -1)
                      AND NOT EXISTS (
                          SELECT 1 FROM memory_records d
                          WHERE d.user_id = r.user_id AND d.target_record_id = r.record_id
                            AND d.status IN ('deleted','superseded'))
                    LIMIT ?
                )""", (user_id, now, now, now, max(1, min(int(limit), 1000))),
            ).fetchone()
        return int(row[0])

    def fingerprint(self, *, user_id: str) -> str:
        payload = [record.model_dump() for record in self.list_records(user_id=user_id, limit=1000)]
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _rows(self, clause: str, params: tuple[Any, ...]) -> list[MemoryRecord]:
        if not self.path.exists() and "record_id" in clause:
            return []
        with closing(self._connect()) as connection:
            rows = connection.execute("SELECT * FROM memory_records " + clause, params).fetchall()
        return [self._decode(row) for row in rows]

    @staticmethod
    def _decode(row: sqlite3.Row) -> MemoryRecord:
        return MemoryRecord(
            record_id=row["record_id"], user_id=row["user_id"],
            layer=MemoryLayer(row["layer"]), kind=row["kind"], source=row["source"],
            evidence_id=row["evidence_id"], confidence=float(row["confidence"]),
            created_at=int(row["created_at"]), valid_from=int(row["valid_from"]),
            expires_at=row["expires_at"], status=MemoryStatus(row["status"]),
            memory_key=row["memory_key"], payload=json.loads(row["payload_json"] or "{}"),
            why_used=row["why_used"], target_record_id=row["target_record_id"],
        )
