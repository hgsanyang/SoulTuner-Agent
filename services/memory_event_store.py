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
from typing import Any, Iterable

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
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path is not None else default_memory_db_path()

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
    ) -> MemoryRecord | None:
        if side_effects_disabled():
            return None
        user_id = str(user_id or "").strip()
        if not user_id:
            raise ValueError("user_id is required")
        now = int(now_ms if now_ms is not None else time.time() * 1000)
        record = MemoryRecord(
            record_id=str(uuid.uuid4()),
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
            payload=dict(payload or {}),
            why_used=str(why_used or ""),
            target_record_id=target_record_id,
        )
        data = record.model_dump()
        with closing(self._connect()) as connection:
            connection.execute(
                """INSERT INTO memory_records(
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
            connection.commit()
        return record

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

    def get(self, *, user_id: str, record_id: str) -> MemoryRecord | None:
        rows = self._rows("WHERE user_id = ? AND record_id = ?", (user_id, record_id))
        return rows[0] if rows else None

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
        params.append(max(1, min(int(limit), 1000)))
        return self._rows(clause, tuple(params))

    def effective_records(self, *, user_id: str, now_ms: int | None = None, limit: int = 200) -> list[MemoryRecord]:
        now = int(now_ms if now_ms is not None else time.time() * 1000)
        rows = self.list_records(user_id=user_id, limit=max(limit * 4, 400))
        deleted = {row.target_record_id for row in rows if row.status == MemoryStatus.DELETED and row.target_record_id}
        active = [
            row for row in rows
            if row.status == MemoryStatus.ACTIVE
            and row.record_id not in deleted
            and (row.expires_at is None or row.expires_at > now)
        ]
        explicit_keys = {row.memory_key for row in active if row.layer == MemoryLayer.EXPLICIT and row.memory_key}
        explicit_conflict_keys: set[str] = set()
        for row in active:
            if row.layer != MemoryLayer.EXPLICIT:
                continue
            field = str(row.payload.get("field") or "")
            value = str(row.payload.get("value") or "").strip()
            conflict_field = PREFERENCE_CONFLICT_FIELDS.get(field)
            if conflict_field and value:
                explicit_conflict_keys.add(f"preference:{conflict_field}:{value.casefold()}")
        active = [
            row for row in active
            if not (
                row.layer == MemoryLayer.INFERRED
                and row.memory_key in (explicit_keys | explicit_conflict_keys)
            )
        ]
        # Repeated consolidation reinforces a memory by appending a newer record.
        # Keep the ledger immutable while exposing only the newest effective value.
        newest_by_key: set[tuple[MemoryLayer, str]] = set()
        effective: list[MemoryRecord] = []
        for row in active:
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
        allowed_sources = {"user_action", "user_statement", "slate_feedback"}
        records = self.list_records(
            user_id=user_id,
            layers=[MemoryLayer.RAW_EVENT],
            limit=max(1, min(int(limit), 200)),
        )
        return [record for record in records if record.source in allowed_sources]

    def pending_evidence_count(self, *, user_id: str, limit: int = 200) -> int:
        """Count evidence newer than the latest consolidation audit marker."""
        rows = self.list_records(user_id=user_id, limit=max(10, min(int(limit), 1000)))
        latest_audit = next(
            (row.created_at for row in rows if row.kind == "consolidation_audit"),
            -1,
        )
        return sum(
            1
            for row in rows
            if row.layer == MemoryLayer.RAW_EVENT
            and row.source in {"user_action", "user_statement", "slate_feedback"}
            and row.created_at > latest_audit
        )

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
