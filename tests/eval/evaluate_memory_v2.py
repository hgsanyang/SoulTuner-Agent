"""Deterministic Memory v2 development harness.

The independent sealed multi-session benchmark is intentionally not stored in
this repository. This harness checks storage invariants before that blind gate.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from services.memory_event_store import MemoryEventStore
from services.memory_models import MemoryLayer


def evaluate() -> dict[str, object]:
    checks: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory() as directory:
        store = MemoryEventStore(Path(directory) / "memory.sqlite3")
        explicit = store.append(
            user_id="alice", layer=MemoryLayer.EXPLICIT, kind="preference",
            source="user_explicit", evidence_id="manual",
            payload={"field": "add_moods", "value": "Warm"},
            memory_key="preference:add_moods:warm",
        )
        store.append(
            user_id="alice", layer=MemoryLayer.INFERRED, kind="preference",
            source="slate_feedback_inference", evidence_id="feedback-1",
            payload={"field": "add_moods", "value": "Warm"}, confidence=0.7,
            memory_key="preference:add_moods:warm",
        )
        checks.append({"name": "explicit_overrides_inferred", "passed": len(store.effective_records(user_id="alice")) == 1})
        checks.append({"name": "cross_user_isolation", "passed": store.list_records(user_id="bob") == []})
        before = store.fingerprint(user_id="alice")
        assert explicit is not None
        store.tombstone(user_id="alice", target_record_id=explicit.record_id)
        after = store.fingerprint(user_id="alice")
        checks.append({"name": "append_only_tombstone", "passed": before != after and len(store.list_records(user_id="alice")) == 3})
        checks.append({"name": "deleted_record_not_effective", "passed": all(row.record_id != explicit.record_id for row in store.effective_records(user_id="alice"))})

    passed = sum(bool(check["passed"]) for check in checks)
    return {"passed": passed, "total": len(checks), "checks": checks, "sealed": False}


if __name__ == "__main__":
    print(json.dumps(evaluate(), ensure_ascii=False, indent=2))
