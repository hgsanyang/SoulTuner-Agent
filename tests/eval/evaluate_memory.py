"""Deterministic memory lifecycle and relevance ruler.

This evaluator does not call an LLM or external memory service. It verifies the
non-negotiable safety properties around L0 evidence, L1 precedence, L2 expiry,
user isolation, and query-relevant retrieval.

Run:
    python -m tests.eval.evaluate_memory
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

from services.memory_event_store import MemoryEventStore
from services.memory_models import MemoryLayer
from services.memory_retriever import MemoryRelevanceRetriever


def _check(name: str, passed: bool, detail: str = "") -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def evaluate() -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        store = MemoryEventStore(Path(tmp) / "memory.sqlite3")
        now = 1_000_000
        evidence = store.append(
            user_id="alice", layer=MemoryLayer.RAW_EVENT,
            kind="slate_feedback", source="slate_feedback", evidence_id="f1",
            payload={"rating": "negative", "note": "too noisy"}, now_ms=now,
        )
        inferred = store.append(
            user_id="alice", layer=MemoryLayer.INFERRED,
            kind="preference", source="memory_consolidator", evidence_id=evidence.record_id,
            payload={
                "field": "add_moods", "value": "Warm",
                "retrieval_cues": ["warm calm music for a rainy evening"],
            },
            confidence=0.85,
            memory_key="preference:add_moods:warm",
            expires_at=now + 50_000,
            now_ms=now + 1,
        )
        store.append(
            user_id="bob", layer=MemoryLayer.INFERRED,
            kind="preference", source="memory_consolidator", evidence_id="b1",
            payload={"field": "add_moods", "value": "Energetic"},
            confidence=0.9,
            memory_key="preference:add_moods:energetic",
            expires_at=now + 50_000,
            now_ms=now + 2,
        )

        active = store.effective_records(user_id="alice", now_ms=now + 10)
        selected = MemoryRelevanceRetriever(min_relevance=0.05).retrieve(
            query="calm rainy evening",
            records=active,
            max_facts=3,
            now_ms=now + 10,
        )
        expired = store.effective_records(user_id="alice", now_ms=now + 60_000)
        store.append(
            user_id="alice", layer=MemoryLayer.EXPLICIT,
            kind="preference", source="user_explicit", evidence_id="manual",
            payload={"field": "add_moods", "value": "Warm"},
            memory_key="preference:add_moods:warm",
            now_ms=now + 20,
        )
        explicit = store.effective_records(user_id="alice", now_ms=now + 30)

        checks = [
            _check("l0_is_evidence_not_preference", evidence.layer == MemoryLayer.RAW_EVENT),
            _check("cross_user_isolation", all(row.user_id == "alice" for row in active)),
            _check("relevance_selects_l2", bool(selected and selected[0].record.record_id == inferred.record_id)),
            _check("expired_l2_hidden", all(row.record_id != inferred.record_id for row in expired)),
            _check(
                "explicit_overrides_same_l2_key",
                sum(row.memory_key == inferred.memory_key for row in explicit) == 1
                and next(row for row in explicit if row.memory_key == inferred.memory_key).layer == MemoryLayer.EXPLICIT,
            ),
        ]
        passed = sum(1 for row in checks if row["passed"])
        return {"passed": passed, "total": len(checks), "checks": checks}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = evaluate()
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"Memory lifecycle eval: {report['passed']}/{report['total']} passed")
        for row in report["checks"]:
            print(f"- {'PASS' if row['passed'] else 'FAIL'} {row['name']}")
    return 0 if report["passed"] == report["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
