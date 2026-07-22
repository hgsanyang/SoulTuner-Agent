import asyncio

from services.memory_consolidator import MemoryConsolidator
from services.memory_event_store import MemoryEventStore
from services.memory_models import MemoryLayer


def _add_evidence(store, user_id, text, now):
    return store.append(
        user_id=user_id,
        layer=MemoryLayer.RAW_EVENT,
        kind="conversation_statement",
        source="user_statement",
        evidence_id="",
        payload={"user_text": text},
        now_ms=now,
    )


def test_consolidator_rejects_unknown_evidence_and_explicit_conflict(tmp_path):
    store = MemoryEventStore(tmp_path / "memory.sqlite3")
    first = _add_evidence(store, "u1", "first signal", 1000)
    second = _add_evidence(store, "u1", "second signal", 2000)
    store.append(
        user_id="u1",
        layer=MemoryLayer.EXPLICIT,
        kind="preference",
        source="user_explicit",
        evidence_id="manual",
        payload={"field": "avoid_moods", "value": "Sad"},
        memory_key="preference:avoid_moods:sad",
        now_ms=2500,
    )

    async def generator(_user_id, _evidence):
        return {
            "candidates": [
                {
                    "field": "add_moods",
                    "value": "Sad",
                    "confidence": 0.95,
                    "evidence_ids": [first.record_id, second.record_id],
                },
                {
                    "field": "add_genres",
                    "value": "Indie",
                    "confidence": 0.9,
                    "evidence_ids": [first.record_id, "another-user-record"],
                },
            ]
        }

    report = asyncio.run(
        MemoryConsolidator(store, generator=generator).consolidate(user_id="u1")
    )

    assert report.accepted == []
    reasons = {item.reason for item in report.rejected}
    assert "explicit_memory_takes_precedence" in reasons
    assert "unknown_or_cross_user_evidence" in reasons
    assert len(report.prompt_hash) == 64
    assert report.total_tokens == 0


def test_consolidator_abstains_with_insufficient_evidence(tmp_path):
    store = MemoryEventStore(tmp_path / "memory.sqlite3")
    _add_evidence(store, "u1", "one signal", 1000)

    async def should_not_run(_user_id, _evidence):
        raise AssertionError("generator should not run")

    report = asyncio.run(
        MemoryConsolidator(store, generator=should_not_run, min_evidence=2).consolidate(user_id="u1")
    )

    assert report.abstained is True
    assert report.accepted == []


def test_grouped_provider_payload_is_normalized_without_inventing_confidence():
    payload = MemoryConsolidator._normalize_llm_payload(
        {
            "candidates": [
                {
                    "add_moods": ["Calm", "Warm"],
                    "evidence_ids": ["e1", "e2"],
                    "decision_summary": "Repeated preference.",
                }
            ],
            "abstained": False,
        }
    )

    assert [(item["field"], item["value"]) for item in payload["candidates"]] == [
        ("add_moods", "Calm"),
        ("add_moods", "Warm"),
    ]
    assert all(item["confidence"] == 0.0 for item in payload["candidates"])
