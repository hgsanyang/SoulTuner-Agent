import asyncio

import pytest
from pydantic import ValidationError

from services.memory_consolidator import InferredPreferenceCandidate, MemoryConsolidator
from services.memory_event_store import MemoryEventStore
from services.memory_models import MemoryLayer


@pytest.mark.parametrize("scope,ttl,expected", [
    ("雨天在家工作", None, 45), ("雨天在家工作", 30, 30),
    ("  ", None, 45), ("global", None, 90), ("temporary", None, 14),
    ("雨天", 1, 7), ("雨天", 365, 180),
])
def test_free_scene_ttl_preserves_labels(tmp_path, scope, ttl, expected):
    candidate = InferredPreferenceCandidate(
        field="add_genres", value="Rock", confidence=.9, scope=scope, ttl_days=ttl,
    )
    consolidator = MemoryConsolidator(MemoryEventStore(tmp_path / "memory.db"), default_ttl_days=45)
    assert consolidator._bounded_ttl(candidate) == expected
    assert candidate.scope == (scope.strip() or "contextual")


@pytest.mark.parametrize("ttl", [0, -1, 366])
def test_out_of_schema_ttl_rejected(ttl):
    with pytest.raises(ValidationError):
        InferredPreferenceCandidate(field="add_genres", value="Rock", confidence=.9, ttl_days=ttl)


def test_free_scene_and_rejected_candidate_do_not_abort_batch(tmp_path):
    store = MemoryEventStore(tmp_path / "memory.db")
    evidence = [store.append(
        user_id="u", layer=MemoryLayer.RAW_EVENT, kind="conversation_statement",
        source="user_statement", evidence_id="", payload={"user_text": text},
    ) for text in ("雨天工作想听摇滚", "下雨工作时还是喜欢摇滚")]

    async def generate(*args):
        return {"candidates": [dict(
            field=field, value="Rock", scope="雨天在家工作", confidence=.95,
            evidence_ids=[e.record_id for e in evidence],
        ) for field in ("invalid_field", "add_genres")]}

    report = asyncio.run(MemoryConsolidator(store, generator=generate).consolidate(user_id="u"))
    assert len(report.accepted) == 1
    assert report.accepted[0].ttl_days == 45
    assert report.accepted[0].scope == "雨天在家工作"
    assert report.rejected[0].reason == "unsupported_field"


def test_malformed_candidate_does_not_discard_valid_neighbor():
    proposal = MemoryConsolidator._parse_proposal({"candidates": [
        {"field": "add_genres", "value": "Rock", "confidence": .9, "ttl_days": 999},
        {"field": "add_genres", "value": "Jazz", "confidence": .9},
        "invalid object",
    ]})
    assert [c.value for c in proposal.candidates] == ["Jazz"]
    assert len(proposal._invalid_candidates) == 2
    assert all(c.reason == "invalid_candidate_schema" for c in proposal._invalid_candidates)


def test_oversized_memory_batch_is_not_silently_truncated():
    with pytest.raises(ValueError):
        MemoryConsolidator._parse_proposal({"candidates": [{}] * 21})
