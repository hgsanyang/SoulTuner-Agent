from services.memory_models import MemoryLayer, MemoryRecord
from services.memory_retriever import RetrievedMemory
from tests.eval.memory_benchmark_worker import (
    _candidate_satisfies,
    _memory_can_discriminate,
    _memory_matches_candidate,
)


def _memory(field: str, value: str) -> RetrievedMemory:
    record = MemoryRecord(
        record_id="m1",
        user_id="u1",
        layer=MemoryLayer.EXPLICIT,
        kind="preference",
        source="test",
        evidence_id="e1",
        confidence=1.0,
        created_at=1,
        valid_from=1,
        payload={"field": field, "value": value},
    )
    return RetrievedMemory(record, score=1.0, relevance=1.0, why_used="test")


def test_hard_constraints_are_objective_and_fail_closed() -> None:
    candidate = {"language": "Cantonese", "year": 1995, "playable": True}

    assert _candidate_satisfies(candidate, {"language": "cantonese", "year_min": 1990})
    assert not _candidate_satisfies(candidate, {"language": "English"})
    assert not _candidate_satisfies({}, {"playable": True})


def test_memory_is_only_a_soft_candidate_adjustment() -> None:
    candidate = {"mood": "Calm"}

    assert _memory_matches_candidate(_memory("add_moods", "Calm"), candidate) > 0
    assert _memory_matches_candidate(_memory("avoid_moods", "Calm"), candidate) < 0
    assert _memory_matches_candidate(_memory("add_moods", "Energetic"), candidate) == 0


def test_memory_is_applicable_only_when_it_can_change_candidate_order() -> None:
    record = _memory("add_moods", "Calm").record

    assert _memory_can_discriminate(record, [{"mood": "Calm"}, {"mood": "Energetic"}])
    assert not _memory_can_discriminate(record, [{"mood": "Calm"}, {"mood": "Calm"}])
    assert not _memory_can_discriminate(record, [{"genre": "Rock"}, {"genre": "Folk"}])
