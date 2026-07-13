from services.memory_models import MemoryLayer, MemoryRecord, MemoryStatus
from services.memory_retriever import MemoryRelevanceRetriever


def _record(record_id, value, cues, created_at=1000):
    return MemoryRecord(
        record_id=record_id,
        user_id="u1",
        layer=MemoryLayer.INFERRED,
        kind="preference",
        source="memory_consolidator",
        evidence_id="e1",
        confidence=0.85,
        created_at=created_at,
        valid_from=created_at,
        status=MemoryStatus.ACTIVE,
        memory_key=f"preference:add_moods:{value.casefold()}",
        payload={
            "field": "add_moods",
            "value": value,
            "retrieval_cues": cues,
            "decision_summary": "validated preference",
        },
    )


def test_relevance_retriever_selects_query_related_memory():
    rainy = _record("rainy", "Warm", ["rainy calm soft music", "quiet on rainy days"])
    workout = _record("workout", "Energetic", ["energetic workout music", "strong gym rhythm"])
    retriever = MemoryRelevanceRetriever(min_relevance=0.05)

    results = retriever.retrieve(
        query="calm music for a rainy day",
        records=[workout, rainy],
        max_facts=1,
        now_ms=2000,
    )

    assert results
    assert results[0].record.record_id == "rainy"


def test_structured_retrieval_excludes_l3_by_default():
    l3 = _record("episode", "Warm", ["quiet rainy evening"])
    object.__setattr__(l3, "layer", MemoryLayer.EPISODIC)
    retriever = MemoryRelevanceRetriever(min_relevance=0.0)

    assert retriever.retrieve(query="rainy", records=[l3], include_episodic=False) == []
    assert retriever.retrieve(query="rainy", records=[l3], include_episodic=True)
