from services.memory_models import MemoryLayer, MemoryRecord
from services.memory_retriever import MemoryRelevanceRetriever


class _FakeSemanticScorer:
    name = "fake-semantic"

    def score(self, query: str, documents: list[str]) -> list[float]:
        del query
        return [0.9 if "Calm" in document else 0.1 for document in documents]


def test_retriever_selects_explicit_memory_and_preserves_evidence() -> None:
    record = MemoryRecord(
        record_id="m1",
        user_id="u1",
        layer=MemoryLayer.EXPLICIT,
        kind="preference",
        source="user_explicit",
        evidence_id="e1",
        confidence=1.0,
        created_at=1,
        valid_from=1,
        memory_key="preference:add_moods:calm",
        payload={
            "field": "add_moods",
            "value": "Calm",
            "scope": "global",
            "retrieval_cues": ["quiet calming music"],
        },
    )

    selected = MemoryRelevanceRetriever().retrieve(
        query="quiet music",
        records=[record],
        now_ms=2,
    )

    assert len(selected) == 1
    assert selected[0].model_dump()["memory_id"] == "m1"
    assert selected[0].model_dump()["evidence_ids"] == ["e1"]


def test_retriever_can_use_a_semantic_backend_without_keyword_rules() -> None:
    record = MemoryRecord(
        record_id="m1",
        user_id="u1",
        layer=MemoryLayer.EXPLICIT,
        kind="preference",
        source="user_explicit",
        evidence_id="e1",
        confidence=1.0,
        created_at=1,
        valid_from=1,
        payload={"field": "add_moods", "value": "Calm"},
    )
    retriever = MemoryRelevanceRetriever(
        min_relevance=0.5,
        semantic_scorer=_FakeSemanticScorer(),
    )

    selected = retriever.retrieve(query="跨语言表达", records=[record], now_ms=2)

    assert retriever.backend_name == "fake-semantic"
    assert selected[0].relevance == 0.9


def test_retriever_applies_separate_layer_thresholds_and_top_one_per_layer() -> None:
    records = [
        MemoryRecord(
            record_id=f"m{index}",
            user_id="u1",
            layer=layer,
            kind="preference",
            source="test",
            evidence_id=f"e{index}",
            confidence=1.0,
            created_at=index,
            valid_from=index,
            payload={"field": "add_moods", "value": value},
        )
        for index, (layer, value) in enumerate(
            [
                (MemoryLayer.EXPLICIT, "Calm"),
                (MemoryLayer.EXPLICIT, "Calm again"),
                (MemoryLayer.INFERRED, "Calm inferred"),
            ],
            start=1,
        )
    ]

    class _Scores:
        name = "scores"

        def score(self, query: str, documents: list[str]) -> list[float]:
            del query, documents
            return [0.9, 0.8, 0.7]

    selected = MemoryRelevanceRetriever(
        semantic_scorer=_Scores(),
        layer_thresholds={MemoryLayer.EXPLICIT: 0.85, MemoryLayer.INFERRED: 0.65},
        max_per_layer=1,
    ).retrieve(query="calm", records=records, now_ms=4)

    assert [item.record.record_id for item in selected] == ["m1", "m3"]
