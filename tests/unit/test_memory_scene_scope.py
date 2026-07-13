"""自由场景标签与 L3 时序检索的行为测试。

原则：场景适用性完全由语义相关性裁决——场景标签是 LLM 自由命名的文本，
并入记忆语义文本参与打分；当前场景（若有）作为 query 增强。
确定性代码不做任何固定场景词表匹配。
"""

from services.memory_models import MemoryLayer, MemoryRecord
from services.memory_retriever import (
    DEFAULT_LAYER_THRESHOLDS,
    MemoryRelevanceRetriever,
    RELEVANCE_POLICY_VERSION,
)

DAY_MS = 86_400_000
NOW = 1_800_000_000_000


def _record(
    record_id: str,
    *,
    layer: MemoryLayer = MemoryLayer.INFERRED,
    scope: str = "global",
    value: str = "Calm",
    cues: list[str] | None = None,
    created_at: int = NOW - DAY_MS,
    confidence: float = 0.9,
) -> MemoryRecord:
    return MemoryRecord(
        record_id=record_id,
        user_id="u1",
        layer=layer,
        kind="preference",
        source="test",
        evidence_id="e1",
        confidence=confidence,
        created_at=created_at,
        valid_from=created_at,
        payload={
            "field": "add_moods",
            "value": value,
            "scope": scope,
            "retrieval_cues": cues if cues is not None else [],
        },
    )


class OverlapScorer:
    """Deterministic semantic stand-in: relevance = token overlap.

    High when any query token appears in the document, low otherwise —
    enough to exercise the semantic scene gating without a real model.
    """

    name = "overlap-test"

    def score(self, query, documents):
        tokens = [token for token in str(query).split() if token]
        return [
            0.9 if any(token in document for token in tokens) else 0.01
            for document in documents
        ]


class UniformScorer:
    name = "uniform-test"

    def __init__(self, relevance: float = 0.9):
        self.relevance = relevance

    def score(self, query, documents):
        return [self.relevance] * len(documents)


def _retriever(scorer=None, **kwargs) -> MemoryRelevanceRetriever:
    return MemoryRelevanceRetriever(
        semantic_scorer=scorer or UniformScorer(),
        layer_thresholds=DEFAULT_LAYER_THRESHOLDS,
        **kwargs,
    )


def test_free_scene_label_joins_semantic_text_and_gates_by_relevance():
    records = [
        _record("m-driving", scope="夜里一个人开车", value="Energetic"),
        _record("m-global", scope="global", value="Calm", cues=["听点歌"]),
    ]
    retriever = _retriever(scorer=OverlapScorer(), max_per_layer=2)

    # 睡前 query 与"开车"场景标签语义不相关 → 场景记忆被阈值挡住
    sleep = retriever.retrieve(query="睡前 听点歌", records=records, now_ms=NOW)
    assert [item.record.record_id for item in sleep] == ["m-global"]

    # 当前场景作为 query 增强后，场景记忆通过语义门槛
    driving = retriever.retrieve(
        query="听点歌", records=records, scene="夜里一个人开车", now_ms=NOW
    )
    assert {item.record.record_id for item in driving} == {"m-driving", "m-global"}


def test_scene_label_is_traceable_in_why_used():
    records = [_record("m1", scope="雨天在家工作", cues=["工作时听"])]
    selected = _retriever(scorer=OverlapScorer()).retrieve(
        query="工作时听", records=records, now_ms=NOW
    )
    assert selected
    assert "scene_label=雨天在家工作" in selected[0].why_used


def test_lifecycle_scopes_do_not_pollute_semantic_text():
    # contextual/global/temporary 是生命周期语义，不该作为语义文本参与匹配
    records = [_record("m1", scope="contextual", value="Warm", cues=[])]
    selected = _retriever(scorer=OverlapScorer()).retrieve(
        query="contextual", records=records, now_ms=NOW
    )
    assert selected == []


def test_episodic_memory_decays_faster_than_stable_preference():
    age = 30 * DAY_MS
    episodic = _record("m-episode", layer=MemoryLayer.EPISODIC, created_at=NOW - age)
    inferred = _record("m-inferred", layer=MemoryLayer.INFERRED, created_at=NOW - age)
    selected = _retriever(max_per_layer=2).retrieve(
        query="来点歌",
        records=[episodic, inferred],
        include_episodic=True,
        now_ms=NOW,
    )
    by_id = {item.record.record_id: item for item in selected}
    assert by_id["m-inferred"].score > by_id["m-episode"].score
    assert "episodic_half_life_days=14" in by_id["m-episode"].why_used


def test_fresh_episode_outranks_stale_episode():
    fresh = _record("m-fresh", layer=MemoryLayer.EPISODIC, created_at=NOW - 2 * DAY_MS)
    stale = _record("m-stale", layer=MemoryLayer.EPISODIC, created_at=NOW - 60 * DAY_MS)
    selected = _retriever(max_per_layer=2).retrieve(
        query="来点歌", records=[fresh, stale], include_episodic=True, now_ms=NOW
    )
    assert selected[0].record.record_id == "m-fresh"


def test_describe_exposes_auditable_policy():
    policy = _retriever(max_per_layer=1).describe()
    assert policy["policy_version"] == RELEVANCE_POLICY_VERSION
    assert policy["backend"] == "uniform-test"
    assert policy["layer_thresholds"]["L1"] == DEFAULT_LAYER_THRESHOLDS[MemoryLayer.EXPLICIT]
    assert policy["layer_thresholds"]["L2"] == DEFAULT_LAYER_THRESHOLDS[MemoryLayer.INFERRED]
    assert policy["max_per_layer"] == 1
