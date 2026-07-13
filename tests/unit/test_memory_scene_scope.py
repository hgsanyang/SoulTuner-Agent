"""场景作用域与 L3 时序检索的确定性行为测试。

核心不变量：
1. 场景绑定的记忆只在同场景生效；场景未知时 fail-closed 不注入；
2. global/contextual/temporary 记忆不受场景影响；
3. L3 episodic 记忆时间衰减快于稳定偏好，旧事件不能压过新事件；
4. 检索策略（阈值/版本/backend）可审计。
"""

from services.memory_models import MemoryLayer, MemoryRecord
from services.memory_retriever import (
    DEFAULT_LAYER_THRESHOLDS,
    MemoryRelevanceRetriever,
    RELEVANCE_POLICY_VERSION,
    normalize_scene,
)

DAY_MS = 86_400_000
NOW = 1_800_000_000_000


def _record(
    record_id: str,
    *,
    layer: MemoryLayer = MemoryLayer.INFERRED,
    scope: str = "global",
    value: str = "Calm",
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
            "retrieval_cues": ["安静放松的音乐"],
        },
    )


class UniformScorer:
    """Semantic scorer stub: every document is equally relevant."""

    name = "uniform-test"

    def __init__(self, relevance: float = 0.9):
        self.relevance = relevance

    def score(self, query, documents):
        return [self.relevance] * len(documents)


def _retriever(**kwargs) -> MemoryRelevanceRetriever:
    return MemoryRelevanceRetriever(
        semantic_scorer=UniformScorer(),
        layer_thresholds=DEFAULT_LAYER_THRESHOLDS,
        **kwargs,
    )


def test_scene_scoped_memory_only_applies_in_its_scene():
    records = [
        _record("m-driving", scope="driving", value="Energetic"),
        _record("m-global", scope="global", value="Calm"),
    ]
    retriever = _retriever()

    in_sleep = retriever.retrieve(query="睡前听点歌", records=records, scene="sleep", now_ms=NOW)
    assert [item.record.record_id for item in in_sleep] == ["m-global"]

    in_driving = retriever.retrieve(query="开车听点歌", records=records, scene="driving", now_ms=NOW)
    assert {item.record.record_id for item in in_driving} == {"m-driving", "m-global"}


def test_unknown_scene_fails_closed_for_scene_scoped_memory():
    records = [
        _record("m-rainy", scope="rainy", value="Melancholy"),
        _record("m-contextual", scope="contextual", value="Warm"),
    ]
    selected = _retriever().retrieve(query="来点歌", records=records, scene="", now_ms=NOW)
    assert [item.record.record_id for item in selected] == ["m-contextual"]


def test_matching_scene_gets_bounded_boost_and_traceable_reason():
    records = [
        _record("m-focus", scope="focus", value="Instrumental", confidence=0.8),
        _record("m-global", scope="global", value="Calm", confidence=0.8),
    ]
    selected = _retriever(max_per_layer=2).retrieve(
        query="工作时听的歌", records=records, scene="focus", now_ms=NOW
    )
    by_id = {item.record.record_id: item for item in selected}
    assert by_id["m-focus"].score > by_id["m-global"].score
    assert "scene=focus" in by_id["m-focus"].why_used
    # bounded: boost never exceeds 0.05
    assert by_id["m-focus"].score - by_id["m-global"].score <= 0.05 + 1e-9


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


def test_normalize_scene_maps_structured_labels_only():
    assert normalize_scene("Driving") == "driving"
    assert normalize_scene("Study") == "focus"
    assert normalize_scene("Late Night") == "late_night"
    assert normalize_scene("Rainy Day") == "rainy"
    assert normalize_scene("完全未知的场景") == ""
    assert normalize_scene(None) == ""
