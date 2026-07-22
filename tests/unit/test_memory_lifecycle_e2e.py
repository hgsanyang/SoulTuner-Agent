"""L1/L2/L3 生命周期端到端测试（gateway 级）。

覆盖交接文档 Batch C3/C4 的核心承诺：
- 显式修正会压制推断偏好，并把提到该值的旧 episodic 记忆一并失效（修正传播）；
- TTL 衰减对检索真实生效；
- 单条弱反馈不会直接变成永久偏好（最少证据门槛）；
- memory_trace 携带可审计的检索策略（阈值/版本/场景）。
"""

import asyncio

from services.memory_consolidator import MemoryConsolidator
from services.memory_event_store import MemoryEventStore
from services.memory_gateway import MemoryGateway
from services.memory_models import MemoryLayer
from services.memory_retriever import (
    DEFAULT_LAYER_THRESHOLDS,
    MemoryRelevanceRetriever,
)

DAY_MS = 86_400_000


class FakePrimary:
    def __init__(self):
        self.manager = self
        self.preferences = []
        self.removed = []

    def remember_event(self, event_type, title, artist, user_id, extra):
        pass

    def remember_preference(self, user_id, preferences):
        self.preferences.append((user_id, preferences))

    def get_user_profile(self, user_id, limit=30):
        return {}

    def delete_memory(self, user_id, *, title="", artist="", memory_type=""):
        return True

    def clear_learned_preferences(self, user_id):
        return True

    def remove_semantic_preference(self, user_id, field, value):
        self.removed.append((user_id, field, value))
        return True

    def delete_inferred_preference(self, user_id, **kwargs):
        return True


class UniformScorer:
    name = "uniform-test"

    def score(self, query, documents):
        return [0.9] * len(documents)


def _gateway(store: MemoryEventStore, **kwargs) -> MemoryGateway:
    return MemoryGateway(
        primary=FakePrimary(),
        event_store=store,
        relevance_retriever=MemoryRelevanceRetriever(
            semantic_scorer=UniformScorer(),
            layer_thresholds=DEFAULT_LAYER_THRESHOLDS,
        ),
        **kwargs,
    )


def _seed_inferred(store, *, field="add_genres", value="EDM", user_id="u1"):
    return store.append(
        user_id=user_id,
        layer=MemoryLayer.INFERRED,
        kind="preference",
        source="memory_consolidator",
        evidence_id="e1",
        confidence=0.9,
        payload={"field": field, "value": value, "retrieval_cues": ["电子舞曲"]},
        memory_key=f"preference:{field}:{value.casefold()}",
    )


def _seed_episode(store, *, description, user_id="u1", scope="contextual"):
    return store.append(
        user_id=user_id,
        layer=MemoryLayer.EPISODIC,
        kind="episode_summary",
        source="conversation",
        evidence_id="e-ep",
        confidence=0.65,
        payload={"description": description, "scope": scope},
    )


def _retrieved_values(gateway, *, query="来点音乐", user_id="u1", scene=""):
    context = asyncio.run(
        gateway.retrieve_context(query=query, user_id=user_id, scene=scene)
    )
    return [record["value"] or record.get("field") for record in context["retrieved_records"]], context


def test_explicit_correction_retires_inferred_and_matching_episodes(tmp_path):
    store = MemoryEventStore(tmp_path / "m.sqlite3")
    _seed_inferred(store, value="EDM")
    _seed_episode(store, description="用户上周在派对后想听 EDM 舞曲")
    _seed_episode(store, description="用户备考期间喜欢安静钢琴")
    gateway = _gateway(store)

    gateway.forget_preference_item(user_id="u1", field="add_genres", value="EDM")

    effective = store.effective_records(user_id="u1", limit=100)
    layers_and_text = [
        (record.layer, str(record.payload.get("description") or record.payload.get("value") or ""))
        for record in effective
    ]
    assert all("EDM" not in text for _, text in layers_and_text)
    # 无关 episode 不受修正影响
    assert any(record.layer == MemoryLayer.EPISODIC for record in effective)


def test_deleting_inferred_record_propagates_to_episodes(tmp_path):
    store = MemoryEventStore(tmp_path / "m.sqlite3")
    inferred = _seed_inferred(store, value="Heavy Metal")
    _seed_episode(store, description="那天深夜循环 Heavy Metal 现场版")
    gateway = _gateway(store)

    assert gateway.delete_memory_record(user_id="u1", record_id=inferred.record_id)

    remaining = store.effective_records(user_id="u1", limit=100)
    assert all(
        "Heavy Metal" not in str(record.payload.get("description") or "")
        and "Heavy Metal" != str(record.payload.get("value") or "")
        for record in remaining
    )


def test_expired_inferred_never_reaches_retrieval(tmp_path):
    now = 1_800_000_000_000
    store = MemoryEventStore(tmp_path / "m.sqlite3", clock_ms=lambda: now)
    store.append(
        user_id="u1",
        layer=MemoryLayer.INFERRED,
        kind="preference",
        source="memory_consolidator",
        evidence_id="e1",
        confidence=0.95,
        payload={"field": "add_moods", "value": "Nostalgic"},
        memory_key="preference:add_moods:nostalgic",
        expires_at=now - DAY_MS,
    )
    store.append(
        user_id="u1",
        layer=MemoryLayer.INFERRED,
        kind="preference",
        source="memory_consolidator",
        evidence_id="e2",
        confidence=0.8,
        payload={"field": "add_moods", "value": "Calm"},
        memory_key="preference:add_moods:calm",
        expires_at=now + 30 * DAY_MS,
    )
    values, _ = _retrieved_values(_gateway(store))
    assert "Calm" in values
    assert "Nostalgic" not in values


def test_single_weak_feedback_cannot_become_permanent_preference(tmp_path):
    store = MemoryEventStore(tmp_path / "m.sqlite3")

    def never_called(user_id, payload):
        raise AssertionError("consolidator must abstain before calling the model")

    consolidator = MemoryConsolidator(store, generator=never_called, min_evidence=2)
    gateway = _gateway(store, consolidator=consolidator, enable_consolidation=True)

    async def run():
        await gateway.remember_slate_feedback(
            exposure_id="exp-1", rating="off", reasons=["太吵"], user_id="u1"
        )
        return await gateway.consolidate_user(user_id="u1", force=True)

    report = asyncio.run(run())
    assert report.get("abstained") is True
    effective = store.effective_records(user_id="u1", limit=100)
    assert all(record.layer != MemoryLayer.INFERRED for record in effective)


def test_memory_trace_carries_policy_version_thresholds_and_scene(tmp_path):
    store = MemoryEventStore(tmp_path / "m.sqlite3")
    _seed_inferred(store, field="add_moods", value="Calm")
    gateway = _gateway(store)

    _, context = _retrieved_values(gateway, scene="focus")
    trace = context["memory_trace"]
    policy = trace["relevance_policy"]
    assert trace["scene"] == "focus"
    assert policy["policy_version"] == "open-calibration-v1"
    assert policy["layer_thresholds"]["L2"] == DEFAULT_LAYER_THRESHOLDS[MemoryLayer.INFERRED]
    assert policy["backend"] == "uniform-test"


def test_env_threshold_override_is_auditable(monkeypatch):
    monkeypatch.setenv("MEMORY_RELEVANCE_THRESHOLD_L2", "0.5")
    from services.memory_gateway import _configured_relevance_retriever

    policy = _configured_relevance_retriever().describe()
    assert policy["layer_thresholds"]["L2"] == 0.5
    assert policy["layer_thresholds"]["L1"] == DEFAULT_LAYER_THRESHOLDS[MemoryLayer.EXPLICIT]
