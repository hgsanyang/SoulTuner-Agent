"""Retrieval-side memory v3 (MV3-3): L3 temporal, contradiction suppression,
link-aware expansion, and the silence-decision trace."""

from services.memory_models import MemoryLayer, MemoryRecord
from services.memory_retriever import DEFAULT_LAYER_THRESHOLDS, MemoryRelevanceRetriever

DAY = 86_400_000
NOW = 1_800_000_000_000


class UniformScorer:
    name = "uniform-test"

    def score(self, query, documents):
        return [0.9] * len(documents)


def _rec(
    rid,
    *,
    layer=MemoryLayer.INFERRED,
    value="Calm",
    canonical=None,
    created_at=NOW - DAY,
    payload_extra=None,
):
    payload = {"field": "add_moods", "value": value}
    if canonical:
        payload["canonical_memory_id"] = canonical
    if payload_extra:
        payload.update(payload_extra)
    return MemoryRecord(
        record_id=rid, user_id="u1", layer=layer, kind="preference", source="t",
        evidence_id="e", confidence=0.9, created_at=created_at, valid_from=created_at,
        payload=payload,
    )


def _retriever(**kw):
    return MemoryRelevanceRetriever(
        semantic_scorer=UniformScorer(), layer_thresholds=DEFAULT_LAYER_THRESHOLDS, **kw
    )


def test_episode_uses_occurred_at_not_write_time():
    # written now, but the event occurred 60 days ago → should decay as old
    fresh_event = _rec("e1", layer=MemoryLayer.EPISODIC, value="Recent",
                       created_at=NOW, payload_extra={"occurred_at": NOW - 2 * DAY})
    old_event = _rec("e2", layer=MemoryLayer.EPISODIC, value="Old",
                     created_at=NOW, payload_extra={"occurred_at": NOW - 60 * DAY})
    selected = _retriever(max_per_layer=2).retrieve(
        query="来点歌", records=[fresh_event, old_event], include_episodic=True, now_ms=NOW
    )
    assert selected[0].record.record_id == "e1"  # fresh occurrence outranks old


def test_expired_episode_is_never_injected():
    expired = _rec("e1", layer=MemoryLayer.EPISODIC, value="Exam",
                   payload_extra={"occurred_at": NOW - 20 * DAY, "valid_until": NOW - DAY})
    live = _rec("e2", layer=MemoryLayer.EPISODIC, value="Trip",
                payload_extra={"occurred_at": NOW - DAY, "valid_until": NOW + 10 * DAY})
    selected = _retriever(max_per_layer=3).retrieve(
        query="q", records=[expired, live], include_episodic=True, now_ms=NOW
    )
    ids = {item.record.record_id for item in selected}
    assert "e2" in ids and "e1" not in ids


def test_contradicted_memory_is_suppressed_at_read_time():
    old = _rec("m1", value="EDM", canonical="m1")
    newer = _rec("m2", value="Folk", canonical="m2",
                 payload_extra={"links": [{"target_memory_id": "m1", "relation": "contradicts", "reason": "x"}]})
    selected = _retriever(max_per_layer=5).retrieve(query="q", records=[old, newer], now_ms=NOW)
    values = {str(item.record.payload.get("value")) for item in selected}
    assert "Folk" in values and "EDM" not in values


def test_same_scene_link_expands_bounded():
    trace = {}
    a = _rec("m1", value="Calm", canonical="m1",
             payload_extra={"links": [{"target_memory_id": "m2", "relation": "same_scene", "reason": "x"}]})
    b = _rec("m2", value="Quiet", canonical="m2")
    selected = _retriever(max_per_layer=1).retrieve(
        query="q", records=[a, b], now_ms=NOW, trace=trace
    )
    values = [str(item.record.payload.get("value")) for item in selected]
    # max_per_layer=1 would keep only one, but the same_scene link pulls in m2
    assert "Quiet" in values
    assert trace["added_via_link"] >= 1
    assert any("via_link" in item.why_used for item in selected)


def test_silence_trace_reports_suppression_counts():
    trace = {}
    old = _rec("m1", value="EDM", canonical="m1")
    newer = _rec("m2", value="Folk", canonical="m2",
                 payload_extra={"links": [{"target_memory_id": "m1", "relation": "contradicts", "reason": "x"}]})
    _retriever(max_per_layer=5).retrieve(query="q", records=[old, newer], now_ms=NOW, trace=trace)
    assert trace["considered"] == 2
    assert trace["suppressed_contradicted"] == 1
    assert trace["selected"] == 1
    assert trace["stayed_silent"] is False


def test_silence_trace_flags_full_silence_when_all_suppressed():
    trace = {}
    expired = _rec("e1", layer=MemoryLayer.EPISODIC, value="Old",
                   payload_extra={"occurred_at": NOW - 40 * DAY, "valid_until": NOW - DAY})
    _retriever(max_per_layer=3).retrieve(
        query="q", records=[expired], include_episodic=True, now_ms=NOW, trace=trace
    )
    assert trace["considered"] == 1
    assert trace["suppressed_expired"] == 1
    assert trace["stayed_silent"] is True
