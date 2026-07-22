"""A-MEM linking write-side + L3 episode temporal write (memory v3 MV3-2)."""

import asyncio

from services.memory_consolidator import MemoryConsolidator, MemoryConsolidationProposal
from services.memory_event_store import MemoryEventStore
from services.memory_gateway import MemoryGateway
from services.memory_models import MemoryLayer, MemoryStatus


class FakePrimary:
    def __init__(self):
        self.manager = self

    def remember_event(self, *a, **k):
        pass

    def remember_preference(self, *a, **k):
        pass

    def remember_inferred_preference(self, user_id, record):
        return True

    def get_user_profile(self, user_id, limit=30):
        return {}

    def clear_learned_preferences(self, user_id):
        return True


def _seed_l2(store, *, field="add_moods", value="Calm", user_id="u1"):
    return store.append(
        user_id=user_id,
        layer=MemoryLayer.INFERRED,
        kind="preference",
        source="memory_consolidator",
        evidence_id="e0",
        confidence=0.9,
        payload={"field": field, "value": value},
        memory_key=f"preference:{field}:{value.casefold()}",
    )


def _l0(store, *, value, user_id="u1", eid):
    return store.append(
        user_id=user_id,
        layer=MemoryLayer.RAW_EVENT,
        kind="conversation_statement",
        source="user_statement",
        evidence_id=eid,
        confidence=1.0,
        payload={"user_text": value},
    )


def test_consolidator_only_keeps_links_to_real_active_memories(tmp_path):
    store = MemoryEventStore(tmp_path / "m.sqlite3")
    seed = _seed_l2(store, field="add_moods", value="Calm")
    canonical = seed.record_id
    e1 = _l0(store, value="喜欢温暖", eid="ev1")
    e2 = _l0(store, value="喜欢温暖治愈", eid="ev2")

    def generator(user_id, payload):
        return MemoryConsolidationProposal(candidates=[{
            "field": "add_moods",
            "value": "Warm",
            "scope": "global",
            "confidence": 0.9,
            "evidence_ids": [e1.record_id, e2.record_id],
            "counter_evidence_ids": [],
            "ttl_days": 60,
            "retrieval_cues": ["warm"],
            "decision_summary": "repeated warm evidence",
            "links": [
                {"target_memory_id": canonical, "relation": "refines", "reason": "narrows Calm"},
                {"target_memory_id": "ghost-id", "relation": "same_scene", "reason": "bogus"},
            ],
        }])

    consolidator = MemoryConsolidator(store, generator=generator, min_evidence=2)
    report = asyncio.run(consolidator.consolidate(user_id="u1"))
    assert len(report.accepted) == 1
    links = report.accepted[0].links
    # bogus target dropped, real one kept
    assert [link_["target_memory_id"] for link_ in links] == [canonical]
    assert links[0]["relation"] == "refines"


def test_refines_link_supersedes_target_via_gateway(tmp_path):
    store = MemoryEventStore(tmp_path / "m.sqlite3")
    seed = _seed_l2(store, field="add_moods", value="Calm")
    e1 = _l0(store, value="喜欢温暖", eid="ev1")
    e2 = _l0(store, value="喜欢温暖治愈", eid="ev2")

    def generator(user_id, payload):
        return MemoryConsolidationProposal(candidates=[{
            "field": "add_moods", "value": "Warm", "scope": "global", "confidence": 0.9,
            "evidence_ids": [e1.record_id, e2.record_id], "counter_evidence_ids": [],
            "ttl_days": 60, "retrieval_cues": ["warm"], "decision_summary": "evolve",
            "links": [{"target_memory_id": seed.record_id, "relation": "evolves_from", "reason": "later"}],
        }])

    consolidator = MemoryConsolidator(store, generator=generator, min_evidence=2)
    gateway = MemoryGateway(
        primary=FakePrimary(), event_store=store, consolidator=consolidator,
        enable_consolidation=True,
    )
    report = asyncio.run(gateway.consolidate_user(user_id="u1", force=True))
    assert not report.get("skipped")

    effective = store.effective_records(user_id="u1", limit=100)
    values = {str(r.payload.get("value")) for r in effective if r.layer == MemoryLayer.INFERRED}
    # the evolved-from Calm is superseded; only Warm remains effective
    assert "Warm" in values
    assert "Calm" not in values
    # history preserved: a supersede tombstone exists in the raw ledger
    all_records = store.list_records(user_id="u1", limit=200)
    assert any(r.status == MemoryStatus.SUPERSEDED for r in all_records)


def test_contradicts_link_does_not_delete_target(tmp_path):
    store = MemoryEventStore(tmp_path / "m.sqlite3")
    seed = _seed_l2(store, field="add_genres", value="EDM")
    e1 = _l0(store, value="不太喜欢电子了", eid="ev1")
    e2 = _l0(store, value="最近想听民谣", eid="ev2")

    def generator(user_id, payload):
        return MemoryConsolidationProposal(candidates=[{
            "field": "add_genres", "value": "Folk", "scope": "global", "confidence": 0.9,
            "evidence_ids": [e1.record_id, e2.record_id], "counter_evidence_ids": [],
            "ttl_days": 60, "retrieval_cues": ["folk"], "decision_summary": "shift",
            "links": [{"target_memory_id": seed.record_id, "relation": "contradicts", "reason": "opposite"}],
        }])

    consolidator = MemoryConsolidator(store, generator=generator, min_evidence=2)
    gateway = MemoryGateway(
        primary=FakePrimary(), event_store=store, consolidator=consolidator,
        enable_consolidation=True,
    )
    asyncio.run(gateway.consolidate_user(user_id="u1", force=True))

    effective = store.effective_records(user_id="u1", limit=100)
    values = {str(r.payload.get("value")) for r in effective if r.layer == MemoryLayer.INFERRED}
    # contradiction does not silently delete; both remain (suppression is read-side)
    assert "EDM" in values and "Folk" in values


def test_remember_text_captures_episode_temporal_fields(tmp_path):
    store = MemoryEventStore(tmp_path / "m.sqlite3")
    gateway = MemoryGateway(primary=FakePrimary(), event_store=store)
    now = 1_800_000_000_000

    asyncio.run(gateway.remember_text(
        description="上周末通宵备考很焦虑",
        user_id="u1",
        extra={"scene": "备考", "mood": "焦虑", "occurred_at": now - 3 * 86_400_000,
               "valid_until": now + 10 * 86_400_000, "source": "conversation"},
    ))
    episodes = [r for r in store.list_records(user_id="u1", limit=50) if r.layer == MemoryLayer.EPISODIC]
    assert episodes
    payload = episodes[0].payload
    assert payload["occurred_at"] == now - 3 * 86_400_000
    assert payload["valid_until"] == now + 10 * 86_400_000
    assert payload["mood"] == "焦虑"
    assert payload["scene"] == "备考"
