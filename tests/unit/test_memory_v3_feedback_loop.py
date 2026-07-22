"""Feedback -> memory correction loop E2E (memory v3 MV3-4)."""

import asyncio

from services.memory_consolidator import MemoryConsolidator, MemoryConsolidationProposal
from services.memory_event_store import MemoryEventStore
from services.memory_gateway import MemoryGateway, _feedback_polarity
from services.memory_models import MemoryLayer


class FakePrimary:
    def __init__(self):
        self.manager = self
        self.removed = []

    def remember_event(self, *a, **k):
        pass

    def remember_preference(self, *a, **k):
        pass

    def get_user_profile(self, user_id, limit=30):
        return {}

    def delete_memory(self, user_id, *, title="", artist="", memory_type=""):
        return True

    def clear_learned_preferences(self, user_id):
        return True

    def remove_semantic_preference(self, user_id, field, value):
        self.removed.append((field, value))
        return True

    def delete_inferred_preference(self, user_id, **kwargs):
        return True


def _gateway(store, consolidator=None):
    return MemoryGateway(
        primary=FakePrimary(), event_store=store, consolidator=consolidator,
        enable_consolidation=consolidator is not None,
    )


# ---- polarity classification ----

def test_feedback_polarity_classification():
    assert _feedback_polarity("great", []) == "positive"
    assert _feedback_polarity("off", []) == "negative"
    assert _feedback_polarity("partial", []) == "negative"
    assert _feedback_polarity("partial", ["太吵"]) == "negative"  # partial + reason chip
    assert _feedback_polarity("great", ["太吵"]) == "positive"  # explicit positive rating wins
    assert _feedback_polarity("", []) == "neutral"


def test_slate_feedback_records_polarity(tmp_path):
    store = MemoryEventStore(tmp_path / "m.sqlite3")
    gateway = _gateway(store)
    asyncio.run(gateway.remember_slate_feedback(
        exposure_id="exp-1", rating="off", reasons=["太吵", "太悲伤"], user_id="u1"
    ))
    l0 = [r for r in store.list_records(user_id="u1", limit=50) if r.kind == "slate_feedback"]
    assert l0 and l0[0].payload["polarity"] == "negative"
    assert l0[0].payload["reasons"] == ["太吵", "太悲伤"]


# ---- one weak feedback must not become a permanent preference ----

def test_single_weak_negative_feedback_abstains(tmp_path):
    store = MemoryEventStore(tmp_path / "m.sqlite3")

    def must_not_call(user_id, payload):
        raise AssertionError("consolidator must abstain before invoking the model")

    consolidator = MemoryConsolidator(store, generator=must_not_call, min_evidence=2)
    gateway = _gateway(store, consolidator)

    async def run():
        await gateway.remember_slate_feedback(exposure_id="e", rating="off", reasons=["太吵"], user_id="u1")
        return await gateway.consolidate_user(user_id="u1", force=True)

    report = asyncio.run(run())
    assert report.get("abstained") is True
    effective = store.effective_records(user_id="u1", limit=50)
    assert all(r.layer != MemoryLayer.INFERRED for r in effective)


# ---- repeated negative feedback becomes counter-evidence (avoid preference) ----

def test_repeated_negative_feedback_can_form_avoid_preference(tmp_path):
    store = MemoryEventStore(tmp_path / "m.sqlite3")
    seen_evidence = {}

    def generator(user_id, payload):
        # the consolidator sees the negative feedback evidence with polarity
        seen_evidence["payload"] = payload
        negatives = [e for e in payload if e["payload"].get("polarity") == "negative"]
        assert len(negatives) >= 2
        return MemoryConsolidationProposal(candidates=[{
            "field": "avoid_moods", "value": "Energetic", "scope": "global", "confidence": 0.9,
            "evidence_ids": [e["record_id"] for e in negatives[:2]], "counter_evidence_ids": [],
            "ttl_days": 45, "retrieval_cues": ["不要太吵"], "decision_summary": "repeated 太吵",
        }])

    consolidator = MemoryConsolidator(store, generator=generator, min_evidence=2)
    gateway = _gateway(store, consolidator)

    async def run():
        await gateway.remember_slate_feedback(exposure_id="e1", rating="off", reasons=["太吵"], user_id="u1")
        await gateway.remember_slate_feedback(exposure_id="e2", rating="too_noisy", reasons=["太吵"], user_id="u1")
        return await gateway.consolidate_user(user_id="u1", force=True)

    report = asyncio.run(run())
    assert report.get("accepted")
    values = {c["value"] for c in report["accepted"]}
    assert "Energetic" in values


# ---- explicit correction supersedes + propagates to linked episodes ----

def test_explicit_correction_supersedes_and_invalidates_episodes(tmp_path):
    store = MemoryEventStore(tmp_path / "m.sqlite3")
    store.append(
        user_id="u1", layer=MemoryLayer.INFERRED, kind="preference",
        source="memory_consolidator", evidence_id="e1", confidence=0.9,
        payload={"field": "add_genres", "value": "EDM"},
        memory_key="preference:add_genres:edm",
    )
    store.append(
        user_id="u1", layer=MemoryLayer.EPISODIC, kind="episode_summary",
        source="conversation", evidence_id="e2", confidence=0.6,
        payload={"description": "那天派对循环 EDM", "scope": "contextual"},
    )
    gateway = _gateway(store)

    ok = gateway.forget_preference_item(user_id="u1", field="add_genres", value="EDM")
    assert ok

    effective = store.effective_records(user_id="u1", limit=100)
    # both the inferred preference and the EDM-mentioning episode are gone
    assert all("EDM" not in str(r.payload.get("value") or "") for r in effective)
    assert all("EDM" not in str(r.payload.get("description") or "") for r in effective)
