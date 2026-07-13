from services.memory_event_store import MemoryEventStore
from services.memory_models import MemoryLayer


def test_memory_ledger_is_append_only_and_tombstones_deletion(tmp_path):
    store = MemoryEventStore(tmp_path / "memory.sqlite3")
    record = store.append(
        user_id="u1",
        layer=MemoryLayer.EXPLICIT,
        kind="preference",
        source="user_explicit",
        evidence_id="e1",
        payload={"field": "add_moods", "value": "Warm"},
        memory_key="preference:add_moods:warm",
    )
    assert record is not None

    deleted = store.tombstone(user_id="u1", target_record_id=record.record_id)

    assert deleted is not None
    assert len(store.list_records(user_id="u1")) == 2
    assert store.effective_records(user_id="u1") == []


def test_memory_ledger_strictly_isolates_users(tmp_path):
    store = MemoryEventStore(tmp_path / "memory.sqlite3")
    first = store.append(
        user_id="u1", layer=MemoryLayer.RAW_EVENT, kind="like", source="user_action",
        evidence_id="e1", payload={"title": "A"},
    )
    store.append(
        user_id="u2", layer=MemoryLayer.RAW_EVENT, kind="like", source="user_action",
        evidence_id="e2", payload={"title": "B"},
    )

    assert first is not None
    assert [row.payload["title"] for row in store.list_records(user_id="u1")] == ["A"]
    assert store.get(user_id="u2", record_id=first.record_id) is None
    assert store.tombstone(user_id="u2", target_record_id=first.record_id) is None


def test_expired_inference_is_hidden_and_explicit_overrides_inferred(tmp_path):
    store = MemoryEventStore(tmp_path / "memory.sqlite3")
    now = 10_000
    store.append(
        user_id="u1", layer=MemoryLayer.INFERRED, kind="preference", source="inference",
        evidence_id="e1", payload={"field": "add_moods", "value": "Warm"},
        memory_key="preference:add_moods:warm", expires_at=20_000, now_ms=now,
    )
    store.append(
        user_id="u1", layer=MemoryLayer.EXPLICIT, kind="preference", source="user_explicit",
        evidence_id="e2", payload={"field": "add_moods", "value": "Warm"},
        memory_key="preference:add_moods:warm", now_ms=now + 1,
    )
    store.append(
        user_id="u1", layer=MemoryLayer.INFERRED, kind="preference", source="inference",
        evidence_id="e3", payload={"field": "avoid_moods", "value": "Sad"},
        memory_key="preference:avoid_moods:sad", expires_at=now - 1, now_ms=now,
    )

    effective = store.effective_records(user_id="u1", now_ms=now + 2)

    assert len(effective) == 1
    assert effective[0].layer == MemoryLayer.EXPLICIT


def test_eval_read_only_does_not_create_memory_database(tmp_path, monkeypatch):
    path = tmp_path / "memory.sqlite3"
    monkeypatch.setenv("EVAL_DISABLE_SIDE_EFFECTS", "1")
    store = MemoryEventStore(path)

    result = store.append(
        user_id="u1", layer=MemoryLayer.RAW_EVENT, kind="play", source="test",
        evidence_id="e1", payload={},
    )

    assert result is None
    assert not path.exists()


def test_reinforced_inference_exposes_only_newest_effective_record(tmp_path):
    store = MemoryEventStore(tmp_path / "memory.sqlite3")
    common = {
        "user_id": "u1",
        "layer": MemoryLayer.INFERRED,
        "kind": "preference",
        "source": "memory_consolidator",
        "payload": {"field": "add_moods", "value": "Warm"},
        "memory_key": "preference:add_moods:warm",
    }
    store.append(evidence_id="e1", confidence=0.75, now_ms=1000, **common)
    newest = store.append(evidence_id="e2", confidence=0.9, now_ms=2000, **common)

    effective = store.effective_records(user_id="u1", now_ms=3000)

    assert len(effective) == 1
    assert effective[0].record_id == newest.record_id
    assert effective[0].confidence == 0.9


def test_explicit_opposite_preference_suppresses_inferred_conflict(tmp_path):
    store = MemoryEventStore(tmp_path / "memory.sqlite3")
    store.append(
        user_id="u1", layer=MemoryLayer.INFERRED, kind="preference",
        source="memory_consolidator", evidence_id="e1",
        payload={"field": "add_moods", "value": "Sad"},
        memory_key="preference:add_moods:sad", now_ms=1000,
    )
    store.append(
        user_id="u1", layer=MemoryLayer.EXPLICIT, kind="preference",
        source="user_explicit", evidence_id="manual",
        payload={"field": "avoid_moods", "value": "Sad"},
        memory_key="preference:avoid_moods:sad", now_ms=2000,
    )

    effective = store.effective_records(user_id="u1", now_ms=3000)

    assert len(effective) == 1
    assert effective[0].layer == MemoryLayer.EXPLICIT
