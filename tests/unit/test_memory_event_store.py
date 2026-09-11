from services.memory_event_store import MemoryEventStore
from services.memory_models import MemoryLayer, MemoryStatus


def test_old_explicit_survives_busy_ledger_and_suppresses_new_conflict(tmp_path):
    store = MemoryEventStore(tmp_path / "memory.sqlite3")
    explicit = store.append(
        user_id="u", layer=MemoryLayer.EXPLICIT, kind="preference", source="user_explicit",
        evidence_id="manual", payload={"field": "avoid_moods", "value": "Sad"},
        memory_key="preference:avoid_moods:sad", now_ms=1,
    )
    for i in range(410):
        store.append(user_id="u", layer=MemoryLayer.RAW_EVENT, kind="play", source="user_action",
                     evidence_id=str(i), payload={}, now_ms=2)
    inferred = store.append(
        user_id="u", layer=MemoryLayer.INFERRED, kind="preference", source="inference",
        evidence_id="guess", payload={"field": "add_moods", "value": "Sad"},
        memory_key="preference:add_moods:sad", now_ms=3,
    )
    rows = store.effective_records(user_id="u", now_ms=4, limit=2)
    assert rows[0].record_id == explicit.record_id
    assert inferred.record_id not in {r.record_id for r in rows}
    # Deletion removes both admission priority and conflict suppression.
    store.tombstone(user_id="u", target_record_id=explicit.record_id)
    assert store.effective_records(user_id="u", now_ms=4, limit=1)[0].record_id == inferred.record_id
    assert store.effective_records(user_id="u", now_ms=4, limit=0) == []


def test_expired_records_do_not_hide_older_live_memory(tmp_path):
    store = MemoryEventStore(tmp_path / "memory.sqlite3")
    live = store.append(user_id="u", layer=MemoryLayer.INFERRED, kind="preference",
                        source="inference", evidence_id="live", payload={}, now_ms=1)
    for i in range(410):
        store.append(user_id="u", layer=MemoryLayer.INFERRED, kind="preference",
                     source="inference", evidence_id=str(i), payload={}, now_ms=2, expires_at=3)
    assert [r.record_id for r in store.effective_records(user_id="u", now_ms=4)] == [live.record_id]


def test_evidence_window_counts_only_eligible_user_records(tmp_path):
    store = MemoryEventStore(tmp_path / "memory.sqlite3", clock_ms=lambda: 10)
    live = store.append(user_id="u", layer=MemoryLayer.RAW_EVENT, kind="like",
                        source="user_action", evidence_id="live", payload={}, now_ms=1)
    for i in range(25):
        store.append(user_id="u", layer=MemoryLayer.RAW_EVENT, kind="audit",
                     source="system", evidence_id=str(i), payload={}, now_ms=2)
    assert [r.record_id for r in store.recent_evidence(user_id="u", limit=1)] == [live.record_id]
    assert store.recent_evidence(user_id="u", limit=0) == []


def test_bulk_forget_has_no_listing_limit_and_preserves_other_users_and_layers(tmp_path):
    store = MemoryEventStore(tmp_path / "memory.sqlite3")
    for i in range(1005):
        store.append(user_id="u1", layer=MemoryLayer.INFERRED, kind="preference",
                     source="inference", evidence_id=str(i), payload={})
    for user, layer in (("u2", MemoryLayer.INFERRED), ("u1", MemoryLayer.EXPLICIT)):
        store.append(user_id=user, layer=layer, kind="preference", source="test", evidence_id="keep", payload={})
    assert store.tombstone_layer(user_id="u1", layer=MemoryLayer.INFERRED) == 1005
    assert store.tombstone_layer(user_id="u1", layer=MemoryLayer.INFERRED) == 0
    assert [r.layer for r in store.effective_records(user_id="u1", limit=2000)] == [MemoryLayer.EXPLICIT]
    assert len(store.effective_records(user_id="u2")) == 1


def test_deleted_evidence_cannot_return_through_a_layer_filtered_read(tmp_path):
    store = MemoryEventStore(tmp_path / "memory.sqlite3")
    record = store.append(user_id="u1", layer=MemoryLayer.RAW_EVENT, kind="like",
                          source="user_action", evidence_id="e", payload={"title": "A"})
    assert record is not None
    store.tombstone(user_id="u1", target_record_id=record.record_id)
    assert store.recent_evidence(user_id="u1") == []
    assert store.pending_evidence_count(user_id="u1") == 0


def test_future_memory_is_not_effective_yet(tmp_path):
    store = MemoryEventStore(tmp_path / "memory.sqlite3")
    store.append(user_id="u1", layer=MemoryLayer.RAW_EVENT, kind="like",
                 source="user_action", evidence_id="e", payload={}, now_ms=2000)
    assert store.effective_records(user_id="u1", now_ms=1000) == []


def test_pending_count_watermark_survives_listing_window(tmp_path):
    store = MemoryEventStore(tmp_path / "memory.sqlite3", clock_ms=lambda: 5000)
    store.append(user_id="u1", layer=MemoryLayer.RAW_EVENT, kind="consolidation_audit",
                 source="system", evidence_id="", payload={}, now_ms=2000)
    # Late-imported old events occur after the audit in ledger sequence order.
    for _ in range(12):
        store.append(user_id="u1", layer=MemoryLayer.RAW_EVENT, kind="like",
                     source="user_action", evidence_id="e", payload={}, now_ms=1000)
    assert store.pending_evidence_count(user_id="u1", limit=10) == 0
    store.append(user_id="u1", layer=MemoryLayer.RAW_EVENT, kind="like",
                 source="user_action", evidence_id="new", payload={}, now_ms=3000)
    assert store.pending_evidence_count(user_id="u1", limit=10) == 1


def test_pending_count_ignores_future_events_and_other_owner_audit(tmp_path):
    store = MemoryEventStore(tmp_path / "memory.sqlite3", clock_ms=lambda: 5000)
    for timestamp in (1000, 9000):
        store.append(user_id="u1", layer=MemoryLayer.RAW_EVENT, kind="like",
                     source="user_action", evidence_id="e", payload={}, now_ms=timestamp)
    store.append(user_id="u2", layer=MemoryLayer.RAW_EVENT, kind="consolidation_audit",
                 source="system", evidence_id="", payload={}, now_ms=4000)
    assert store.pending_evidence_count(user_id="u1") == 1


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


def test_injected_clock_and_id_factory_make_records_reproducible(tmp_path):
    store = MemoryEventStore(
        tmp_path / "memory.sqlite3",
        clock_ms=lambda: 12345,
        id_factory=lambda: "fixed-record-id",
    )

    record = store.append(
        user_id="u1", layer=MemoryLayer.RAW_EVENT, kind="like",
        source="user_action", evidence_id="e1", payload={"title": "A"},
    )

    assert record.record_id == "fixed-record-id"
    assert record.created_at == 12345


def test_tombstoned_l0_is_not_reused_as_consolidation_evidence(tmp_path):
    store = MemoryEventStore(tmp_path / "memory.sqlite3")
    record = store.append(
        user_id="u1", layer=MemoryLayer.RAW_EVENT, kind="conversation_statement",
        source="user_statement", evidence_id="e1", payload={"user_text": "I like jazz"},
    )
    store.tombstone(user_id="u1", target_record_id=record.record_id)

    assert store.recent_evidence(user_id="u1") == []
    assert store.pending_evidence_count(user_id="u1") == 0


def test_supersession_hides_the_target_record(tmp_path):
    ids = iter(["old", "replacement-marker"])
    store = MemoryEventStore(tmp_path / "memory.sqlite3", id_factory=lambda: next(ids))
    store.append(
        user_id="u1",
        layer=MemoryLayer.EXPLICIT,
        kind="preference",
        source="user_action",
        evidence_id="e1",
        payload={"field": "add_moods", "value": "Calm"},
        memory_key="preference:add_moods:calm",
        now_ms=1,
    )
    store.append(
        user_id="u1",
        layer=MemoryLayer.EXPLICIT,
        kind="supersession",
        source="user_action",
        evidence_id="e2",
        payload={"superseded_record_id": "old"},
        status=MemoryStatus.SUPERSEDED,
        target_record_id="old",
        now_ms=2,
    )

    assert store.effective_records(user_id="u1", now_ms=3) == []


def test_reinforcement_preserves_canonical_memory_identity(tmp_path):
    ids = iter(["canonical", "revision"])
    store = MemoryEventStore(tmp_path / "memory.sqlite3", id_factory=lambda: next(ids))
    common = {
        "user_id": "u1",
        "layer": MemoryLayer.EXPLICIT,
        "kind": "preference",
        "source": "user_action",
        "payload": {"field": "add_moods", "value": "Calm"},
        "memory_key": "preference:add_moods:calm",
    }
    first = store.append(evidence_id="e1", now_ms=1, **common)
    second = store.append(evidence_id="e2", now_ms=2, **common)

    assert first.payload["canonical_memory_id"] == "canonical"
    assert second.record_id == "revision"
    assert second.payload["canonical_memory_id"] == "canonical"
