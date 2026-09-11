from types import SimpleNamespace

import pytest

from services.memory_event_store import MemoryEventStore
from services.memory_gateway import MemoryGateway
from services.memory_models import MemoryLayer


def test_projection_failure_is_not_success_and_recovery_survives_restart(tmp_path):
    path = tmp_path / "memory.sqlite3"
    store = MemoryEventStore(path)
    record = store.append(user_id="alice", layer=MemoryLayer.INFERRED, kind="preference",
                         source="test", evidence_id="e", payload={"value": "Rock"}, memory_key="rock")
    manager = SimpleNamespace(delete_inferred_preference=lambda *a, **k: False)
    gateway = MemoryGateway(primary=SimpleNamespace(manager=manager), event_store=store,
                            enable_graphzep_sidecar=False)
    with pytest.raises(RuntimeError, match="pending"):
        gateway.delete_memory_record(user_id="alice", record_id=record.record_id)
    assert store.pending_deletions(user_id="alice") == [record.record_id]
    assert store.pending_deletions(user_id="bob") == []
    assert store.effective_records(user_id="alice")[0].record_id == record.record_id

    reopened = MemoryEventStore(path)
    manager.delete_inferred_preference = lambda *a, **k: True
    recovered = MemoryGateway(primary=SimpleNamespace(manager=manager), event_store=reopened,
                              enable_graphzep_sidecar=False)
    assert recovered.retry_pending_deletions(user_id="alice") == {"completed": 1, "pending": 0}
    assert reopened.pending_deletions(user_id="alice") == []
    assert reopened.effective_records(user_id="alice") == []
    def already_done(*args, **kwargs):
        raise AssertionError("completed deletions must not replay against new preferences")
    manager.delete_inferred_preference = already_done
    assert recovered.delete_memory_record(user_id="alice", record_id=record.record_id)
    assert len(reopened.list_records(user_id="alice")) == 2


def test_deletion_intent_cannot_claim_other_users_record(tmp_path):
    store = MemoryEventStore(tmp_path / "memory.sqlite3")
    record = store.append(user_id="alice", layer=MemoryLayer.EPISODIC, kind="episode",
                         source="test", evidence_id="e", payload={})
    assert not store.stage_deletion(user_id="bob", record_id=record.record_id)
    assert store.pending_deletions(user_id="bob") == []


def test_lineage_deletion_never_resurrects_old_versions_or_deletes_newer_ones(tmp_path):
    store = MemoryEventStore(tmp_path / "memory.sqlite3")
    records = [store.append(user_id="alice", layer=MemoryLayer.INFERRED, kind="preference",
                            source="test", evidence_id=str(i), payload={}, memory_key="rock")
               for i in range(3)]
    assert store.tombstone_lineage(user_id="alice", record_id=records[1].record_id)
    assert [r.record_id for r in store.effective_records(user_id="alice")] == [records[2].record_id]
    assert store.tombstone_lineage(user_id="alice", record_id=records[2].record_id)
    assert store.effective_records(user_id="alice") == []


def test_retry_of_older_explicit_version_preserves_new_preference(tmp_path):
    store = MemoryEventStore(tmp_path / "memory.sqlite3")
    records = [store.append(user_id="alice", layer=MemoryLayer.EXPLICIT, kind="preference",
                            source="test", evidence_id=str(i),
                            payload={"field": "add_genres", "value": "Rock"}, memory_key="rock")
               for i in range(2)]
    def must_not_delete(*args, **kwargs):
        raise AssertionError("new preference must not be removed")
    gateway = MemoryGateway(primary=SimpleNamespace(manager=SimpleNamespace(
        remove_semantic_preference=must_not_delete)), event_store=store, enable_graphzep_sidecar=False)
    assert gateway.delete_memory_record(user_id="alice", record_id=records[0].record_id)
    assert [r.record_id for r in store.effective_records(user_id="alice")] == [records[1].record_id]
