from types import SimpleNamespace

import pytest

from services.memory_event_store import MemoryEventStore
from services.memory_gateway import MemoryGateway
from services.memory_models import MemoryLayer
from retrieval.user_memory import UserMemoryManager


class Projection:
    def __init__(self):
        self.rows = {}
        self.offline = False
        self.snapshots = 0

    def snapshot_learned_preferences(self, user):
        self.snapshots += 1
        return [dict(row) for (owner, _), row in self.rows.items() if owner == user]

    def delete_learned_snapshot(self, user, targets):
        if self.offline:
            raise ConnectionError("offline")
        for target in targets:
            key = (user, target["memory_key"])
            if self.rows.get(key) == target:
                self.rows.pop(key)
        return True


def append(store, projection, user, key, layer=MemoryLayer.INFERRED):
    record = store.append(user_id=user, layer=layer, kind="preference", source="test",
                          evidence_id="e", payload={"field": "add_moods", "value": key}, memory_key=key)
    if layer == MemoryLayer.INFERRED:
        projection.rows[user, key] = {"memory_key": key, "ledger_record_id": record.record_id, "updated_at": 1}
    return record


@pytest.mark.parametrize("failure", ["remote", "ledger"])
def test_bulk_recovery_preserves_later_versions_and_other_owners(tmp_path, monkeypatch, failure):
    path = tmp_path / "memory.sqlite3"
    store, projection = MemoryEventStore(path), Projection()
    append(store, projection, "alice", "old")
    keep = append(store, projection, "alice", "explicit", MemoryLayer.EXPLICIT)
    append(store, projection, "bob", "old")
    gateway = MemoryGateway(primary=projection, event_store=store, enable_consolidation=False)
    if failure == "remote":
        projection.offline = True
    else:
        def fail(**kwargs):
            raise OSError("disk temporarily unavailable")
        monkeypatch.setattr(store, "tombstone_layer", fail)
    with pytest.raises((ConnectionError, OSError)):
        gateway.clear_learned_preferences(user_id="alice")
    assert gateway.pending_deletion_count(user_id="alice") == 1
    assert gateway.pending_deletion_count(user_id="bob") == 0
    # Restart against the persisted pending operation, then receive new evidence.
    store = MemoryEventStore(path)
    newer = append(store, projection, "alice", "old")
    new_key = append(store, projection, "alice", "new")
    projection.offline = False
    gateway = MemoryGateway(primary=projection, event_store=store, enable_consolidation=False)
    assert gateway.retry_pending_deletions(user_id="alice") == {"completed": 1, "pending": 0}
    assert projection.snapshots == 1
    assert set(projection.rows) == {("alice", "old"), ("alice", "new"), ("bob", "old")}
    assert {r.record_id for r in store.effective_records(user_id="alice")} == {
        keep.record_id, newer.record_id, new_key.record_id,
    }
    assert gateway.pending_deletion_count(user_id="alice") == 0


def test_unsupported_adapter_fails_before_deleting_anything(tmp_path):
    gateway = MemoryGateway(primary=SimpleNamespace(), event_store=MemoryEventStore(tmp_path / "m.sqlite3"),
                            enable_consolidation=False)
    with pytest.raises(RuntimeError, match="bulk_deletion_snapshot_unavailable"):
        gateway.clear_learned_preferences(user_id="alice")


def test_bulk_watermark_has_no_ui_listing_limit(tmp_path):
    store = MemoryEventStore(tmp_path / "m.sqlite3")
    for i in range(205):
        store.append(user_id="alice", layer=MemoryLayer.INFERRED, kind="preference", source="test",
                     evidence_id=str(i), payload={})
    watermark = store.layer_watermark(user_id="alice", layer=MemoryLayer.INFERRED)
    later = store.append(user_id="alice", layer=MemoryLayer.INFERRED, kind="preference", source="test",
                         evidence_id="new", payload={})
    assert store.tombstone_layer(user_id="alice", layer=MemoryLayer.INFERRED, through_seq=watermark) == 205
    assert [r.record_id for r in store.effective_records(user_id="alice")] == [later.record_id]


def test_neo4j_bulk_delete_binds_owner_and_captured_version():
    captured = []
    class Client:
        def execute_query(self, query, params):
            captured.append((query, params))
            return [{"deleted_count": 0}]
    manager = UserMemoryManager(Client())
    targets = [{"memory_key": "warm", "ledger_record_id": "v1", "updated_at": 123}]
    assert manager.delete_inferred_snapshot("alice", targets) is True
    query, params = captured[0]
    assert params == {"user_id": "alice", "targets": targets}
    assert "User {id: $user_id}" in query
    assert "= target.ledger_record_id" in query and "= target.updated_at" in query


def test_old_operation_cannot_finish_a_new_pending_operation(tmp_path):
    store = MemoryEventStore(tmp_path / "m.sqlite3")
    store.stage_bulk_deletion(user_id="alice", snapshot={"operation_id": "first"})
    store.finish_bulk_deletion(user_id="alice", operation_id="first")
    store.stage_bulk_deletion(user_id="alice", snapshot={"operation_id": "second"})
    store.finish_bulk_deletion(user_id="alice", operation_id="first")
    assert store.bulk_deletion_snapshot(user_id="alice")["operation_id"] == "second"
