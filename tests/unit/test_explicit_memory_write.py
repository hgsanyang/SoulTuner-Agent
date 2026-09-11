from types import SimpleNamespace

import pytest

from retrieval.user_memory import UserMemoryManager, normalize_explicit_preferences
from services.memory_event_store import MemoryEventStore
from services.memory_gateway import MemoryGateway, Neo4jPreferenceAdapter
from services.memory_models import MemoryLayer


@pytest.mark.parametrize("payload", [ {}, {"invented": ["warm"]}, {"add_moods": []},
    {"add_moods": [12]}, {"add_moods": [""]}, {"add_moods": "warm"},
    {"language_preference": ["Chinese"]}, {"add_moods": ["x" * 161]},
    {"add_moods": ["Warm"], "avoid_moods": ["warm"]},
])
def test_invalid_preferences_rejected_before_any_write(tmp_path, payload):
    def forbidden(*args):
        raise AssertionError("no write allowed")
    store = MemoryEventStore(tmp_path / "ledger.sqlite3")
    gateway = MemoryGateway(primary=SimpleNamespace(remember_preference=forbidden), event_store=store)
    with pytest.raises(ValueError):
        gateway.remember_preference(user_id="alice", preferences=payload)
    assert not store.path.exists()


def test_normalization_is_case_insensitive_and_non_mutating():
    original = {"add_moods": [" Warm ", "warm"]}
    assert normalize_explicit_preferences(original) == {"add_moods": ["Warm"]}
    assert original["add_moods"] == [" Warm ", "warm"]


@pytest.mark.parametrize("response", [None, [], [{"user_id": "someone-else"}]])
def test_unconfirmed_database_write_does_not_create_success_ledger(tmp_path, response):
    class Client:
        def execute_query(self, *args):
            return response
    manager = UserMemoryManager(Client())
    store = MemoryEventStore(tmp_path / "ledger.sqlite3")
    gateway = MemoryGateway(primary=Neo4jPreferenceAdapter(manager), event_store=store)
    with pytest.raises(RuntimeError, match="memory_write_unavailable"):
        gateway.remember_preference(user_id="alice", preferences={"add_moods": ["Warm"]})
    assert store.pending_preference_write(user_id="alice") is not None
    assert store.effective_records(user_id="alice") == []


def test_database_exception_propagates_without_sensitive_public_message():
    class Client:
        def execute_query(self, *args):
            raise ConnectionError("private host")
    with pytest.raises(RuntimeError, match="^memory_write_unavailable$"):
        UserMemoryManager(Client()).update_semantic_preferences("alice", {"add_moods": ["Warm"]})


@pytest.mark.parametrize("result", [None, False])
def test_adapter_must_explicitly_confirm_write(tmp_path, result):
    gateway = MemoryGateway(primary=SimpleNamespace(remember_preference=lambda *a: result),
                            event_store=MemoryEventStore(tmp_path / "ledger.sqlite3"))
    with pytest.raises(RuntimeError, match="memory_write_not_confirmed"):
        gateway.remember_preference(user_id="alice", preferences={"add_moods": ["Warm"]})


def test_confirmed_correction_retires_opposite_ledger_preference(tmp_path):
    store = MemoryEventStore(tmp_path / "ledger.sqlite3")
    gateway = MemoryGateway(primary=SimpleNamespace(remember_preference=lambda *a: True), event_store=store)
    gateway.remember_preference(user_id="alice", preferences={"add_moods": ["Warm"]})
    gateway.remember_preference(user_id="bob", preferences={"add_moods": ["Warm"]})
    gateway.remember_preference(user_id="alice", preferences={"avoid_moods": ["warm"]})
    active = store.effective_records(user_id="alice")
    assert len(active) == 1 and active[0].payload["field"] == "avoid_moods"
    assert store.effective_records(user_id="bob")[0].payload["field"] == "add_moods"
    gateway.remember_preference(user_id="alice", preferences={"add_moods": ["Warm"]})
    active = store.effective_records(user_id="alice")
    assert len(active) == 1 and active[0].layer == MemoryLayer.EXPLICIT
    assert active[0].payload["field"] == "add_moods"
