import asyncio
import copy

import pytest

from services.authorized_memory_writer import AuthorizedMemoryWriter
from services.memory_event_store import MemoryEventStore
from services.memory_models import MemoryLayer


def setup_writer(tmp_path, *, source="user_statement", evidence_owner="alice", commit=None):
    store = MemoryEventStore(tmp_path / "memory.sqlite3")
    evidence = store.append(user_id=evidence_owner, layer=MemoryLayer.RAW_EVENT,
                            kind="conversation_statement", source=source, evidence_id="", payload={"user_text": "轻柔的音乐"})
    delta = {"memory_type": "explicit_preference", "values": {"add_moods": ["Warm"]},
             "evidence_id": evidence.record_id, "confidence": 1.0}
    calls = []
    async def apply(user, value):
        calls.append((user, value))
        await asyncio.sleep(0)
        return {"success": True}
    writer = AuthorizedMemoryWriter(user_id="alice", approved_delta=delta,
                                    evidence_ids=[evidence.record_id], event_store=store, commit=commit or apply)
    return writer, store, evidence, delta, calls


@pytest.mark.parametrize("field,value", [("memory_type", "episodic"), ("confidence", .9),
                                        ("values", {"avoid_genres": ["rock"]}), ("evidence_id", "invented")])
def test_model_cannot_change_approved_delta(tmp_path, field, value):
    writer, _, _, delta, calls = setup_writer(tmp_path)
    delta[field] = value
    assert asyncio.run(writer("alice", delta))["success"] is False
    assert calls == []


@pytest.mark.parametrize("source,owner", [("assistant", "alice"), ("system", "alice"), ("user_statement", "bob")])
def test_only_owned_user_evidence_is_eligible(tmp_path, source, owner):
    writer, _, _, delta, calls = setup_writer(tmp_path, source=source, evidence_owner=owner)
    assert asyncio.run(writer("alice", delta))["success"] is False
    assert calls == []


def test_deleted_evidence_and_forged_owner_are_denied(tmp_path):
    writer, store, evidence, delta, calls = setup_writer(tmp_path)
    assert asyncio.run(writer("bob", delta))["success"] is False
    store.tombstone(user_id="alice", target_record_id=evidence.record_id)
    assert asyncio.run(writer("alice", delta))["success"] is False
    assert calls == []


def test_concurrent_identical_calls_write_once_and_grant_is_immutable(tmp_path):
    writer, _, _, delta, calls = setup_writer(tmp_path)
    original = copy.deepcopy(delta)
    delta["values"]["add_moods"].append("Changed")
    async def run():
        return await asyncio.gather(writer("alice", original), writer("alice", original))
    assert all(result["success"] for result in asyncio.run(run()))
    assert len(calls) == 1


def test_uncertain_outcome_is_not_blindly_retried(tmp_path):
    calls = []
    async def fail(*args):
        calls.append(1)
        raise RuntimeError("private database endpoint")
    writer, _, _, delta, _ = setup_writer(tmp_path, commit=fail)
    async def run():
        for _ in range(2):
            result = await writer("alice", delta)
            assert result["success"] is False
            assert "private database" not in str(result)
    asyncio.run(run())
    assert len(calls) == 1


def test_authorizer_integrates_with_registry(tmp_path):
    from agent.music_tool_registry import build_music_tool_registry
    from schemas.tool_plan import ToolName
    writer, _, _, delta, calls = setup_writer(tmp_path)
    registry = build_music_tool_registry(user_id="alice", query="轻柔一点", authorized_memory_writer=writer)
    assert asyncio.run(registry.get(ToolName.COMMIT_MEMORY_DELTA)(delta, {}))["success"] is True
    assert len(calls) == 1


def test_expired_capability_and_read_only_mode_never_commit(tmp_path, monkeypatch):
    writer, _, _, delta, calls = setup_writer(tmp_path)
    monkeypatch.setenv("EVAL_DISABLE_SIDE_EFFECTS", "1")
    assert asyncio.run(writer("alice", delta))["success"] is False
    monkeypatch.delenv("EVAL_DISABLE_SIDE_EFFECTS")
    writer._expires_at = 0
    assert asyncio.run(writer("alice", delta))["success"] is False
    assert calls == []


def test_cancellation_after_start_requires_reconciliation(tmp_path):
    calls = []
    async def run():
        entered = asyncio.Event()
        async def commit(*args):
            calls.append(1)
            entered.set()
            await asyncio.Event().wait()
        writer, _, _, delta, _ = setup_writer(tmp_path, commit=commit)
        task = asyncio.create_task(writer("alice", delta))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert (await writer("alice", delta))["success"] is False
    asyncio.run(run())
    assert calls == [1]
