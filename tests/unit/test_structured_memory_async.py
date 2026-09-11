import asyncio
import threading
from types import SimpleNamespace

import pytest

from services.memory_gateway import MemoryGateway


def test_default_structured_memory_never_constructs_sidecars(monkeypatch):
    monkeypatch.delenv("MEMORY_MODE", raising=False)
    monkeypatch.setenv("MEMORY_EPISODIC_BACKENDS", "graphzep,mem0")
    def forbidden(*args, **kwargs):
        raise AssertionError("default structured mode must not construct optional sidecars")
    monkeypatch.setattr("services.memory_gateway._configured_episodic_adapters", forbidden)
    gateway = MemoryGateway(primary=SimpleNamespace(), enable_event_ledger=False)
    assert gateway.mode == "structured"
    assert gateway.episodic_adapters == []


def test_default_memory_io_and_scoring_run_off_event_loop():
    loop_thread = threading.get_ident()
    calls = []
    def check(name, result):
        def run(*args, **kwargs):
            assert threading.get_ident() != loop_thread
            calls.append(name)
            return result
        return run
    gateway = MemoryGateway(
        primary=SimpleNamespace(get_user_profile=check("profile", {})),
        event_store=SimpleNamespace(effective_records=check("ledger", [])),
        relevance_retriever=SimpleNamespace(retrieve=check("score", []), backend_name="test",
                                           describe=lambda: {}),
        enable_consolidation=False,
        memory_mode="structured",
    )
    result = asyncio.run(gateway.retrieve_context(query="安静的音乐", user_id="alice"))
    assert calls == ["profile", "ledger", "score"]
    assert result["retrieved_records"] == []
    assert result["episodic_backends"] == {}


@pytest.mark.parametrize("failed", ["profile", "ledger", "both"])
def test_memory_outage_is_explicit_and_other_backend_survives(failed):
    def unavailable(*args, **kwargs):
        raise RuntimeError("private-database-host-and-credential")
    gateway = MemoryGateway(
        primary=SimpleNamespace(get_user_profile=unavailable if failed in {"profile", "both"}
                                else lambda *a, **k: {"avoid_genres": ["metal"]}),
        event_store=SimpleNamespace(effective_records=unavailable if failed in {"ledger", "both"}
                                    else lambda **k: []),
        relevance_retriever=SimpleNamespace(retrieve=lambda **k: [], backend_name="test", describe=lambda: {}),
        enable_consolidation=False, memory_mode="structured",
    )
    result = asyncio.run(gateway.retrieve_context(query="雨天", user_id="alice"))
    trace = result["memory_trace"]
    assert trace["status"] == "degraded"
    assert bool(trace["profile_error"]) == (failed in {"profile", "both"})
    assert bool(trace["ledger_error"]) == (failed in {"ledger", "both"})
    if failed == "ledger":
        assert result["profile"]["avoid_genres"] == ["metal"]
    else:
        with pytest.raises(RuntimeError, match="^memory_profile_unavailable$"):
            gateway.get_user_profile("alice")
    assert "private-database" not in str(result)
