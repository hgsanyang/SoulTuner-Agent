"""Failure harness for the bounded ToolPlan orchestrator.

Promotion gate prerequisite: the orchestrator must behave predictably under
tool failure, timeout, empty observation, retry/replan, clarification, and
resource-bound violations before ToolPlan execution can become the default.
Each scenario is exercised deterministically with injected executors.
"""

import asyncio
import threading
from contextvars import ContextVar

import pytest

from agent.tool_orchestrator import BoundedToolOrchestrator, ToolRegistry
from schemas.tool_plan import ToolCall, ToolName, ToolObservation, ToolPlan


@pytest.mark.parametrize("payload,status,success", [
    ({"songs": [], "source": "web"}, "empty", True),
    ({"songs": [], "error": "offline"}, "error", False),
    ({"songs": [{"title": "kept"}], "error": "partial failure"}, "error", False),
    ({"success": False, "error": ""}, "error", False),
    ({"songs": [{"title": "ok"}], "source": "graph"}, "success", True),
])
def test_registry_envelope_status(payload, status, success):
    registry = ToolRegistry()
    registry.register(ToolName.RETRIEVE_MEMORY, lambda *_: payload)
    obs = _run(BoundedToolOrchestrator(registry), _plan(_memory_call("m"))).observations[0]
    assert (obs.status, obs.success, obs.data) == (status, success, payload)
    assert BoundedToolOrchestrator._needs_replan([obs]) == (status != "success")


def test_sync_timeout_does_not_block_loop_and_propagates_context():
    release = threading.Event()
    finished = threading.Event()
    marker = ContextVar("tool-test", default="missing")
    seen = []

    def slow(*_):
        seen.append(marker.get())
        try:
            release.wait(2)
        finally:
            finished.set()

    async def run():
        token = marker.set("request")
        registry = ToolRegistry()
        registry.register(ToolName.RETRIEVE_MEMORY, slow)
        try:
            result = await BoundedToolOrchestrator(registry, timeout_seconds=.1).run(_plan(_memory_call("m")))
            assert result.observations[0].status == "timeout"
            assert not finished.is_set()
            assert seen == ["request"]
        finally:
            release.set()
            marker.reset(token)

    asyncio.run(run())
    assert finished.wait(2)


def test_sync_factory_returning_coroutine_is_awaited():
    async def answer():
        return ["ok"]
    registry = ToolRegistry()
    registry.register(ToolName.RETRIEVE_MEMORY, lambda *_: answer())
    assert _run(BoundedToolOrchestrator(registry), _plan(_memory_call("m"))).observations[0].data == ["ok"]


def test_failed_dependency_blocks_memory_write_but_not_read_fallback():
    registry = ToolRegistry()
    registry.register(ToolName.RETRIEVE_MEMORY, lambda *_: {"success": False, "error": "offline"})
    def forbidden(*args):
        raise AssertionError("write must not be invoked")
    registry.register(ToolName.COMMIT_MEMORY_DELTA, forbidden)
    plan = _plan(_memory_call("m"), ToolCall(
        id="write", name=ToolName.COMMIT_MEMORY_DELTA, depends_on=["m"],
        arguments={"memory_type": "inferred_preference", "evidence_id": "e"},
    ))
    result = _run(BoundedToolOrchestrator(registry), plan)
    assert result.by_call_id["write"].status == "skipped"
    assert result.by_call_id["write"].error == "write blocked by failed dependency"


def test_total_budget_includes_replanner_and_preserves_results():
    registry = ToolRegistry()
    registry.register(ToolName.RETRIEVE_MEMORY, lambda *_: [])
    async def replan(*_):
        await asyncio.sleep(5)
    result = _run(BoundedToolOrchestrator(registry, total_timeout_seconds=.1),
                  _plan(_memory_call("m")), replanner=replan)
    assert result.observations[0].status == "empty"
    assert result.replans_used == 0


def test_total_budget_marks_unstarted_dependencies_timeout():
    registry = ToolRegistry()
    async def slow(*_):
        await asyncio.sleep(5)
    registry.register(ToolName.RETRIEVE_MEMORY, slow)
    result = _run(BoundedToolOrchestrator(registry, total_timeout_seconds=.1),
                  _plan(_memory_call("a"), _memory_call("b", ["a"])))
    assert [item.status for item in result.observations] == ["timeout", "timeout"]


def test_timeout_keeps_worker_capacity_until_real_completion(monkeypatch):
    import agent.tool_orchestrator as module
    slots = threading.BoundedSemaphore(1)
    monkeypatch.setattr(module, "_SYNC_SLOTS", slots)
    release, finished = threading.Event(), threading.Event()

    def work(*_):
        try:
            release.wait(2)
        finally:
            finished.set()

    async def run():
        registry = ToolRegistry()
        registry.register(ToolName.RETRIEVE_MEMORY, work)
        orchestrator = BoundedToolOrchestrator(registry, timeout_seconds=.1)
        try:
            first = await orchestrator.run(_plan(_memory_call("one")))
            assert first.observations[0].status == "timeout"
            second = await orchestrator.run(_plan(_memory_call("two")))
            assert second.observations[0].status == "error"
            assert "capacity exhausted" in second.observations[0].error
        finally:
            release.set()
        await asyncio.to_thread(finished.wait, 2)

    asyncio.run(run())


def _plan(*calls: ToolCall, max_replans: int = 1, **kwargs) -> ToolPlan:
    return ToolPlan(
        request_mode="recommendation",
        tool_calls=list(calls),
        max_replans=max_replans,
        **kwargs,
    )


def _memory_call(call_id: str, depends_on: list[str] | None = None) -> ToolCall:
    return ToolCall(
        id=call_id,
        name=ToolName.RETRIEVE_MEMORY,
        arguments={"query": "q", "limit": 5},
        depends_on=depends_on or [],
    )


def _run(orchestrator, plan, **kwargs):
    return asyncio.run(orchestrator.run(plan, **kwargs))


# ---------- single-tool failure modes → observation, never a raise ----------

def test_tool_error_is_captured_as_failed_observation():
    registry = ToolRegistry()

    def boom(args, deps):
        raise RuntimeError("downstream exploded")

    registry.register(ToolName.RETRIEVE_MEMORY, boom)
    result = _run(BoundedToolOrchestrator(registry), _plan(_memory_call("m1"), max_replans=0))

    obs = result.by_call_id["m1"]
    assert obs.success is False
    assert obs.status == "error"
    assert "downstream exploded" in obs.error
    assert obs.duration_ms >= 0.0


def test_tool_timeout_is_captured_as_timeout_observation():
    registry = ToolRegistry()

    async def hang(args, deps):
        await asyncio.sleep(5)
        return {"data": 1}

    registry.register(ToolName.RETRIEVE_MEMORY, hang)
    orchestrator = BoundedToolOrchestrator(registry, timeout_seconds=0.05)
    result = _run(orchestrator, _plan(_memory_call("m1"), max_replans=0))

    obs = result.by_call_id["m1"]
    assert obs.success is False
    assert obs.status == "timeout"
    assert "exceeded" in obs.error


def test_empty_result_is_success_but_flagged_empty():
    registry = ToolRegistry()
    registry.register(ToolName.RETRIEVE_MEMORY, lambda args, deps: [])
    result = _run(BoundedToolOrchestrator(registry), _plan(_memory_call("m1"), max_replans=0))

    obs = result.by_call_id["m1"]
    assert obs.success is True
    assert obs.status == "empty"


# ---------- replan / retry recovery ----------

def test_failed_call_triggers_replan_that_recovers():
    registry = ToolRegistry()
    calls_seen: list[str] = []

    def flaky(args, deps):
        calls_seen.append("m1")
        raise RuntimeError("first attempt fails")

    def healthy(args, deps):
        calls_seen.append("m2")
        return {"songs": [1, 2, 3]}

    registry.register(ToolName.RETRIEVE_MEMORY, flaky)
    registry.register(ToolName.SEARCH_GRAPH, healthy)

    def replanner(plan, observations):
        # Recover by scheduling a NEW call (fresh id) that reads healthily.
        return _plan(
            ToolCall(id="m2", name=ToolName.SEARCH_GRAPH, arguments={"limit": 10}),
            max_replans=1,
        )

    result = _run(BoundedToolOrchestrator(registry), _plan(_memory_call("m1")), replanner=replanner)

    assert result.replans_used == 1
    assert result.by_call_id["m1"].status == "error"
    assert result.by_call_id["m2"].success is True
    assert calls_seen == ["m1", "m2"]


def test_replan_budget_is_respected_even_if_still_failing():
    registry = ToolRegistry()
    registry.register(ToolName.RETRIEVE_MEMORY, lambda a, d: (_ for _ in ()).throw(RuntimeError("always")))
    registry.register(ToolName.SEARCH_GRAPH, lambda a, d: (_ for _ in ()).throw(RuntimeError("also")))

    replan_count = {"n": 0}

    def replanner(plan, observations):
        replan_count["n"] += 1
        return _plan(
            ToolCall(id=f"g{replan_count['n']}", name=ToolName.SEARCH_GRAPH, arguments={}),
            max_replans=1,
        )

    result = _run(BoundedToolOrchestrator(registry), _plan(_memory_call("m1")), replanner=replanner)
    # max_replans=1 → exactly one replan attempt, no infinite loop
    assert result.replans_used == 1
    assert replan_count["n"] == 1


def test_metadata_needs_replan_flag_triggers_replan():
    registry = ToolRegistry()
    registry.register(
        ToolName.RETRIEVE_MEMORY,
        lambda a, d: {"data": "ok", "metadata": {"needs_replan": True}},
    )
    registry.register(ToolName.SEARCH_GRAPH, lambda a, d: {"songs": [1]})

    replanned = {"n": 0}

    def replanner(plan, observations):
        replanned["n"] += 1
        return _plan(ToolCall(id="g1", name=ToolName.SEARCH_GRAPH, arguments={}), max_replans=1)

    result = _run(BoundedToolOrchestrator(registry), _plan(_memory_call("m1")), replanner=replanner)
    assert replanned["n"] == 1
    assert result.replans_used == 1


def test_no_replanner_means_failure_is_returned_as_is():
    registry = ToolRegistry()
    registry.register(ToolName.RETRIEVE_MEMORY, lambda a, d: (_ for _ in ()).throw(RuntimeError("x")))
    result = _run(BoundedToolOrchestrator(registry), _plan(_memory_call("m1")))
    assert result.replans_used == 0
    assert result.by_call_id["m1"].success is False


# ---------- clarification / empty plan short-circuit ----------

def test_clarification_plan_executes_no_tools():
    registry = ToolRegistry()
    plan = ToolPlan(
        request_mode="conversation",
        needs_clarification=True,
        clarification_question="你想听什么风格？",
    )
    result = _run(BoundedToolOrchestrator(registry), plan)
    assert result.observations == []
    assert result.replans_used == 0


def test_empty_tool_calls_execute_nothing():
    registry = ToolRegistry()
    result = _run(BoundedToolOrchestrator(registry), _plan(max_replans=0))
    assert result.observations == []


# ---------- resource bounds & invalid graphs raise (fail-loud, not silent) ----------

def test_exceeding_max_total_calls_raises():
    registry = ToolRegistry()
    registry.register(ToolName.RETRIEVE_MEMORY, lambda a, d: {"ok": 1})
    registry.register(ToolName.SEARCH_GRAPH, lambda a, d: {"ok": 1})
    orchestrator = BoundedToolOrchestrator(registry, max_total_calls=1)
    plan = _plan(
        _memory_call("m1"),
        ToolCall(id="g1", name=ToolName.SEARCH_GRAPH, arguments={}),
        max_replans=0,
    )
    with pytest.raises(ValueError, match="bounded total call limit"):
        _run(orchestrator, plan)


def test_unregistered_tool_surfaces_as_error_observation():
    registry = ToolRegistry()  # nothing registered
    result = _run(BoundedToolOrchestrator(registry), _plan(_memory_call("m1"), max_replans=0))
    obs = result.by_call_id["m1"]
    assert obs.success is False
    assert obs.status == "error"
    assert "not registered" in obs.error


def test_replanner_returning_invalid_plan_raises():
    registry = ToolRegistry()
    registry.register(ToolName.RETRIEVE_MEMORY, lambda a, d: (_ for _ in ()).throw(RuntimeError("x")))

    # A replanner that returns a schema-invalid plan (dependency cycle) must
    # surface loudly, not silently produce a broken execution.
    def bad_replanner(plan, observations):
        return {
            "request_mode": "recommendation",
            "max_replans": 1,
            "tool_calls": [
                {"id": "a", "name": "search_graph", "arguments": {}, "depends_on": ["b"]},
                {"id": "b", "name": "search_graph", "arguments": {}, "depends_on": ["a"]},
            ],
        }

    with pytest.raises(ValueError, match="cycle"):
        _run(BoundedToolOrchestrator(registry), _plan(_memory_call("m1")), replanner=bad_replanner)


# ---------- dependency ordering & view ----------

def test_dependent_call_sees_predecessor_observation():
    registry = ToolRegistry()
    seen_deps: dict[str, list[str]] = {}

    def first(args, deps):
        return {"value": "from_first"}

    def second(args, deps):
        seen_deps["m2"] = sorted(deps.keys())
        assert deps["m1"].data == {"value": "from_first"}
        return {"value": "from_second"}

    registry.register(ToolName.RETRIEVE_MEMORY, first)
    registry.register(ToolName.SEARCH_GRAPH, second)
    plan = _plan(
        _memory_call("m1"),
        ToolCall(id="m2", name=ToolName.SEARCH_GRAPH, arguments={}, depends_on=["m1"]),
        max_replans=0,
    )
    result = _run(BoundedToolOrchestrator(registry), plan)
    assert seen_deps["m2"] == ["m1"]
    assert result.by_call_id["m2"].success is True


def test_all_observations_returned_in_plan_call_order():
    registry = ToolRegistry()
    registry.register(ToolName.RETRIEVE_MEMORY, lambda a, d: {"ok": 1})
    registry.register(ToolName.SEARCH_GRAPH, lambda a, d: {"ok": 2})
    plan = _plan(
        _memory_call("m1"),
        ToolCall(id="g1", name=ToolName.SEARCH_GRAPH, arguments={}),
        max_replans=0,
    )
    result = _run(BoundedToolOrchestrator(registry), plan)
    assert [o.call_id for o in result.observations] == ["m1", "g1"]
    assert all(isinstance(o, ToolObservation) for o in result.observations)


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
