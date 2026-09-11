"""Bounded, schema-validated execution for ToolPlan v1."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from threading import BoundedSemaphore
from dataclasses import dataclass, field
import inspect
import time
from typing import Any, Awaitable, Callable

from schemas.tool_plan import ToolCall, ToolName, ToolObservation, ToolPlan


ToolExecutor = Callable[[dict[str, Any], dict[str, ToolObservation]], Any | Awaitable[Any]]
Replanner = Callable[[ToolPlan, list[ToolObservation]], ToolPlan | Awaitable[ToolPlan]]

# Slots remain occupied until the underlying work actually ends, even when its
# caller times out. Do not allow retries to accumulate an unbounded work queue.
_SYNC_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="soultuner-tool")
_SYNC_SLOTS = BoundedSemaphore(4)


async def _invoke_executor(executor, arguments, dependencies):
    if inspect.iscoroutinefunction(executor):
        value = executor(arguments, dependencies)
    else:
        if not _SYNC_SLOTS.acquire(blocking=False):
            raise RuntimeError("synchronous tool capacity exhausted")
        try:
            future = _SYNC_POOL.submit(copy_context().run, executor, arguments, dependencies)
        except BaseException:
            _SYNC_SLOTS.release()
            raise
        future.add_done_callback(lambda _: _SYNC_SLOTS.release())
        value = await asyncio.wrap_future(future)
    return await value if inspect.isawaitable(value) else value


def _result_state(value):
    """Interpret existing registry envelopes without discarding partial data."""
    metadata = dict(value.get("metadata") or {}) if isinstance(value, dict) and isinstance(value.get("metadata", {}), dict) else {}
    if isinstance(value, dict):
        error = str(value.get("error") or value.get("error_message") or "")
        failed = bool(error) or value.get("success") is False
        if failed:
            metadata["needs_replan"] = True
            metadata["partial"] = bool(value.get("songs")) or bool(metadata.get("partial"))
            return False, "error", error or "tool reported failure", metadata
        if "songs" in value and not value["songs"]:
            return True, "empty", "", metadata
    empty = value is None or (isinstance(value, (list, dict, str, tuple)) and len(value) == 0)
    return True, "empty" if empty else "success", "", metadata


@dataclass
class ToolRegistry:
    executors: dict[ToolName, ToolExecutor] = field(default_factory=dict)

    def register(self, name: ToolName, executor: ToolExecutor) -> None:
        self.executors[name] = executor

    def get(self, name: ToolName) -> ToolExecutor:
        if name not in self.executors:
            raise KeyError(f"Tool is not registered: {name.value}")
        return self.executors[name]


@dataclass
class OrchestrationResult:
    plan: ToolPlan
    observations: list[ToolObservation]
    replans_used: int

    @property
    def by_call_id(self) -> dict[str, ToolObservation]:
        return {observation.call_id: observation for observation in self.observations}


class BoundedToolOrchestrator:
    def __init__(
        self,
        registry: ToolRegistry,
        *,
        timeout_seconds: float = 20.0,
        max_total_calls: int = 8,
        total_timeout_seconds: float = 60.0,
    ):
        self.registry = registry
        self.timeout_seconds = max(0.1, float(timeout_seconds))
        self.max_total_calls = max(1, int(max_total_calls))
        self.total_timeout_seconds = max(.1, float(total_timeout_seconds))

    async def _execute_call(
        self,
        call: ToolCall,
        observations: dict[str, ToolObservation],
        budget: float | None = None,
    ) -> ToolObservation:
        started = time.perf_counter()
        dependency_view = {dependency: observations[dependency] for dependency in call.depends_on}
        # Reads and fallback inspection can use partial observations. Writes
        # must never consume a failed dependency as if it were validated input.
        if call.name == ToolName.COMMIT_MEMORY_DELTA and any(
            not item.success or item.status in {"error", "timeout", "skipped"}
            for item in dependency_view.values()
        ):
            return ToolObservation(
                call_id=call.id, tool_name=call.name, success=False, status="skipped",
                error="write blocked by failed dependency", metadata={"needs_replan": True},
            )
        try:
            executor = self.registry.get(call.name)
            value = await asyncio.wait_for(
                _invoke_executor(executor, dict(call.arguments), dependency_view),
                timeout=min(self.timeout_seconds, budget) if budget is not None else self.timeout_seconds,
            )
            duration = (time.perf_counter() - started) * 1000
            success, status, error, metadata = _result_state(value)
            return ToolObservation(
                call_id=call.id,
                tool_name=call.name,
                success=success,
                status=status,
                error=error[:500],
                data=value,
                duration_ms=duration,
                metadata=metadata if isinstance(metadata, dict) else {},
            )
        except asyncio.TimeoutError:
            return ToolObservation(
                call_id=call.id,
                tool_name=call.name,
                success=False,
                status="timeout",
                error=f"tool exceeded {self.timeout_seconds:.1f}s",
                duration_ms=(time.perf_counter() - started) * 1000,
            )
        except Exception as exc:
            return ToolObservation(
                call_id=call.id,
                tool_name=call.name,
                success=False,
                status="error",
                error=str(exc)[:500],
                duration_ms=(time.perf_counter() - started) * 1000,
            )

    @staticmethod
    def _needs_replan(observations: list[ToolObservation]) -> bool:
        return any(
            not observation.success
            or observation.status in {"empty", "error", "timeout"}
            or bool(observation.metadata.get("needs_replan"))
            for observation in observations
        )

    async def run(
        self,
        plan: ToolPlan,
        *,
        replanner: Replanner | None = None,
    ) -> OrchestrationResult:
        plan = ToolPlan.model_validate(plan)
        if plan.needs_clarification or not plan.tool_calls:
            return OrchestrationResult(plan=plan, observations=[], replans_used=0)

        observations: dict[str, ToolObservation] = {}
        calls: dict[str, ToolCall] = {call.id: call for call in plan.tool_calls}
        replans_used = 0
        deadline = time.monotonic() + self.total_timeout_seconds

        while True:
            if len(calls) > self.max_total_calls:
                raise ValueError("ToolPlan exceeds bounded total call limit")
            pending = {call_id for call_id in calls if call_id not in observations}
            while pending:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    for call_id in pending:
                        observations[call_id] = ToolObservation(
                            call_id=call_id, tool_name=calls[call_id].name, success=False,
                            status="timeout", error="total tool-plan budget exhausted",
                        )
                    break
                ready = [
                    calls[call_id]
                    for call_id in sorted(pending)
                    if set(calls[call_id].depends_on).issubset(observations)
                ]
                if not ready:
                    raise ValueError("No executable tool calls remain; dependency graph is invalid")
                results = await asyncio.gather(
                    *[self._execute_call(call, observations, remaining) for call in ready]
                )
                observations.update({result.call_id: result for result in results})
                pending -= {call.id for call in ready}

            ordered = [observations[call_id] for call_id in calls]
            if (
                replanner is None
                or time.monotonic() >= deadline
                or replans_used >= plan.max_replans
                or not self._needs_replan(ordered)
            ):
                break
            try:
                revised = await asyncio.wait_for(
                    _invoke_executor(replanner, plan, ordered),
                    timeout=max(.001, deadline - time.monotonic()),
                )
            except asyncio.TimeoutError:
                break
            revised = ToolPlan.model_validate(revised)
            for call in revised.tool_calls:
                if call.id in calls and call.id not in observations:
                    raise ValueError(f"replanner reused pending call id: {call.id}")
                if call.id not in observations:
                    calls[call.id] = call
            plan = revised
            replans_used += 1

        return OrchestrationResult(
            plan=plan,
            observations=[observations[call_id] for call_id in calls if call_id in observations],
            replans_used=replans_used,
        )

