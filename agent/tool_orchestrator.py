"""Bounded, schema-validated execution for ToolPlan v1."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import inspect
import time
from typing import Any, Awaitable, Callable

from schemas.tool_plan import ToolCall, ToolName, ToolObservation, ToolPlan


ToolExecutor = Callable[[dict[str, Any], dict[str, ToolObservation]], Any | Awaitable[Any]]
Replanner = Callable[[ToolPlan, list[ToolObservation]], ToolPlan | Awaitable[ToolPlan]]


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
    ):
        self.registry = registry
        self.timeout_seconds = max(0.1, float(timeout_seconds))
        self.max_total_calls = max(1, int(max_total_calls))

    async def _execute_call(
        self,
        call: ToolCall,
        observations: dict[str, ToolObservation],
    ) -> ToolObservation:
        started = time.perf_counter()
        dependency_view = {dependency: observations[dependency] for dependency in call.depends_on}
        try:
            executor = self.registry.get(call.name)
            value = executor(dict(call.arguments), dependency_view)
            if inspect.isawaitable(value):
                value = await asyncio.wait_for(value, timeout=self.timeout_seconds)
            duration = (time.perf_counter() - started) * 1000
            empty = value is None or value == [] or value == {} or value == ""
            metadata = value.get("metadata", {}) if isinstance(value, dict) else {}
            return ToolObservation(
                call_id=call.id,
                tool_name=call.name,
                success=True,
                status="empty" if empty else "success",
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

        while True:
            if len(calls) > self.max_total_calls:
                raise ValueError("ToolPlan exceeds bounded total call limit")
            pending = {call_id for call_id in calls if call_id not in observations}
            while pending:
                ready = [
                    calls[call_id]
                    for call_id in sorted(pending)
                    if set(calls[call_id].depends_on).issubset(observations)
                ]
                if not ready:
                    raise ValueError("No executable tool calls remain; dependency graph is invalid")
                results = await asyncio.gather(
                    *[self._execute_call(call, observations) for call in ready]
                )
                observations.update({result.call_id: result for result in results})
                pending -= {call.id for call in ready}

            ordered = [observations[call_id] for call_id in calls]
            if (
                replanner is None
                or replans_used >= plan.max_replans
                or not self._needs_replan(ordered)
            ):
                break
            revised = replanner(plan, ordered)
            if inspect.isawaitable(revised):
                revised = await revised
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

