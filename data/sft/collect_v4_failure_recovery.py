"""Build reproducible V4 failure-recovery trajectories from executed ToolPlans.

The harness injects faults into the production ``BoundedToolOrchestrator``. It
does not ask a teacher model to imagine a timeout or empty result. Each output
row contains the initial plan, the observed failure, and the bounded recovery
plan that was actually executed.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.tool_orchestrator import BoundedToolOrchestrator, ToolRegistry  # noqa: E402
from schemas.tool_plan import ToolCall, ToolName, ToolPlan  # noqa: E402


SYSTEM_PROMPT = (
    "You are SoulTuner's bounded ToolPlan recovery planner. "
    "Given the user request, the previously selected ToolPlan, and an actual "
    "tool observation, return one safe ToolPlan JSON. Respect the one-replan "
    "budget, do not invent successful results, and prefer a viable independent "
    "lane or an explicit clarification over repeating the same failed action."
)


@dataclass(frozen=True)
class Scenario:
    name: str
    request_kind: str
    query_template: str
    initial_tool: ToolName
    initial_arguments: dict[str, Any]
    failure_status: str
    recovery_tool: ToolName | None
    recovery_arguments: dict[str, Any]


SCENARIOS = (
    Scenario(
        "unknown_artist_graph_empty",
        "recommendation",
        "推荐一些 {entity} 的代表作，找不到本地歌也别硬凑。",
        ToolName.SEARCH_GRAPH,
        {"artist_entities": ["{entity}"], "limit": 20},
        "empty",
        ToolName.SEARCH_EXTERNAL_MUSIC,
        {"requirements": "{entity} representative songs", "entities": ["{entity}"], "limit": 8},
    ),
    Scenario(
        "audio_timeout_graph_fallback",
        "recommendation",
        "来点{vibe}的歌，先保证有结果。",
        ToolName.SEARCH_AUDIO,
        {"acoustic_queries": ["{acoustic}"], "limit": 30},
        "timeout",
        ToolName.SEARCH_GRAPH,
        {"moods": ["{vibe}"], "limit": 30},
    ),
    Scenario(
        "web_error_local_fallback",
        "recommendation",
        "想听{vibe}的新鲜歌曲，联网不通就从本地找。",
        ToolName.SEARCH_EXTERNAL_MUSIC,
        {"requirements": "{vibe} fresh music", "limit": 8},
        "error",
        ToolName.SEARCH_AUDIO,
        {"acoustic_queries": ["{acoustic}"], "limit": 30},
    ),
    Scenario(
        "library_timeout_retry",
        "library",
        "查看我收藏里和 {entity} 有关的歌。",
        ToolName.READ_LIBRARY,
        {"collection": "saved", "query": "{entity}", "limit": 30},
        "timeout_once",
        ToolName.READ_LIBRARY,
        {"collection": "saved", "query": "{entity}", "limit": 20},
    ),
    Scenario(
        "acquisition_empty_clarify",
        "acquisition",
        "把 {entity} 那首同名歌放进待入库。",
        ToolName.SEARCH_EXTERNAL_MUSIC,
        {"requirements": "{entity} same-title song", "entities": ["{entity}"], "limit": 5},
        "empty",
        None,
        {},
    ),
    Scenario(
        "playable_empty_find_alternative",
        "acquisition",
        "找 {entity} 的可播放正式版，失效版本不要。",
        ToolName.RESOLVE_PLAYABLE_TRACKS,
        {"candidate_source_ids": ["candidate-{index}"], "limit": 5},
        "empty",
        ToolName.SEARCH_EXTERNAL_MUSIC,
        {"requirements": "{entity} official playable version", "entities": ["{entity}"], "limit": 8},
    ),
    Scenario(
        "memory_timeout_continue",
        "recommendation",
        "按我平时的口味来点{vibe}的，记忆不可用也继续推荐。",
        ToolName.RETRIEVE_MEMORY,
        {"query": "{vibe}", "scope": "all", "limit": 8},
        "timeout",
        ToolName.SEARCH_AUDIO,
        {"acoustic_queries": ["{acoustic}"], "limit": 30},
    ),
    Scenario(
        "catalog_gap_error_local",
        "recommendation",
        "想听 {entity}，库存诊断失败时先给本地最接近的。",
        ToolName.INSPECT_CATALOG_GAP,
        {"requirements": {"artist": "{entity}"}},
        "error",
        ToolName.SEARCH_GRAPH,
        {"artist_entities": ["{entity}"], "limit": 30},
    ),
    Scenario(
        "graph_error_dense_fallback",
        "recommendation",
        "找些接近 {entity} 但听感更{vibe}的歌。",
        ToolName.SEARCH_GRAPH,
        {"artist_entities": ["{entity}"], "limit": 30},
        "error",
        ToolName.SEARCH_AUDIO,
        {"acoustic_queries": ["{acoustic}"], "limit": 30},
    ),
    Scenario(
        "external_refusal_local",
        "information",
        "查一下 {entity} 的音乐背景；联网拒绝时只回答本地已有资料。",
        ToolName.SEARCH_EXTERNAL_MUSIC,
        {"requirements": "{entity} music background", "entities": ["{entity}"], "limit": 5},
        "needs_replan",
        ToolName.SEARCH_GRAPH,
        {"artist_entities": ["{entity}"], "limit": 10},
    ),
)

ENTITIES = (
    "The Blue Nile",
    "椅子乐团",
    "Corn Wave",
    "Lamp",
    "Vaundy",
    "The Cure",
    "朴树",
    "Sonic Youth",
    "张震岳",
    "Massive Attack",
    "Slowdive",
    "坂本龙一",
)

VIBES = (
    ("安静柔软", "quiet intimate music with soft dynamics and a warm sparse arrangement"),
    ("克制忧郁", "restrained melancholic music with low energy and spacious instrumentation"),
    ("明亮有节奏", "bright rhythmic music with moderate energy and an uplifting pulse"),
    ("深夜氛围", "late-night atmospheric music with dreamy textures and gentle vocals"),
    ("缓慢推进", "slow-building music with sparse percussion and gradual dynamics"),
)


def _format(value: Any, values: dict[str, Any]) -> Any:
    if isinstance(value, str):
        return value.format(**values)
    if isinstance(value, list):
        return [_format(item, values) for item in value]
    if isinstance(value, dict):
        return {key: _format(item, values) for key, item in value.items()}
    return value


def _plan(
    request_kind: str,
    call_id: str,
    tool: ToolName,
    arguments: dict[str, Any],
    *,
    origin: str = "planner",
) -> ToolPlan:
    return ToolPlan(
        origin=origin,
        request_mode=request_kind,
        tool_calls=[
            ToolCall(
                id=call_id,
                name=tool,
                arguments=arguments,
                reason="bounded harness trajectory",
            )
        ],
        max_replans=1,
        decision_summary="execute the selected bounded lane",
    )


def _executor_for(status: str) -> Callable[[dict[str, Any], dict], Any]:
    calls = {"count": 0}

    async def executor(arguments: dict[str, Any], dependencies: dict) -> Any:
        calls["count"] += 1
        if status in {"timeout", "timeout_once"} and (
            status == "timeout" or calls["count"] == 1
        ):
            # BoundedToolOrchestrator intentionally clamps its timeout floor to
            # 100ms. Sleep beyond that floor so this remains a real timeout.
            await asyncio.sleep(0.2)
            return {"unexpected": True}
        if status == "error":
            raise RuntimeError("controlled downstream failure")
        if status == "empty":
            return []
        if status == "needs_replan":
            return {
                "metadata": {
                    "needs_replan": True,
                    "reason": "controlled provider refusal",
                }
            }
        return {"ok": True, "items": [{"id": "recovered"}]}

    return executor


def _recovery_plan(
    scenario: Scenario,
    values: dict[str, Any],
) -> ToolPlan:
    if scenario.recovery_tool is None:
        return ToolPlan(
            origin="replanner",
            request_mode=scenario.request_kind,
            needs_clarification=True,
            clarification_question="没有找到唯一对应的版本，请告诉我歌曲名或专辑名。",
            max_replans=0,
            decision_summary="the failed discovery left the referent underdetermined",
        )
    return _plan(
        scenario.request_kind,
        "recovery",
        scenario.recovery_tool,
        _format(scenario.recovery_arguments, values),
        origin="replanner",
    )


def _trace_id(scenario: Scenario, index: int, query: str) -> str:
    digest = hashlib.sha256(
        f"{scenario.name}\0{index}\0{query}".encode("utf-8")
    ).hexdigest()[:16]
    return f"harness-{scenario.name}-{digest}"


async def execute_scenario(scenario: Scenario, index: int) -> dict[str, Any]:
    entity = ENTITIES[index % len(ENTITIES)]
    vibe, acoustic = VIBES[index % len(VIBES)]
    values = {
        "entity": entity,
        "vibe": vibe,
        "acoustic": acoustic,
        "index": index,
    }
    query = scenario.query_template.format(**values)
    initial = _plan(
        scenario.request_kind,
        "initial",
        scenario.initial_tool,
        _format(scenario.initial_arguments, values),
    )
    recovery = _recovery_plan(scenario, values)

    registry = ToolRegistry()
    registry.register(scenario.initial_tool, _executor_for(scenario.failure_status))
    if (
        scenario.recovery_tool is not None
        and scenario.recovery_tool != scenario.initial_tool
    ):
        registry.register(scenario.recovery_tool, lambda args, deps: {"ok": True})
    orchestrator = BoundedToolOrchestrator(
        registry,
        timeout_seconds=0.005,
        max_total_calls=2,
    )

    result = await orchestrator.run(
        initial,
        replanner=lambda plan, observations: recovery,
    )
    if result.replans_used != 1:
        raise RuntimeError(f"scenario {scenario.name} did not execute its recovery")
    observations = [
        observation.model_dump(mode="json", exclude_none=True)
        for observation in result.observations
        if observation.call_id == "initial"
    ]
    if len(observations) != 1:
        raise RuntimeError(f"scenario {scenario.name} has no initial observation")

    trace_id = _trace_id(scenario, index, query)
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {"current_query": query, "request_kind": scenario.request_kind},
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            },
            {
                "role": "assistant",
                "content": initial.model_dump_json(exclude_none=True),
            },
            {
                "role": "tool",
                "content": json.dumps(
                    observations,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            },
            {
                "role": "assistant",
                "content": recovery.model_dump_json(exclude_none=True),
            },
        ],
        "meta": {
            "seed_source": "template_expansion",
            "episode_id": f"failure_{scenario.name}_{index:04d}",
            "turn_id": 1,
            "trace_id": trace_id,
            "request_kind": scenario.request_kind,
            "trajectory_kind": "failure_recovery",
            "observation_origin": "harness_execution",
            "execution_environment": "controlled_harness",
            "teacher": {
                "model": "bounded-toolplan-contract-oracle",
                "version": "1.1",
                "vendor": "SoulTuner",
            },
            "reviewer": {
                "model": "schema-and-behavior-tests",
                "version": "1",
                "vendor": "SoulTuner",
            },
            "reviewer_verdict": "accept",
        },
        "lineage": {
            "collector": "collect_v4_failure_recovery",
            "fault_scenario": scenario.name,
        },
    }


async def build_rows(count: int) -> list[dict[str, Any]]:
    if count < len(SCENARIOS):
        raise ValueError(f"count must be at least {len(SCENARIOS)}")
    rows = []
    for index in range(count):
        rows.append(await execute_scenario(SCENARIOS[index % len(SCENARIOS)], index))
    return rows


def write_rows(output: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    scenarios = Counter(row["lineage"]["fault_scenario"] for row in rows)
    return {
        "rows": len(rows),
        "scenarios": dict(sorted(scenarios.items())),
        "trace_ids": len({row["meta"]["trace_id"] for row in rows}),
        "output": str(output),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=600)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/teacher/private/v4/failure_recovery_harness.jsonl"),
    )
    args = parser.parse_args()
    rows = asyncio.run(build_rows(args.count))
    print(json.dumps(write_rows(args.output, rows), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
