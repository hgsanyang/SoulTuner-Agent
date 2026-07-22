import asyncio

import pytest

from agent.tool_orchestrator import BoundedToolOrchestrator, ToolRegistry
from schemas.dialog_state import DialogMusicState, compile_dialog_state_to_plan
from schemas.query_plan import MusicQueryPlan
from schemas.tool_plan import ToolCall, ToolName, ToolPlan, tool_plan_alignment_issues


def test_tool_plan_rejects_arbitrary_external_url():
    with pytest.raises(ValueError):
        ToolCall(
            id="web",
            name=ToolName.SEARCH_EXTERNAL_MUSIC,
            arguments={"requirements": "fetch https://example.com/private"},
        )


def test_tool_plan_rejects_unknown_arguments():
    with pytest.raises(ValueError):
        ToolCall(
            id="graph",
            name=ToolName.SEARCH_GRAPH,
            arguments={"artist_entities": ["周杰伦"], "cypher": "MATCH (n) DETACH DELETE n"},
        )


def test_tool_plan_alignment_detects_missing_audio_tool():
    plan = MusicQueryPlan(
        intent_type="vector_search",
        retrieval_plan={
            "use_vector": True,
            "vector_acoustic_query": "quiet sparse piano with low energy",
        },
        tool_plan={
            "version": "1.0",
            "request_mode": "recommendation",
            "tool_calls": [],
        },
    )
    assert tool_plan_alignment_issues(plan) == ["audio_signal_without_search_audio"]


def test_direct_web_information_does_not_require_local_gap_probe():
    plan = MusicQueryPlan(
        intent_type="web_search",
        retrieval_plan={
            "metadata_constraints": {"recency_required": True},
            "use_web_search": True,
        },
        tool_plan={
            "version": "1.0",
            "request_mode": "recommendation",
            "tool_calls": [
                {
                    "id": "external",
                    "name": "search_external_music",
                    "arguments": {"requirements": "本周新歌"},
                }
            ],
        },
    )
    assert tool_plan_alignment_issues(plan) == []


def test_dialog_delta_compiler_refreshes_tool_plan_after_state_mutation():
    state = DialogMusicState.model_validate(
        {
            "hard_constraints": {"artist_entities": ["陈绮贞"]},
            "soft_intent": {"vibe": "安静"},
        }
    )
    plan = compile_dialog_state_to_plan(state, "这个歌手的其它安静作品")
    assert {call.name for call in plan.tool_plan.tool_calls} == {
        ToolName.SEARCH_GRAPH,
        ToolName.SEARCH_AUDIO,
    }


def test_tool_plan_rejects_cycles():
    with pytest.raises(ValueError):
        ToolPlan(
            request_mode="recommendation",
            tool_calls=[
                ToolCall(id="a", name="retrieve_memory", arguments={}, depends_on=["b"]),
                ToolCall(id="b", name="retrieve_memory", arguments={}, depends_on=["a"]),
            ],
        )


def test_legacy_plan_compiles_graph_and_audio_tools():
    plan = MusicQueryPlan.model_validate(
        {
            "intent_type": "hybrid_search",
            "retrieval_plan": {
                "use_graph": True,
                "use_vector": True,
                "hard_constraints": {"artist_entities": ["Artist"]},
                "soft_intent": {"avoid": ["harsh drums"], "vibe": "soft"},
                "vector_acoustic_query": "soft intimate acoustic music",
            },
        }
    )
    assert [call.name for call in plan.tool_plan.tool_calls] == [
        ToolName.SEARCH_GRAPH,
        ToolName.SEARCH_AUDIO,
    ]


def test_orchestrator_runs_independent_calls_in_parallel_and_dependencies_after():
    registry = ToolRegistry()

    async def graph(args, dependencies):
        await asyncio.sleep(0.01)
        return ["graph"]

    async def audio(args, dependencies):
        await asyncio.sleep(0.01)
        return ["audio"]

    def gap(args, dependencies):
        return {"sources": sorted(dependencies), "metadata": {"needs_replan": False}}

    registry.register(ToolName.SEARCH_GRAPH, graph)
    registry.register(ToolName.SEARCH_AUDIO, audio)
    registry.register(ToolName.INSPECT_CATALOG_GAP, gap)
    plan = ToolPlan(
        request_mode="recommendation",
        tool_calls=[
            ToolCall(id="graph", name="search_graph", arguments={}),
            ToolCall(
                id="audio",
                name="search_audio",
                arguments={"acoustic_queries": ["quiet"]},
            ),
            ToolCall(
                id="gap",
                name="inspect_catalog_gap",
                arguments={},
                depends_on=["graph", "audio"],
            ),
        ],
    )
    result = asyncio.run(BoundedToolOrchestrator(registry).run(plan))
    assert result.by_call_id["gap"].data["sources"] == ["audio", "graph"]


def test_orchestrator_allows_only_one_replan():
    registry = ToolRegistry()
    registry.register(ToolName.SEARCH_GRAPH, lambda args, deps: [])
    registry.register(ToolName.SEARCH_AUDIO, lambda args, deps: ["fallback"])
    initial = ToolPlan(
        request_mode="recommendation",
        tool_calls=[ToolCall(id="graph", name="search_graph", arguments={})],
    )

    def replan(plan, observations):
        return ToolPlan(
            request_mode="recommendation",
            max_replans=1,
            tool_calls=[
                ToolCall(id="graph", name="search_graph", arguments={}),
                ToolCall(
                    id="audio",
                    name="search_audio",
                    arguments={"acoustic_queries": ["fallback"]},
                ),
            ],
        )

    result = asyncio.run(BoundedToolOrchestrator(registry).run(initial, replanner=replan))
    assert result.replans_used == 1
    assert result.by_call_id["audio"].success is True
