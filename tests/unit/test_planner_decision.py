"""PlannerDecisionV2 contract + deterministic compiler (Phase A-1)."""

from schemas.planner_decision import (
    DecisionHard,
    DecisionHints,
    DecisionSoft,
    PlannerDecisionV2,
    compile_to_query_plan,
    decision_token_estimate,
    from_query_plan,
)
from schemas.query_plan import MusicQueryPlan
from schemas.tool_plan import ToolName


def test_compile_graph_intent_builds_graph_tool_and_legacy_fields():
    decision = PlannerDecisionV2(
        intent="graph_search",
        hard=DecisionHard(artist=["周杰伦"], song=["稻香"]),
        tool_names=["graph"],
    )
    plan = compile_to_query_plan(decision)
    assert plan.intent_type == "graph_search"
    # legacy flat fields filled by RetrievalPlan.sync validator
    assert plan.retrieval_plan.graph_artist_entities == ["周杰伦"]
    assert plan.retrieval_plan.use_graph is True
    tools = {c.name for c in plan.tool_plan.tool_calls}
    assert ToolName.SEARCH_GRAPH in tools


def test_compile_hybrid_builds_graph_and_audio():
    decision = PlannerDecisionV2(
        intent="hybrid_search",
        hints=DecisionHints(genre=["Folk"], mood=["Calm"]),
        acoustic_queries=["warm acoustic guitar, slow tempo, rainy afternoon"],
        tool_names=["graph", "dense"],
    )
    plan = compile_to_query_plan(decision)
    tools = {c.name for c in plan.tool_plan.tool_calls}
    assert ToolName.SEARCH_GRAPH in tools
    assert ToolName.SEARCH_AUDIO in tools
    assert plan.retrieval_plan.vector_acoustic_queries[0].startswith("warm acoustic")


def test_compile_clarification_executes_no_tools():
    decision = PlannerDecisionV2(intent="clarification", clarification="你想听什么风格？")
    plan = compile_to_query_plan(decision)
    assert plan.intent_type == "clarification"
    assert plan.tool_plan.needs_clarification is True
    assert plan.tool_plan.tool_calls == []


def test_missing_tool_names_falls_back_to_intent_lanes():
    # a sparse decision (no tool_names) still executes via intent-derived lanes
    decision = PlannerDecisionV2(intent="vector_search", acoustic_queries=["dreamy ambient pads"])
    plan = compile_to_query_plan(decision)
    assert plan.retrieval_plan.use_vector is True
    assert ToolName.SEARCH_AUDIO in {c.name for c in plan.tool_plan.tool_calls}


def test_instrumental_language_normalized_to_flag():
    decision = PlannerDecisionV2(intent="vector_search", hard=DecisionHard(language="Instrumental"),
                                 acoustic_queries=["solo piano, no vocals"], tool_names=["dense"])
    plan = compile_to_query_plan(decision)
    assert plan.retrieval_plan.hard_constraints.instrumental is True
    assert plan.retrieval_plan.hard_constraints.language is None


def test_round_trip_from_query_plan_preserves_execution_semantics():
    # A full MusicQueryPlan (as the strong planner emits) -> V2 -> back -> same
    # intent, entities, acoustic queries and tool set.
    original = MusicQueryPlan.model_validate({
        "intent_type": "hybrid_search",
        "retrieval_plan": {
            "hard_constraints": {"artist_entities": ["陈奕迅"], "language": "Cantonese"},
            "hints": {"genres": ["Pop"], "mood": "Melancholy"},
            "soft_intent": {"vibe": "late night lonely walk", "avoid": ["EDM"]},
            "use_graph": True,
            "use_vector": True,
            "vector_acoustic_queries": ["slow cantonese ballad, soft vocals, night mood"],
        },
        "reasoning": "实体+声学",
    })
    v2 = from_query_plan(original)
    assert v2.intent == "hybrid_search"
    assert v2.hard.artist == ["陈奕迅"]
    assert v2.hard.language == "Cantonese"
    assert set(v2.tool_names) == {"graph", "dense"}

    recompiled = compile_to_query_plan(v2)
    assert recompiled.intent_type == original.intent_type
    assert recompiled.retrieval_plan.hard_constraints.artist_entities == ["陈奕迅"]
    assert recompiled.retrieval_plan.vector_acoustic_queries == original.retrieval_plan.vector_acoustic_queries
    assert {c.name for c in recompiled.tool_plan.tool_calls} == {c.name for c in original.tool_plan.tool_calls}


def test_compact_target_is_smaller_than_full_plan():
    original = MusicQueryPlan.model_validate({
        "intent_type": "graph_search",
        "retrieval_plan": {"hard_constraints": {"artist_entities": ["周杰伦"], "song_entities": ["稻香"]}, "use_graph": True},
        "reasoning": "明确歌手歌名",
    })
    v2 = from_query_plan(original)
    import json
    full_tokens = len(json.dumps(original.model_dump(mode="json"), ensure_ascii=False)) // 4
    assert decision_token_estimate(v2) < full_tokens  # compact target really is smaller
