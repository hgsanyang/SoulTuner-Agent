"""V3 contract: invariants, V2<->V3 reversibility, compile equivalence."""

from __future__ import annotations

import pytest

from schemas.planner_decision import PlannerDecisionV2, compile_to_query_plan
from schemas.planner_decision_v3 import (
    PlannerDecisionV3,
    compile_v3_to_query_plan,
    migrate_v2_to_v3,
    migrate_v3_to_v2,
    v3_to_legacy_intent,
)


def test_valid_v3_decisions():
    PlannerDecisionV3(request_kind="recommendation", tool_names=["graph", "dense"],
                      acoustic_queries=["warm mellow"])
    PlannerDecisionV3(request_kind="information", tool_names=["web"])
    # a fact already in the catalog/knowledge cards need not hit the network
    PlannerDecisionV3(request_kind="information", tool_names=["graph"])
    PlannerDecisionV3(request_kind="acquisition", tool_names=["web", "ingest"])
    PlannerDecisionV3(request_kind="library", tool_names=["library"])
    PlannerDecisionV3(request_kind="conversation")
    PlannerDecisionV3(request_kind="recommendation", response_mode="clarify",
                      clarification="你想要哪种?")


def test_library_and_acquisition_require_their_registered_lanes():
    with pytest.raises(ValueError):
        PlannerDecisionV3(request_kind="acquisition", tool_names=["web"])
    with pytest.raises(ValueError):
        PlannerDecisionV3(request_kind="library", tool_names=["graph"])
    with pytest.raises(ValueError):
        PlannerDecisionV3(request_kind="library")


def test_clarify_is_an_action_not_a_kind():
    # clarify locks to text and runs no tools
    with pytest.raises(ValueError):
        PlannerDecisionV3(request_kind="recommendation", response_mode="clarify")
    with pytest.raises(ValueError):
        PlannerDecisionV3(request_kind="recommendation", response_mode="clarify",
                          clarification="?", tool_names=["graph"])
    # clarification text without clarify mode is rejected
    with pytest.raises(ValueError):
        PlannerDecisionV3(request_kind="recommendation", tool_names=["graph"],
                          clarification="不该出现")


def test_request_kind_requires_capable_lane():
    # web alone cannot serve a recommendation (needs a local recall lane)
    with pytest.raises(ValueError):
        PlannerDecisionV3(request_kind="recommendation", tool_names=["web"])
    # dense (acoustic) cannot answer a factual question
    with pytest.raises(ValueError):
        PlannerDecisionV3(request_kind="information", tool_names=["dense"])
    with pytest.raises(ValueError):
        PlannerDecisionV3(request_kind="recommendation")  # no lanes at all
    with pytest.raises(ValueError):
        PlannerDecisionV3(request_kind="conversation", tool_names=["dense"])


def test_web_search_migration_is_flagged_ambiguous():
    from schemas.planner_decision_v3 import migration_note

    # "Billboard 冠军是谁" vs "推荐几首本周新歌" share the legacy web_search intent
    assert migration_note(PlannerDecisionV2(intent="web_search", tool_names=["web"])).ambiguous
    assert migration_note(PlannerDecisionV2(intent="clarification", clarification="?")).ambiguous
    assert not migration_note(
        PlannerDecisionV2(intent="hybrid_search", tool_names=["graph", "dense"],
                          acoustic_queries=["x"])
    ).ambiguous


@pytest.mark.parametrize(
    "intent,tools,expected_kind",
    [
        ("graph_search", ["graph"], "recommendation"),
        ("vector_search", ["dense"], "recommendation"),
        ("hybrid_search", ["graph", "dense"], "recommendation"),
        ("web_search", ["web"], "information"),
        ("general_chat", [], "conversation"),
        ("acquire_music", [], "acquisition"),
    ],
)
def test_v2_to_v3_kind_mapping(intent, tools, expected_kind):
    v2 = PlannerDecisionV2(intent=intent, tool_names=tools,
                           acoustic_queries=["x"] if "dense" in tools else [])
    v3 = migrate_v2_to_v3(v2)
    assert v3.request_kind == expected_kind
    assert v3.response_mode == "answer"


def test_clarification_migrates_to_response_mode():
    v2 = PlannerDecisionV2(intent="clarification", clarification="想听哪种?")
    v3 = migrate_v2_to_v3(v2)
    assert v3.response_mode == "clarify"
    assert v3.clarification == "想听哪种?"
    assert v3.tool_names == []


@pytest.mark.parametrize(
    "intent,tools",
    [
        ("graph_search", ["graph"]),
        ("vector_search", ["dense"]),
        ("hybrid_search", ["graph", "dense"]),
        ("web_search", ["web"]),
        ("general_chat", []),
    ],
)
def test_round_trip_identity_on_v2_lane_subset(intent, tools):
    v2 = PlannerDecisionV2(intent=intent, tool_names=tools,
                           acoustic_queries=["x"] if "dense" in tools else [])
    back = migrate_v3_to_v2(migrate_v2_to_v3(v2))
    assert back.intent == v2.intent
    assert sorted(back.tool_names) == sorted(v2.tool_names)


def test_compile_equivalence_v2_vs_v3():
    v2 = PlannerDecisionV2(intent="hybrid_search", tool_names=["graph", "dense"],
                           acoustic_queries=["slow warm piano"])
    v3 = migrate_v2_to_v3(v2)
    p2, p3 = compile_to_query_plan(v2), compile_v3_to_query_plan(v3)
    assert p2.intent_type == p3.intent_type
    assert p2.retrieval_plan.use_graph == p3.retrieval_plan.use_graph
    assert p2.retrieval_plan.use_vector == p3.retrieval_plan.use_vector
    assert p2.retrieval_plan.vector_acoustic_queries == p3.retrieval_plan.vector_acoustic_queries


def test_legacy_intent_recovered_from_lanes():
    assert v3_to_legacy_intent(PlannerDecisionV3(request_kind="recommendation",
                                                 tool_names=["graph"])) == "graph_search"
    assert v3_to_legacy_intent(PlannerDecisionV3(request_kind="recommendation", tool_names=["dense"],
                                                 acoustic_queries=["x"])) == "vector_search"
    assert v3_to_legacy_intent(PlannerDecisionV3(request_kind="recommendation",
                                                 tool_names=["graph", "dense"],
                                                 acoustic_queries=["x"])) == "hybrid_search"
    assert v3_to_legacy_intent(PlannerDecisionV3(request_kind="information",
                                                 tool_names=["web"])) == "web_search"
