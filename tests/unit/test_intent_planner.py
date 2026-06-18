import pytest

from agent.intent.parsing import clean_json_response, parse_music_query_plan
from agent.intent.planner import apply_routing_guardrails


def test_clean_json_response_removes_thinking_and_fence():
    raw = '<think>hidden</think>\n```json\n{"intent_type":"general_chat"}\n```'
    assert clean_json_response(raw) == '{"intent_type":"general_chat"}'


def test_parse_music_query_plan_validates_schema():
    plan = parse_music_query_plan('{"intent_type":"vector_search"}')
    assert plan.intent_type == "vector_search"
    assert plan.retrieval_plan.use_graph is False


def test_parse_music_query_plan_rejects_unknown_intent():
    with pytest.raises(ValueError):
        parse_music_query_plan('{"intent_type":"unknown"}')


def test_music_request_guardrail_prevents_general_chat():
    plan = parse_music_query_plan('{"intent_type":"general_chat"}')
    corrected = apply_routing_guardrails(plan, "随便来几首好听的")
    assert corrected.intent_type == "vector_search"
    assert corrected.retrieval_plan.use_vector is True
    assert corrected.retrieval_plan.vector_acoustic_query == "随便来几首好听的"


def test_actual_smalltalk_remains_general_chat():
    plan = parse_music_query_plan('{"intent_type":"general_chat"}')
    corrected = apply_routing_guardrails(plan, "你好，今天过得怎么样")
    assert corrected.intent_type == "general_chat"
