import pytest

from agent.intent.parsing import clean_json_response, parse_music_query_plan
from agent.intent.planner import PlannerResultCache, apply_routing_guardrails, normalize_planner_grounding


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


def test_planner_cache_returns_deep_copy_and_expires():
    now = [100.0]
    cache = PlannerResultCache(ttl_seconds=10, max_entries=2, clock=lambda: now[0])
    plan = parse_music_query_plan('{"intent_type":"vector_search"}')
    cache.put("same-query-profile", plan)

    cached = cache.get("same-query-profile")
    assert cached is not None
    cached.intent_type = "general_chat"
    assert cache.get("same-query-profile").intent_type == "vector_search"

    now[0] = 111.0
    assert cache.get("same-query-profile") is None


def test_planner_cache_key_includes_profile_context():
    common = {
        "user_input": "还是那个感觉",
        "user_preferences": "喜欢 city pop",
        "previous_plan": "",
        "graphzep_facts": "",
        "provider": "dashscope",
        "model_name": "qwen3.7-plus",
        "current_date": "2026-06-21",
    }
    first = PlannerResultCache.make_key(chat_history="上一轮：日语", **common)
    second = PlannerResultCache.make_key(chat_history="上一轮：中文", **common)
    assert first != second


def test_cantopop_grounding_refines_generic_chinese_language():
    plan = parse_music_query_plan(
        """
        {
          "intent_type":"hybrid_search",
          "retrieval_plan":{
            "hard_constraints":{"language":"Chinese","region":"Hong Kong"},
            "graph_region_filter":"Hong Kong",
            "graph_language_filter":"Chinese"
          }
        }
        """
    )

    normalized = normalize_planner_grounding(plan, "90 年代港乐那种怀旧感")

    assert normalized.retrieval_plan.hard_constraints.language == "Cantonese"
    assert normalized.retrieval_plan.graph_language_filter == "Cantonese"
