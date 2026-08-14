import asyncio

import pytest

from agent.intent.parsing import clean_json_response, parse_music_query_plan
from agent.intent.planner import IntentPlanner, PlannerResultCache


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


def test_planner_cache_key_includes_resolved_reference_context():
    common = {
        "user_input": "再来一首类似的",
        "user_preferences": "喜欢 city pop",
        "chat_history": "",
        "previous_plan": "",
        "graphzep_facts": "",
        "provider": "vllm",
        "model_name": "soultuner-planner-v4.2-35b",
        "current_date": "2026-08-13",
        "planner_contract": "v42",
    }
    first = PlannerResultCache.make_key(
        reference_context={"reference_title": "Plastic Love", "reference_artist": "竹内玛利亚"},
        **common,
    )
    second = PlannerResultCache.make_key(
        reference_context={"reference_title": "Stay With Me", "reference_artist": "松原美纪"},
        **common,
    )
    assert first != second


def test_preassembled_context_never_keys_cache_with_raw_long_memory(monkeypatch):
    """Raw memory must not bypass or fragment the canonical bounded view."""

    class FakeLLM:
        model_name = "fake-planner"

    calls = []

    async def fake_plan(_llm, _system, _human, payload):
        calls.append(payload)
        return parse_music_query_plan('{"intent_type":"vector_search"}')

    monkeypatch.setattr("agent.intent.planner.plan_with_generic_structured_output", fake_plan)
    monkeypatch.setattr("agent.intent.planner.settings.intent_llm_provider", "generic")
    monkeypatch.setattr("agent.intent.planner.settings.llm_default_provider", "generic")

    planner = IntentPlanner(lambda: FakeLLM())
    planner._cache = PlannerResultCache(ttl_seconds=60, max_entries=8)
    common = {
        "user_input": "继续刚才的感觉",
        "user_preferences": "【预算后的统一记忆】喜欢舒缓音乐",
        "chat_history": "最近一轮",
        "previous_plan": "",
        "user_id": "u1",
        "conversation_id": "session-1",
        "context_preassembled": True,
    }

    asyncio.run(planner.plan(graphzep_facts="未截断原始记忆 A" * 100, **common))
    asyncio.run(planner.plan(graphzep_facts="完全不同的未截断原始记忆 B" * 100, **common))

    assert len(calls) == 1
