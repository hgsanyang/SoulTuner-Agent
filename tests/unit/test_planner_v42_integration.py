from __future__ import annotations

import asyncio
from copy import deepcopy
import json

import pytest

from agent.intent.adapters import PlannerPayload
from agent.intent.v42_adapter import (
    build_v42_user_message,
    compile_guarded_v42,
    plan_with_v42_contract,
    uses_v42_contract,
)
from deploy.self_hosted_35b.planner_guard import build_safe_plan
from schemas.tool_plan import ToolName, tool_plan_alignment_issues
from data.sft.build_sft_chatml import build_user_message


def test_contract_auto_detection_is_narrow() -> None:
    assert uses_v42_contract("vllm", "soultuner-planner-v4.2-35b", "auto") is True
    assert uses_v42_contract("sglang", "soultuner-v42", "auto") is True
    assert uses_v42_contract("dashscope", "qwen3.7-plus", "auto") is False
    assert uses_v42_contract("vllm", "unrelated-model", "auto") is False
    assert uses_v42_contract("dashscope", "anything", "v42") is True
    with pytest.raises(ValueError, match="Unsupported planner contract"):
        uses_v42_contract("vllm", "soultuner-v42", "unknown")


def test_v42_user_message_matches_training_conditioning_order() -> None:
    payload = PlannerPayload(
        user_input="给我和刚才那首听感相似的",
        user_preferences="喜欢空间感",
        chat_history="用户：播放了一首歌",
        previous_plan="dense=required",
        current_date="2026-08-13",
        retrieved_memories="用户偏好柔和女声\n不喜欢尖锐高频",
    )
    message = build_v42_user_message(
        payload,
        {"reference_title": "Dreams", "reference_artist": "Fleetwood Mac"},
    )
    assert message.splitlines()[0] == "[用户画像] 喜欢空间感"
    assert "[对话历史]" in message
    assert "[长期记忆] 用户偏好柔和女声；不喜欢尖锐高频" in message
    assert "[上轮检索计划] dense=required" in message
    assert "[上轮推荐结果] 1. Dreams — Fleetwood Mac" in message
    assert "[已解析参考歌曲]" not in message
    assert message.endswith("[当前输入] 给我和刚才那首听感相似的")


def test_runtime_formatter_matches_sft_formatter_without_reference_anchor() -> None:
    record = {
        "profile_snapshot": "喜欢空间感",
        "retrieved_memories": ["用户偏好柔和女声", "不喜欢尖锐高频"],
        "chat_history": "用户：播放了一首歌",
        "previous_plan": "dense=required",
        "current_query": "再来一首",
    }
    runtime_message = build_v42_user_message(
        PlannerPayload(
            user_input=record["current_query"],
            user_preferences=record["profile_snapshot"],
            chat_history=record["chat_history"],
            previous_plan=record["previous_plan"],
            current_date="2026-08-13",
            retrieved_memories="\n".join(record["retrieved_memories"]),
        ),
        {},
    )
    assert runtime_message == build_user_message(record)


def test_compile_dense_primary_plan_into_existing_retrieval_and_tools() -> None:
    guarded = build_safe_plan("我心情很差，想听温暖治愈的歌")
    plan = compile_guarded_v42("我心情很差，想听温暖治愈的歌", guarded)
    assert plan.intent_type == "hybrid_search"
    assert plan.retrieval_plan.use_graph is True
    assert plan.retrieval_plan.use_vector is True
    assert plan.retrieval_plan.hints.mood in {"治愈", "温暖", "低落"}
    assert plan.parameters["lane_policy"] == {
        "graph": "optional",
        "dense": "required",
        "web": "off",
    }
    assert {call.name for call in plan.tool_plan.tool_calls} == {
        ToolName.SEARCH_GRAPH,
        ToolName.SEARCH_AUDIO,
    }
    assert tool_plan_alignment_issues(plan) == []


def test_compile_dialogue_and_clarification_without_tools() -> None:
    chat = compile_guarded_v42("你好", build_safe_plan("你好"))
    assert chat.intent_type == "general_chat"
    assert chat.tool_plan.tool_calls == []

    clarify = compile_guarded_v42(
        "给我和刚才那首听感相似的",
        build_safe_plan("给我和刚才那首听感相似的"),
    )
    assert clarify.intent_type == "clarification"
    assert clarify.tool_plan.needs_clarification is True
    assert clarify.tool_plan.tool_calls == []


class _FakeLLM:
    model_name = "soultuner-planner-v4.2-35b"
    openai_api_base = "http://127.0.0.1:8000/v1"


def test_endpoint_candidate_is_guarded_before_compilation(monkeypatch) -> None:
    candidate = deepcopy(build_safe_plan("低音更重、鼓点更大的歌"))
    for generated in ("version", "source", "execution"):
        candidate.pop(generated)

    captured: dict = {}

    async def fake_post(url, body, timeout, headers=None):
        captured.update(url=url, body=body, timeout=timeout, headers=headers)
        return {"choices": [{"message": {"content": json.dumps(candidate, ensure_ascii=False)}}]}

    monkeypatch.setattr("agent.intent.v42_adapter._post_json", fake_post)
    plan, findings = asyncio.run(
        plan_with_v42_contract(
            _FakeLLM(),
            PlannerPayload("低音更重、鼓点更大的歌", "无", "", "", "2026-08-13"),
            max_tokens=1024,
            timeout=60,
            api_key="private-test-key",
        )
    )
    assert captured["url"] == "http://127.0.0.1:8000/v1/chat/completions"
    assert captured["body"]["temperature"] == 0.0
    assert captured["body"]["enable_thinking"] is False
    assert captured["headers"]["Authorization"] == "Bearer private-test-key"
    assert findings == ["模型候选通过结构与策略守卫"]
    assert plan.intent_type == "vector_search"
    assert tool_plan_alignment_issues(plan) == []


def test_malformed_endpoint_output_falls_back_to_safe_plan(monkeypatch) -> None:
    async def fake_post(*args, **kwargs):
        return {"choices": [{"message": {"content": "not-json"}}]}

    monkeypatch.setattr("agent.intent.v42_adapter._post_json", fake_post)
    plan, findings = asyncio.run(
        plan_with_v42_contract(
            _FakeLLM(),
            PlannerPayload("我心情很差，想听治愈的歌", "无", "", "", "2026-08-13"),
            max_tokens=1024,
            timeout=60,
        )
    )
    assert plan.intent_type == "hybrid_search"
    assert plan.retrieval_plan.use_vector is True
    assert findings == ["未提供模型候选，已使用确定性安全计划"]


def test_guard_rejects_semantically_inconsistent_payloads() -> None:
    from deploy.self_hosted_35b.planner_guard import guard_candidate

    dense = deepcopy(build_safe_plan("低音更重、鼓点更大的歌"))
    for generated in ("version", "source", "execution"):
        dense.pop(generated)
    dense["acoustic_queries"] = []
    accepted, findings = guard_candidate("低音更重、鼓点更大的歌", dense)
    assert accepted["source"] == "deterministic_guard"
    assert any("dense=required" in finding for finding in findings)

    chat = deepcopy(build_safe_plan("你好"))
    for generated in ("version", "source", "execution"):
        chat.pop(generated)
    chat["soft"]["goal"] = "偷偷带入检索条件"
    accepted, findings = guard_candidate("你好", chat)
    assert accepted["source"] == "deterministic_guard"
    assert any("不得携带检索字段" in finding for finding in findings)
