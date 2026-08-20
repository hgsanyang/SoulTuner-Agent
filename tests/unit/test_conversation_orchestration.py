"""Role boundaries between the 35B planner and natural-language models."""

from __future__ import annotations

import asyncio

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from config.settings import settings


def test_conversation_factory_uses_its_own_model_settings(monkeypatch) -> None:
    import llms.chat_models as chat_models

    captured: dict[str, object] = {}

    def fake_factory(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(chat_models, "get_chat_model", fake_factory)
    monkeypatch.setattr(settings, "conversation_llm_provider", "dashscope", raising=False)
    monkeypatch.setattr(settings, "conversation_llm_model", "qwen3.7-plus", raising=False)
    monkeypatch.setattr(settings, "intent_llm_provider", "vllm")
    monkeypatch.setattr(settings, "intent_llm_model", "soultuner-planner-v4.2-35b")
    monkeypatch.setattr(settings, "intent_planner_contract", "v42")

    chat_models.get_conversation_chat_model()

    assert captured["provider"] == "dashscope"
    assert captured["model_name"] == "qwen3.7-plus"
    assert captured["temperature"] == 0.7


def test_planner_lora_cannot_be_used_as_conversation_model(monkeypatch) -> None:
    import llms.chat_models as chat_models

    monkeypatch.setattr(settings, "conversation_llm_provider", "vllm", raising=False)
    monkeypatch.setattr(
        settings,
        "conversation_llm_model",
        "soultuner-v4.2-35b",
        raising=False,
    )
    monkeypatch.setattr(settings, "intent_llm_provider", "vllm")
    monkeypatch.setattr(settings, "intent_llm_model", "soultuner-v4.2-35b")
    monkeypatch.setattr(settings, "intent_planner_contract", "v42")

    with pytest.raises(chat_models.PlannerRoleViolation, match="planner-only"):
        chat_models.get_conversation_chat_model()


def test_same_vllm_endpoint_can_serve_base_chat_and_planner_lora(monkeypatch) -> None:
    """The role boundary is served-model identity, not server/process identity."""

    import llms.chat_models as chat_models

    captured: dict[str, object] = {}

    def fake_factory(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(chat_models, "get_chat_model", fake_factory)
    monkeypatch.setattr(settings, "conversation_llm_provider", "vllm")
    monkeypatch.setattr(settings, "conversation_llm_model", "qwen3.6-35b-chat")
    monkeypatch.setattr(settings, "intent_llm_provider", "vllm")
    monkeypatch.setattr(settings, "intent_llm_model", "soultuner-v4.2-35b")
    monkeypatch.setattr(settings, "intent_planner_contract", "v42")

    chat_models.get_conversation_chat_model()

    assert captured["provider"] == "vllm"
    assert captured["model_name"] == "qwen3.6-35b-chat"


def test_explainer_inherits_conversation_model_not_planner(monkeypatch) -> None:
    import llms.chat_models as chat_models

    captured: dict[str, object] = {}

    def fake_factory(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(chat_models, "get_chat_model", fake_factory)
    monkeypatch.setattr(settings, "conversation_llm_provider", "dashscope", raising=False)
    monkeypatch.setattr(settings, "conversation_llm_model", "qwen3.7-plus", raising=False)
    monkeypatch.setattr(settings, "explain_llm_provider", "")
    monkeypatch.setattr(settings, "explain_llm_model", "")
    monkeypatch.setattr(settings, "intent_llm_provider", "vllm")
    monkeypatch.setattr(settings, "intent_llm_model", "soultuner-planner-v4.2-35b")
    monkeypatch.setattr(settings, "intent_planner_contract", "v42")

    chat_models.get_explain_chat_model()

    assert captured["provider"] == "dashscope"
    assert captured["model_name"] == "qwen3.7-plus"


def test_general_chat_node_never_calls_retrieval_or_planner_llm(monkeypatch) -> None:
    import agent.music_graph as music_graph

    conversation_llm = FakeListChatModel(responses=["当然可以。你更想聊旋律，还是歌词？"])
    monkeypatch.setattr(music_graph, "get_conversation_llm", lambda: conversation_llm)
    monkeypatch.setattr(
        music_graph,
        "get_llm",
        lambda: (_ for _ in ()).throw(AssertionError("general chat reused retrieval/planner model")),
    )

    graph = music_graph.MusicRecommendationGraph.__new__(
        music_graph.MusicRecommendationGraph
    )
    graph._explanation_queues = {}
    result = asyncio.run(
        graph.general_chat_node(
            {
                "input": "我今天有点累，陪我聊聊音乐吧",
                "chat_history": [],
                "prompt_context": {
                    "chat_history": "",
                    "chat_memory": "用户喜欢温柔的音乐",
                },
                "assembled_context": {"explicit_profile": "喜欢温柔音乐"},
                "metadata": {"request_id": "request-1"},
                "step_count": 0,
            }
        )
    )

    assert "旋律" in result["final_response"]
    assert result["response_meta"]["role"] == "conversation"
    assert result["response_meta"]["planner_used_for_generation"] is False


def test_clarification_decision_stays_planner_owned_but_is_naturally_rendered(
    monkeypatch,
) -> None:
    import agent.music_graph as music_graph

    conversation_llm = FakeListChatModel(
        responses=["我能接着帮你找，不过先确认一下：你说的是上一轮的哪一首？"]
    )
    monkeypatch.setattr(music_graph, "get_conversation_llm", lambda: conversation_llm)

    graph = music_graph.MusicRecommendationGraph.__new__(
        music_graph.MusicRecommendationGraph
    )
    graph._explanation_queues = {}
    result = asyncio.run(
        graph.clarification_node(
            {
                "input": "给我和刚才那首相似的",
                "chat_history": [],
                "graphzep_facts": "用户喜欢空间感",
                "clarification": {
                    "question": "你指的是上一轮哪一首歌？",
                    "reason": "unresolved_reference",
                    "options": ["第一首", "第二首"],
                },
                "step_count": 0,
            }
        )
    )

    assert "哪一首" in result["final_response"]
    assert result["clarification_options"] == ["第一首", "第二首"]
    assert result["response_meta"]["role"] == "conversation_clarification"
    assert result["response_meta"]["planner_decision_preserved"] is True
