from llms import chat_models
from config.settings import settings


def test_intent_chat_model_uses_teacher_model_by_default(monkeypatch):
    captured = {}

    def fake_get_chat_model(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(chat_models, "get_chat_model", fake_get_chat_model)
    monkeypatch.setattr(settings, "intent_llm_provider", "dashscope")
    monkeypatch.setattr(settings, "intent_llm_model", "qwen3.7-plus")
    monkeypatch.setattr(settings, "intent_llm_fast_model", "qwen-turbo")
    monkeypatch.setattr(settings, "planner_quality_mode", "teacher")

    chat_models.get_intent_chat_model()

    assert captured["model_name"] == "qwen3.7-plus"


def test_intent_chat_model_uses_fast_model_only_when_explicit(monkeypatch):
    captured = {}

    def fake_get_chat_model(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(chat_models, "get_chat_model", fake_get_chat_model)
    monkeypatch.setattr(settings, "intent_llm_provider", "dashscope")
    monkeypatch.setattr(settings, "intent_llm_model", "qwen3.7-plus")
    monkeypatch.setattr(settings, "intent_llm_fast_model", "qwen-turbo")
    monkeypatch.setattr(settings, "planner_quality_mode", "fast")

    chat_models.get_intent_chat_model()

    assert captured["model_name"] == "qwen-turbo"
