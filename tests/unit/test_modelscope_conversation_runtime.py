from __future__ import annotations

import json
from urllib.error import URLError

from deploy.modelscope_space import conversation_runtime


class _Response:
    def __init__(self, content: str):
        self.content = content

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self) -> bytes:
        return json.dumps(
            {"choices": [{"message": {"content": self.content}}]},
            ensure_ascii=False,
        ).encode("utf-8")


def _rows():
    return [
        {"title": "Rain Window", "artist": "Open Artist", "tags": ["安静", "氛围"]},
        {"title": "Soft Light", "artist": "Open Artist 2", "tags": ["温暖"]},
    ]


def test_recommendation_opening_uses_base_served_model_not_planner(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return _Response("雨天就从这两首安静但明亮的作品开始，听完告诉我你更喜欢哪种空间感。")

    monkeypatch.setenv("SOULTUNER_CHAT_MODEL", "qwen3.6-35b-a3b")
    monkeypatch.setenv("SOULTUNER_PLANNER_MODEL", "soultuner-v4.2-35b")
    monkeypatch.setenv("SOULTUNER_PLANNER_BASE_URL", "http://127.0.0.1:8000/v1")
    monkeypatch.setattr(conversation_runtime.urllib.request, "urlopen", fake_urlopen)

    text, status = conversation_runtime.recommendation_opening(
        "暴雨天想听安静但不压抑的音乐",
        {"evidence": {"brief_reason": "由听感向量主导"}},
        _rows(),
        {"positive_tags": {"氛围": 2}},
    )

    assert "空间感" in text
    assert status == "35B 基座自然语言已就绪"
    assert captured["url"] == "http://127.0.0.1:8000/v1/chat/completions"
    assert captured["payload"]["model"] == "qwen3.6-35b-a3b"
    assert captured["payload"]["model"] != "soultuner-v4.2-35b"
    assert captured["payload"]["enable_thinking"] is False


def test_planner_model_cannot_be_reused_for_prose(monkeypatch):
    monkeypatch.setenv("SOULTUNER_CHAT_MODEL", "soultuner-v4.2-35b")
    monkeypatch.setenv("SOULTUNER_PLANNER_MODEL", "soultuner-v4.2-35b")

    text, status = conversation_runtime.recommendation_opening(
        "想听安静的",
        {"evidence": {"brief_reason": "听感检索"}},
        _rows(),
    )

    assert "Rain Window" in text
    assert "RuntimeError" in status


def test_recommendation_opening_safely_falls_back_when_endpoint_is_cold(monkeypatch):
    monkeypatch.setenv("SOULTUNER_CHAT_MODEL", "qwen3.6-35b-a3b")
    monkeypatch.setenv("SOULTUNER_PLANNER_MODEL", "soultuner-v4.2-35b")
    monkeypatch.setenv("SOULTUNER_PLANNER_BASE_URL", "http://127.0.0.1:8000/v1")
    monkeypatch.setattr(
        conversation_runtime.urllib.request,
        "urlopen",
        lambda *args, **kwargs: (_ for _ in ()).throw(URLError("cold")),
    )

    text, status = conversation_runtime.recommendation_opening(
        "想听安静的",
        {"evidence": {"brief_reason": "听感检索"}},
        _rows(),
    )

    assert "Rain Window" in text
    assert status == "自然语言安全回退（URLError）"


def test_general_chat_passes_recent_history_and_session_memory(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return _Response("当然记得，你更偏爱安静的氛围感。今天想继续这个方向吗？")

    monkeypatch.setenv("SOULTUNER_CHAT_MODEL", "qwen3.6-35b-a3b")
    monkeypatch.setenv("SOULTUNER_PLANNER_MODEL", "soultuner-v4.2-35b")
    monkeypatch.setenv("SOULTUNER_PLANNER_BASE_URL", "http://127.0.0.1:8000/v1")
    monkeypatch.setattr(conversation_runtime.urllib.request, "urlopen", fake_urlopen)

    text, status = conversation_runtime.general_chat(
        "你还记得我喜欢什么吗？",
        [{"role": "user", "content": "我喜欢安静的氛围音乐"}],
        {
            "positive_tags": {"氛围": 3},
            "last_recommendation_query": "雨天安静但不压抑",
            "last_recommendations": [{"title": "Rain Window", "artist": "Open Artist"}],
        },
    )

    user_prompt = captured["payload"]["messages"][1]["content"]
    assert "安静的氛围音乐" in user_prompt
    assert "氛围" in user_prompt
    assert "Rain Window" in user_prompt
    assert "雨天安静但不压抑" in user_prompt
    assert "安静" in text
    assert status == "35B 基座自然语言已就绪"


def test_space_app_uses_one_orchestrated_conversation_surface():
    source = conversation_runtime.__file__.replace("conversation_runtime.py", "app.py")
    content = open(source, encoding="utf-8").read()
    assert "from conversation_runtime import general_chat, recommendation_opening" in content
    assert "opening, conversation_status = recommendation_opening(" in content
    assert "def unified_turn(" in content
    assert 'api_name="conversation"' in content
