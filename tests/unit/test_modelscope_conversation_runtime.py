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
        {
            "song_id": "rain",
            "title": "Rain Window",
            "artist": "Open Artist",
            "tags": ["安静", "氛围"],
            "final_score": 0.82,
            "graph_score": 0.71,
            "dense_score": 0.91,
            "dense_source": "m2d2_aura",
            "reason": "听感向量和目录标签共同支持。",
        },
        {
            "song_id": "light",
            "title": "Soft Light",
            "artist": "Open Artist 2",
            "tags": ["温暖"],
            "final_score": 0.76,
        },
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


def test_dialogue_renderer_receives_planner_decision_and_bounded_evidence(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return _Response("第一首的排序依据来自当前检索证据，而不是重新生成歌单。")

    monkeypatch.setenv("SOULTUNER_CHAT_MODEL", "qwen3.6-35b-a3b")
    monkeypatch.setenv("SOULTUNER_PLANNER_MODEL", "soultuner-v4.2-35b")
    monkeypatch.setenv("SOULTUNER_PLANNER_BASE_URL", "http://127.0.0.1:8000/v1")
    monkeypatch.setattr(conversation_runtime.urllib.request, "urlopen", fake_urlopen)

    conversation_runtime.general_chat(
        "为什么第一首排在前面？",
        [],
        {},
        planner_decision={
            "task_mode": "dialogue",
            "dialogue_mode": "information",
            "response_mode": "answer",
            "evidence": {"brief_reason": "解释当前结果"},
        },
        evidence_rows=_rows(),
    )

    prompt = captured["payload"]["messages"][1]["content"]
    assert '"dialogue_mode": "information"' in prompt
    assert "Rain Window" in prompt
    assert "不能自行把 dialogue 改成 recommendation" in prompt


def test_dialogue_renderer_receives_ordered_current_playlist_and_selected_song(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return _Response("第一首按真实融合分数排在前面。")

    monkeypatch.setenv("SOULTUNER_CHAT_MODEL", "qwen3.6-35b-a3b")
    monkeypatch.setenv("SOULTUNER_PLANNER_MODEL", "soultuner-v4.2-35b")
    monkeypatch.setenv("SOULTUNER_PLANNER_BASE_URL", "http://127.0.0.1:8000/v1")
    monkeypatch.setattr(conversation_runtime.urllib.request, "urlopen", fake_urlopen)

    conversation_runtime.general_chat(
        "为什么第一首排在前面？",
        planner_decision={"task_mode": "dialogue", "dialogue_mode": "chat"},
        current_playlist_rows=_rows(),
        selected_song_id="rain",
    )

    prompt = captured["payload"]["messages"][1]["content"]
    payload = json.loads(prompt[prompt.index("{") :])
    first = payload["current_playlist"][0]
    assert first["rank"] == 1
    assert first["is_currently_playing"] is True
    assert first["final_score"] == 0.82
    assert first["dense_source"] == "m2d2_aura"
    assert "不得声称没有排序逻辑" in prompt


def test_clarification_fallback_preserves_planner_question(monkeypatch):
    monkeypatch.setattr(
        conversation_runtime,
        "_request_prose",
        lambda *_args: (_ for _ in ()).throw(TimeoutError("cold")),
    )

    text, status = conversation_runtime.general_chat(
        "跟刚才那首差不多的",
        planner_decision={
            "task_mode": "recommendation",
            "response_mode": "clarify",
            "clarification": "你指的是当前正在播放的歌曲吗？",
        },
    )

    assert text == "你指的是当前正在播放的歌曲吗？"
    assert status == "自然语言安全回退（TimeoutError）"


def test_space_app_uses_one_orchestrated_conversation_surface():
    source = conversation_runtime.__file__.replace("conversation_runtime.py", "app.py")
    content = open(source, encoding="utf-8").read()
    assert "from conversation_runtime import general_chat, recommendation_opening" in content
    assert "opening, conversation_status = recommendation_opening(" in content
    assert "def unified_turn(" in content
    assert 'api_name="conversation"' in content
    assert "classify_turn" not in content
    unified = content[content.index("def unified_turn(") : content.index("def recommend(")]
    assert unified.index("plan_request(") < unified.index("planner_turn_kind(plan)")
    assert "selected_song_id" in unified
    assert "current_playlist_rows=list(previous_rows or [])" in unified
    assert "novel_result_window(" in content
    assert "history_state = gr.State([])" in content
    assert 'storage_key="soultuner-history-v2"' not in content
    assert "outputs=conversation_outputs" in content
