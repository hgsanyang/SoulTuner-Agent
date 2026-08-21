from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SPACE = Path(__file__).resolve().parents[2] / "deploy" / "modelscope_space"
sys.path.insert(0, str(SPACE))
SPEC = importlib.util.spec_from_file_location("space_conversation_ui", SPACE / "conversation_ui.py")
assert SPEC and SPEC.loader
ui = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ui)


def test_multi_turn_chat_appends_base_model_reply_and_keeps_recent_history(monkeypatch) -> None:
    captured = {}

    def fake_chat(message, history, memory):
        captured.update(message=message, history=history, memory=memory)
        return "可以，我们从雨声和空间感聊起。", "35B 基座自然语言已就绪"

    monkeypatch.setattr(ui, "general_chat", fake_chat)
    old_history = [
        {"role": "user", "content": f"old-{index}"}
        for index in range(20)
    ]
    cleared, updated, status = ui.continue_general_chat(
        "为什么雨天适合 ambient？",
        old_history,
        {"positive_tags": {"ambient": 2}},
    )

    assert cleared == ""
    assert captured["message"] == "为什么雨天适合 ambient？"
    assert captured["history"] == old_history
    assert updated[-2:] == [
        {"role": "user", "content": "为什么雨天适合 ambient？"},
        {"role": "assistant", "content": "可以，我们从雨声和空间感聊起。"},
    ]
    assert len(updated) == 16
    assert status == "自然对话：35B 基座自然语言已就绪"


def test_multi_turn_chat_exposes_safe_fallback_status(monkeypatch) -> None:
    monkeypatch.setattr(
        ui,
        "general_chat",
        lambda *_args: ("我在，你可以继续说说此刻的心情。", "自然语言安全回退（TimeoutError）"),
    )

    _, updated, status = ui.continue_general_chat("陪我聊聊", [], {})

    assert updated[-1]["content"].startswith("我在")
    assert "安全回退" in status


def test_empty_and_reset_chat_do_not_call_model(monkeypatch) -> None:
    monkeypatch.setattr(
        ui,
        "general_chat",
        lambda *_args: (_ for _ in ()).throw(AssertionError("model should not be called")),
    )

    assert ui.continue_general_chat("  ", [], {}) == ("", [], "自然对话：请输入一条消息。")
    assert ui.reset_general_chat() == ([], "自然对话：等待消息。")

