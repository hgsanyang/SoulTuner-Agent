"""Small UI adapter for the base-model multi-turn Gradio chat."""

from __future__ import annotations

from typing import Any

from conversation_runtime import general_chat


def continue_general_chat(
    message: str,
    history: list[dict[str, str]] | None,
    memory: dict[str, Any] | None,
) -> tuple[str, list[dict[str, str]], str]:
    clean = str(message or "").strip()
    current = [dict(item) for item in (history or []) if isinstance(item, dict)]
    if not clean:
        return "", current, "自然对话：请输入一条消息。"

    reply, status = general_chat(clean, current, memory)
    updated = [
        *current,
        {"role": "user", "content": clean},
        {"role": "assistant", "content": reply},
    ]
    # Bound browser state as well as the prompt passed by conversation_runtime.
    return "", updated[-16:], f"自然对话：{status}"


def reset_general_chat() -> tuple[list[dict[str, str]], str]:
    return [], "自然对话：等待消息。"

