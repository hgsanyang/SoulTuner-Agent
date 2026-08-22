"""Deterministic turn routing for the unified public conversation surface.

The base 35B model writes prose and the SoulTuner LoRA writes guarded plans.
This module only decides which role should receive the current turn and carries
the last recommendation request into short refinement follow-ups.
"""

from __future__ import annotations

import re
from typing import Any


_EXPLANATION_PREFIXES = (
    "为什么",
    "怎么理解",
    "什么是",
    "解释",
    "聊聊",
    "你觉得",
    "区别",
)
_RECOMMENDATION_ACTIONS = (
    "推荐",
    "来点",
    "来首",
    "听点",
    "想听",
    "找歌",
    "找一",
    "歌单",
    "播放",
    "换一组",
    "再来",
    "类似的歌",
)
_REFINEMENT_MARKERS = (
    "刚才",
    "刚刚",
    "上一组",
    "这组",
    "同样",
    "再",
    "更",
    "换成",
    "不要",
    "少一点",
    "多一点",
    "保持",
)
_MUSIC_CUES = (
    "摇滚",
    "流行",
    "电子",
    "爵士",
    "民谣",
    "说唱",
    "嘻哈",
    "朋克",
    "金属",
    "古典",
    "氛围",
    "轻音乐",
    "钢琴",
    "吉他",
    "人声",
    "纯音乐",
    "夜跑",
    "通勤",
    "旅行",
    "学习",
    "睡前",
)


def _contains(text: str, values: tuple[str, ...]) -> bool:
    lowered = text.casefold()
    return any(value.casefold() in lowered for value in values)


def classify_turn(message: str, last_recommendation_query: str = "") -> str:
    """Return ``recommendation`` or ``conversation`` for one visible turn."""

    clean = str(message or "").strip()
    if not clean:
        return "conversation"
    if clean.startswith(_EXPLANATION_PREFIXES) and not _contains(clean, ("推荐", "找歌", "来点", "来首")):
        return "conversation"
    if _contains(clean, _RECOMMENDATION_ACTIONS):
        return "recommendation"
    if last_recommendation_query and _contains(clean, _REFINEMENT_MARKERS):
        return "recommendation"
    # A compact genre/scene utterance such as "公路旅行" or "英伦摇滚" is
    # normally a discovery request.  Longer explanatory sentences stay chat.
    if len(clean) <= 18 and _contains(clean, _MUSIC_CUES):
        return "recommendation"
    if re.search(r"(?:适合|用于).{0,12}(?:听|歌|音乐)$", clean):
        return "recommendation"
    return "conversation"


def contextualize_recommendation(message: str, last_recommendation_query: str = "") -> str:
    """Resolve a short refinement against the last recommendation request."""

    clean = str(message or "").strip()
    previous = str(last_recommendation_query or "").strip()
    if previous and _contains(clean, _REFINEMENT_MARKERS):
        return f"{previous}\n[本轮调整] {clean}"
    return clean


def bounded_history(history: list[dict[str, Any]] | None, *, limit: int = 20) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for item in history or []:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "")
        content = str(item.get("content") or "").strip()
        if role in {"user", "assistant"} and content:
            rows.append({"role": role, "content": content[:1200]})
    return rows[-limit:]


def append_turn(
    history: list[dict[str, Any]] | None,
    user_message: str,
    assistant_message: str,
) -> list[dict[str, str]]:
    return bounded_history(
        [
            *bounded_history(history),
            {"role": "user", "content": str(user_message or "").strip()},
            {"role": "assistant", "content": str(assistant_message or "").strip()},
        ]
    )


def history_for_planner(history: list[dict[str, Any]] | None) -> str:
    labels = {"user": "用户", "assistant": "SoulTuner"}
    return "\n".join(f"{labels[item['role']]}：{item['content']}" for item in bounded_history(history, limit=8))
