"""Natural-language role for the single-instance 35B ModelScope demo.

The base Qwen served model writes user-facing prose.  The SoulTuner LoRA model
remains planner-only and is never called from this module.  Both model names may
be exposed by the same OpenAI-compatible vLLM process.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from typing import Any


DEFAULT_CHAT_MODEL = "qwen3.6-35b-a3b"
DEFAULT_PLANNER_MODEL = "soultuner-v4.2-35b"

_PROSE_SYSTEM = """你是 SoulTuner 的自然语言对话助手。你只负责用户可见的自然语言，不能输出规划 JSON，不能假装执行检索，也不能编造歌曲、用户记忆或播放状态。结构化检索计划已经由独立的 SoulTuner 35B Planner LoRA 生成并通过守卫。回答简洁、自然，像懂音乐的朋友。"""


def _models() -> tuple[str, str]:
    chat_model = os.getenv("SOULTUNER_CHAT_MODEL", DEFAULT_CHAT_MODEL).strip()
    planner_model = os.getenv("SOULTUNER_PLANNER_MODEL", DEFAULT_PLANNER_MODEL).strip()
    if not chat_model:
        raise RuntimeError("SOULTUNER_CHAT_MODEL is empty")
    normalized_chat = chat_model.casefold().replace("_", "-")
    planner_named_chat = "soultuner" in normalized_chat and ("v4.2" in normalized_chat or "v42" in normalized_chat)
    if chat_model.casefold() == planner_model.casefold() or planner_named_chat:
        raise RuntimeError("chat served model must differ from planner-only LoRA")
    return chat_model, planner_model


def _endpoint() -> tuple[str, str]:
    chat_model, _ = _models()
    base = os.getenv("SOULTUNER_CHAT_BASE_URL", "").strip() or os.getenv("SOULTUNER_PLANNER_BASE_URL", "").strip()
    if not base:
        # CPU demo mode intentionally has no model server.  Fail immediately so
        # a recommendation never waits on an endpoint that was not launched.
        raise RuntimeError("35B base chat endpoint is not configured")
    base = base.rstrip("/")
    endpoint = base if base.endswith("/chat/completions") else f"{base}/chat/completions"
    return endpoint, chat_model


def _clean_prose(value: Any) -> str:
    text = re.sub(r"<think>.*?</think>", "", str(value or ""), flags=re.DOTALL)
    text = " ".join(text.strip().split())
    if not text:
        raise ValueError("conversation model returned empty text")
    if text.startswith("{") or text.startswith("["):
        raise ValueError("conversation role returned structured planner output")
    return text[:500]


def _request_prose(user_content: str) -> str:
    endpoint, chat_model = _endpoint()
    payload = json.dumps(
        {
            "model": chat_model,
            "messages": [
                {"role": "system", "content": _PROSE_SYSTEM},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.6,
            "max_tokens": 240,
            "enable_thinking": False,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        ensure_ascii=False,
    ).encode("utf-8")
    token = os.getenv("SOULTUNER_CHAT_API_KEY", "").strip() or os.getenv("SOULTUNER_PLANNER_API_KEY", "").strip()
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(endpoint, data=payload, headers=headers, method="POST")
    timeout = float(os.getenv("SOULTUNER_CHAT_TIMEOUT", "45"))
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    return _clean_prose(body["choices"][0]["message"]["content"])


def _memory_summary(memory: dict[str, Any] | None) -> str:
    data = memory or {}
    positives = data.get("positive_tags") or {}
    negatives = data.get("negative_tags") or {}
    if not isinstance(positives, dict):
        positives = {}
    if not isinstance(negatives, dict):
        negatives = {}

    def weight(mapping: dict[str, Any], key: str) -> int:
        try:
            return int(mapping[key])
        except (TypeError, ValueError):
            return 0

    liked = sorted(positives, key=lambda key: (-weight(positives, key), str(key)))[:6]
    avoided = sorted(negatives, key=lambda key: (-weight(negatives, key), str(key)))[:6]
    last_results = []
    for item in (data.get("last_recommendations") or [])[:8]:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        artist = str(item.get("artist") or "").strip()
        if title:
            last_results.append({"title": title, "artist": artist})
    return json.dumps(
        {
            "session_likes": liked,
            "session_avoids": avoided,
            "last_recommendation_query": str(data.get("last_recommendation_query") or "")[:300],
            "last_recommendations": last_results,
        },
        ensure_ascii=False,
    )


def _fallback_opening(query: str, rows: list[dict[str, Any]]) -> str:
    names = [f"《{row.get('title')}》" for row in rows[:3] if str(row.get("title") or "").strip()]
    if names:
        return f"我按“{query}”整理了这组结果，可以先从{'、'.join(names)}开始试听，再告诉我哪种感觉更接近。"
    return f"我已经按“{query}”整理检索方向；当前没有可靠候选，可以换一种氛围、场景或参考歌曲再试。"


def recommendation_opening(
    query: str,
    plan: dict[str, Any],
    rows: list[dict[str, Any]],
    memory: dict[str, Any] | None = None,
) -> tuple[str, str]:
    """Return a natural recommendation opening plus an observable role status."""

    compact_rows = [
        {
            "title": row.get("title"),
            "artist": row.get("artist"),
            "tags": list(row.get("tags") or [])[:4],
        }
        for row in rows[:5]
    ]
    prompt = (
        "请根据已经完成的检索结果，写一段不超过100字的推荐开场。"
        "不要复述系统流程，不要增加列表外的歌曲；可以自然邀请用户试听后反馈。\n"
        + json.dumps(
            {
                "current_request": query,
                "planner_brief_reason": (plan.get("evidence") or {}).get("brief_reason"),
                "results": compact_rows,
                "memory": json.loads(_memory_summary(memory)),
            },
            ensure_ascii=False,
        )
    )
    try:
        return _request_prose(prompt), "35B 基座自然语言已就绪"
    except Exception as exc:
        return _fallback_opening(query, rows), f"自然语言安全回退（{type(exc).__name__}）"


def general_chat(
    message: str,
    chat_history: list[dict[str, str]] | None = None,
    memory: dict[str, Any] | None = None,
    planner_decision: dict[str, Any] | None = None,
    evidence_rows: list[dict[str, Any]] | None = None,
    current_playlist_rows: list[dict[str, Any]] | None = None,
    selected_song_id: str | None = None,
) -> tuple[str, str]:
    """Render a Planner-owned dialogue decision as natural prose.

    The conversation model never decides whether retrieval should run.  It is
    given the validated Planner decision and, for information turns, bounded
    retrieved evidence.  This mirrors the production Agent's role boundary.
    """

    history = [
        {"role": str(item.get("role") or ""), "content": str(item.get("content") or "")[:500]}
        for item in (chat_history or [])[-8:]
        if isinstance(item, dict)
    ]
    current_plan = planner_decision or {}
    compact_evidence = [
        {
            "title": row.get("title"),
            "artist": row.get("artist"),
            "tags": list(row.get("tags") or [])[:5],
            "reason": row.get("reason"),
        }
        for row in (evidence_rows or [])[:5]
    ]
    selected = str(selected_song_id or "").strip()
    compact_playlist = [
        {
            "rank": index,
            "is_currently_playing": bool(
                selected and str(row.get("song_id") or "").strip() == selected
            ),
            "title": row.get("title"),
            "artist": row.get("artist"),
            "tags": list(row.get("tags") or [])[:5],
            "final_score": row.get("final_score"),
            "graph_score": row.get("graph_score"),
            "dense_score": row.get("dense_score"),
            "dense_source": row.get("dense_source"),
            "reason": row.get("reason"),
        }
        for index, row in enumerate((current_playlist_rows or [])[:12], start=1)
    ]
    prompt = (
        "请自然回应用户当前消息；如果需要追问，一次只问一个问题。"
        "你必须服从已验证的 Planner 决策，不能自行把 dialogue 改成 recommendation，"
        "也不能增加证据列表外的歌曲。response_mode=clarify 时保留 Planner 的澄清目标；"
        "dialogue_mode=information 时只根据 supplied_evidence 回答。"
        "current_playlist 按真实展示顺序排列；用户询问当前歌曲、推荐理由或排序时，"
        "只能引用其中的 rank、分数、标签、reason 与 currently-playing 标记。"
        "不得声称没有排序逻辑、随机排序或为了丰富列表，也不得虚构旋律抓耳等未给出的特征；"
        "证据不足就明确说哪些部分无法从现有证据判断。\n"
        + json.dumps(
            {
                "history": history,
                "session_memory": json.loads(_memory_summary(memory)),
                "current_message": str(message or "").strip(),
                "planner_decision": {
                    "task_mode": current_plan.get("task_mode"),
                    "dialogue_mode": current_plan.get("dialogue_mode"),
                    "response_mode": current_plan.get("response_mode"),
                    "brief_reason": (current_plan.get("evidence") or {}).get("brief_reason"),
                    "clarification": current_plan.get("clarification"),
                },
                "current_playlist": compact_playlist,
                "supplied_evidence": compact_evidence,
            },
            ensure_ascii=False,
        )
    )
    try:
        return _request_prose(prompt), "35B 基座自然语言已就绪"
    except Exception as exc:
        clarification = str(current_plan.get("clarification") or "").strip()
        if clarification:
            return clarification, f"自然语言安全回退（{type(exc).__name__}）"
        return (
            "我在。你可以继续说说现在的心情、场景，或者哪种音乐感觉最重要；需要找歌时我会先交给 Planner 做受控规划。",
            f"自然语言安全回退（{type(exc).__name__}）",
        )
