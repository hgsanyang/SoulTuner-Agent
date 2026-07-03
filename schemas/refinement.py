"""Optional refinement chips for soft music requests.

Clarification is blocking: the agent should ask before recommending when the
reference cannot be resolved.  Refinement is non-blocking: the agent can still
recommend, but offers likely follow-up buttons when the request is soft enough
that ranking may drift.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from schemas.dialog_state import DialogMusicState, load_dialog_state
from schemas.query_plan import MusicQueryPlan


class RefinementOption(BaseModel):
    """A single non-blocking follow-up chip."""

    label: str
    prompt: str
    reason: str = ""
    source: str = "context"


class RefinementSuggestion(BaseModel):
    """Soft-confidence report for the current turn."""

    confidence: float = 1.0
    reason: str = ""
    options: list[RefinementOption] = Field(default_factory=list)


SOFT_AMBIGUITY_CUES = (
    "感觉",
    "氛围",
    "一点",
    "一些",
    "适合",
    "那种",
    "这种",
    "warm",
    "rainy",
    "sunday",
    "afternoon",
    "vibe",
    "mood",
    "beats",
    "lo-fi",
    "lofi",
    "chill",
    "soft",
)


def _norm(text: Any) -> str:
    return str(text or "").strip().casefold()


def _has(text: str, pattern: str) -> bool:
    return bool(re.search(pattern, text, flags=re.IGNORECASE))


def _add_unique(
    options: list[RefinementOption],
    *,
    label: str,
    prompt: str,
    reason: str,
    source: str = "context",
) -> None:
    if any(opt.label == label or opt.prompt == prompt for opt in options):
        return
    options.append(RefinementOption(label=label, prompt=prompt, reason=reason, source=source))


def _profile_options(profile_text: str, options: list[RefinementOption]) -> None:
    profile = _norm(profile_text)
    if "独立" in profile or "indie" in profile:
        _add_unique(
            options,
            label="偏独立一点",
            prompt="偏独立一点，少一点主流流行感",
            reason="来自用户画像里的独立音乐偏好",
            source="profile",
        )
    if "民谣" in profile or "folk" in profile:
        _add_unique(
            options,
            label="偏民谣一点",
            prompt="偏民谣一点，保留自然温暖的质感",
            reason="来自用户画像里的民谣偏好",
            source="profile",
        )
    if "治愈" in profile or "healing" in profile:
        _add_unique(
            options,
            label="更治愈",
            prompt="更治愈一点，但不要太苦情",
            reason="来自用户画像里的治愈偏好",
            source="profile",
        )
    if "怀旧" in profile or "nostalg" in profile:
        _add_unique(
            options,
            label="更怀旧",
            prompt="更怀旧一点，像旧回忆但不要太沉重",
            reason="来自用户画像里的怀旧偏好",
            source="profile",
        )


def _fallback_options(options: list[RefinementOption]) -> None:
    _add_unique(
        options,
        label="更安静",
        prompt="更安静一点，动态收住",
        reason="通用偏好微调",
    )
    _add_unique(
        options,
        label="更有节奏",
        prompt="更有节奏一点，能量往上提",
        reason="通用偏好微调",
    )
    _add_unique(
        options,
        label="更小众",
        prompt="更小众一点，少一点热门榜单感",
        reason="通用偏好微调",
    )
    _add_unique(
        options,
        label="少人声",
        prompt="少人声一点，更适合专注听",
        reason="通用偏好微调",
    )
    _add_unique(
        options,
        label="低动态",
        prompt="低动态一点，不要鼓和人声太顶",
        reason="通用偏好微调",
    )
    _add_unique(
        options,
        label="偏抒情",
        prompt="偏抒情一点，让旋律和情绪更突出",
        reason="通用偏好微调",
    )
    _add_unique(
        options,
        label="更明亮",
        prompt="更明亮一点，别太低落",
        reason="通用偏好微调",
    )


def build_refinement_suggestions(
    *,
    user_input: str,
    plan: MusicQueryPlan | None,
    dialog_state: DialogMusicState | dict[str, Any] | None = None,
    user_profile: str = "",
    max_options: int = 8,
) -> RefinementSuggestion:
    """Return non-blocking refinement chips plus a conservative confidence.

    The confidence is not a model probability.  It is an operational signal:
    high means "route and constraints are concrete"; medium means "recommend
    now, but chips may help"; low should be reserved for blocking
    clarification, which is handled elsewhere.
    """
    if plan is None or plan.intent_type in {"general_chat", "acquire_music", "recommend_by_favorites"}:
        return RefinementSuggestion()

    state = load_dialog_state(dialog_state)
    rp = plan.retrieval_plan
    hard = rp.hard_constraints
    soft = rp.soft_intent
    hints = rp.hints

    text_parts = [
        user_input,
        soft.goal,
        soft.trajectory,
        soft.vibe,
        " ".join(soft.avoid),
        " ".join(hints.genres),
        hints.mood or "",
        hints.scenario or "",
        state.soft_intent.vibe,
        " ".join(state.hints.genres),
        state.hints.mood or "",
        state.hints.scenario or "",
    ]
    text = _norm(" ".join(text_parts))

    has_soft_signal = bool(soft.goal or soft.trajectory or soft.vibe or soft.avoid or hints.genres or hints.mood or hints.scenario)
    soft_cue_count = sum(1 for cue in SOFT_AMBIGUITY_CUES if cue in text)
    should_offer = has_soft_signal or soft_cue_count >= 1 or plan.intent_type in {
        "vector_search",
        "hybrid_search",
        "graph_search",
    }

    options: list[RefinementOption] = []
    if _has(text, r"lo[- ]?fi|lofi|beats?|chill"):
        _add_unique(options, label="更安静", prompt="更安静一点，保留松弛感", reason="lo-fi/chill 需求容易在安静度上漂移")
    if _has(text, r"rain|雨|灰灰|窗外|sunday|afternoon"):
        _add_unique(options, label="更有雨天感", prompt="更有雨天感一点，像雨天下午的室内氛围", reason="强化雨天/下午的场景意象")
        _add_unique(options, label="保留雨天感", prompt="保留雨天感，但换一批更贴近当下心情的歌", reason="支持多轮保留场景")
    if _has(text, r"lo[- ]?fi|lofi|beats?|chill"):
        _add_unique(options, label="更偏 lo-fi beat", prompt="更偏 lo-fi beat，一点节拍但别太吵", reason="强化 lo-fi beat 这个声学锚点")
        _add_unique(options, label="少人声", prompt="少人声一点，更适合专注听", reason="lo-fi 背景听感通常需要降低人声干扰")
    if _has(text, r"人声少|无人声|写代码|focus|coding|study|看书|阅读|sleep|睡|quiet|calm|安静"):
        _add_unique(options, label="少人声", prompt="少人声一点，更适合专注听", reason="专注/安静场景通常需要降低人声干扰")
    if _has(text, r"低动态|少鼓|不刺耳|不吵|quiet|soft|gentle|low dynamic|not loud|not noisy"):
        _add_unique(options, label="低动态", prompt="低动态一点，不要鼓和人声太顶", reason="强化低动态/不刺耳")
        _add_unique(options, label="更不刺耳", prompt="更不刺耳一点，高频和鼓都收住", reason="安静场景需要控制刺激感")
    if _has(text, r"warm|温暖|治愈|gentle|soft"):
        _add_unique(options, label="更温暖", prompt="更温暖治愈一点，但别太苦情", reason="强化温暖治愈取向")
    if _has(text, r"upbeat|uplift|振作|有精神|跑步|running|gym|运动|节奏"):
        _add_unique(options, label="更有节奏", prompt="更有节奏一点，能量再往上提", reason="运动/振作类请求常需要能量微调")
    if _has(text, r"emo|sad|悲伤|难过|别太丧|not sad|less sad"):
        _add_unique(options, label="少一点悲伤", prompt="少一点悲伤，往被托住和释怀的方向走", reason="悲伤相关请求容易过度下坠")
    if _has(text, r"小众|别太烂大街|niche|not mainstream|mainstream"):
        _add_unique(options, label="更小众", prompt="更小众一点，少一点热门榜单感", reason="强化非主流/小众约束")
    if _has(text, r"女声|female vocal|female vocals|vocal"):
        _add_unique(options, label="女声更突出", prompt="女声更突出一点，氛围保持轻盈", reason="强化人声性别和质感")

    _profile_options(user_profile, options)

    if should_offer and len(options) < 5:
        _fallback_options(options)

    confidence = 0.86
    reason = "concrete_constraints"
    if has_soft_signal or soft_cue_count:
        confidence = 0.66
        reason = "soft_intent_open_ended"
    if soft_cue_count >= 3:
        confidence = 0.58
    if state.last_delta.followup:
        confidence = max(confidence, 0.72)
    if hard.language or hints.genres:
        confidence = max(confidence, 0.7)

    return RefinementSuggestion(
        confidence=round(confidence, 2),
        reason=reason,
        options=options[: max(0, max_options)],
    )
