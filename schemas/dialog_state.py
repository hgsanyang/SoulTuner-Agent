"""Structured dialogue state tracking for multi-turn music intent.

The LLM still creates the current turn ``MusicQueryPlan``.  This module keeps
cross-turn inheritance deterministic by applying that plan as a delta to a
small, explicit state object.  It also contains conservative clarification
triggers for follow-up queries that cannot be resolved safely.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from schemas.query_plan import HardConstraints, IntentHints, MusicQueryPlan, SoftIntent


FOLLOWUP_CUES = (
    "类似",
    "同样",
    "这种",
    "那种",
    "这个",
    "那个",
    "刚才",
    "上一首",
    "前面",
    "换成",
    "再",
    "更",
    "不要这种",
    "similar",
    "same vibe",
    "that vibe",
    "like that",
    "more like",
)

UNRESOLVED_REFERENCE_CUES = (
    "类似的",
    "类似听感",
    "同样的氛围",
    "同样的感觉",
    "刚才那首",
    "上一首",
    "换一个",
    "换首",
    "same vibe",
    "that vibe",
    "similar",
    "like that",
)


class ClarificationRequest(BaseModel):
    """A high-precision clarification response."""

    required: bool = False
    reason: str = ""
    question: str = ""
    options: list[str] = Field(default_factory=list)


class DialogMusicState(BaseModel):
    """Session-local structured music state."""

    hard_constraints: HardConstraints = Field(default_factory=HardConstraints)
    soft_intent: SoftIntent = Field(default_factory=SoftIntent)
    hints: IntentHints = Field(default_factory=IntentHints)
    last_intent_type: str = ""
    last_query: str = ""
    last_vector_acoustic_query: str = ""
    turn_count: int = 0


def _norm(text: Any) -> str:
    return str(text or "").strip().casefold()


def _has_any(text: str, cues: tuple[str, ...]) -> bool:
    folded = _norm(text)
    return any(_norm(cue) in folded for cue in cues)


def _merge_unique(previous: list[str], current: list[str]) -> list[str]:
    merged: list[str] = []
    for item in [*previous, *current]:
        value = str(item or "").strip()
        if value and value not in merged:
            merged.append(value)
    return merged


def _looks_like_topic_shift(user_input: str, plan: MusicQueryPlan) -> bool:
    """Return True when a turn should replace rather than inherit state."""
    hard = plan.retrieval_plan.hard_constraints
    text = _norm(user_input)
    explicit_new_entity = bool(hard.artist_entities or hard.song_entities)
    reset_words = ("换个话题", "重新开始", "新歌单", "不要管上面", "from scratch", "new topic")
    return explicit_new_entity or _has_any(text, reset_words)


def should_clarify_before_planning(
    user_input: str,
    dialog_state: DialogMusicState | dict[str, Any] | None,
) -> ClarificationRequest:
    """Ask only for unresolved references that cannot be grounded."""
    state = load_dialog_state(dialog_state)
    if state.turn_count > 0:
        return ClarificationRequest()
    if not _has_any(user_input, UNRESOLVED_REFERENCE_CUES):
        return ClarificationRequest()
    return ClarificationRequest(
        required=True,
        reason="unresolved_reference_without_state",
        question="你说的“这种/类似”我还没有上一轮可继承的音乐上下文。你想按哪一种方向继续？",
        options=[
            "按当前文字重新推荐",
            "告诉我一首参考歌",
            "描述想保留的氛围",
        ],
    )


def load_dialog_state(raw: DialogMusicState | dict[str, Any] | None) -> DialogMusicState:
    if isinstance(raw, DialogMusicState):
        return raw
    if isinstance(raw, dict) and raw:
        return DialogMusicState.model_validate(raw)
    return DialogMusicState()


def infer_dialog_state_from_history(chat_history: list[Any] | None) -> DialogMusicState:
    """Build a small deterministic seed state from legacy chat_history.

    This is a bridge for callers/eval cases that have not yet started sending
    explicit ``dialog_state``.  It intentionally extracts only high-confidence
    language and vibe hints; the current turn's LLM plan is still authoritative.
    """
    if not chat_history:
        return DialogMusicState()

    hard = HardConstraints()
    soft = SoftIntent()
    hints = IntentHints()
    user_turns = 0
    extracted = False

    for message in chat_history:
        role = getattr(message, "type", None) or getattr(message, "role", None)
        content = getattr(message, "content", None)
        if isinstance(message, dict):
            role = message.get("role", role)
            content = message.get("content", content)
        if role not in ("user", "human"):
            continue
        text = _norm(content)
        if not text:
            continue
        user_turns += 1

        if re.search(r"中文|国语|华语|mandarin|chinese", text):
            hard.language = "Chinese"
            extracted = True
        elif re.search(r"英文|英语|english", text):
            hard.language = "English"
            extracted = True
        elif re.search(r"日语|日文|japanese", text):
            hard.language = "Japanese"
            extracted = True
        elif re.search(r"韩语|韩文|korean", text):
            hard.language = "Korean"
            extracted = True
        elif re.search(r"粤语|cantonese", text):
            hard.language = "Cantonese"
            extracted = True

        vibe_parts: list[str] = []
        if re.search(r"空灵|ethereal|airy", text):
            vibe_parts.append("ethereal, airy")
        if re.search(r"女声|female vocal|female vocals", text):
            vibe_parts.append("female vocal")
        if re.search(r"梦幻|dreamy", text):
            vibe_parts.append("dreamy")
        if vibe_parts:
            soft.vibe = ", ".join(vibe_parts)
            extracted = True

        if re.search(r"运动|健身|workout|gym", text):
            hints.scenario = "运动"
            extracted = True
        if re.search(r"放松|relax|calm", text):
            hints.mood = "放松"
            extracted = True
        if re.search(r"钢琴|piano", text):
            hints.genres = _merge_unique(hints.genres, ["piano"])
            extracted = True

    if not extracted:
        return DialogMusicState()
    return DialogMusicState(
        hard_constraints=hard,
        soft_intent=soft,
        hints=hints,
        last_query="",
        turn_count=user_turns,
    )


def apply_plan_delta(
    previous: DialogMusicState | dict[str, Any] | None,
    plan: MusicQueryPlan,
    user_input: str,
) -> DialogMusicState:
    """Apply the current plan as a deterministic delta over prior state."""
    prev = load_dialog_state(previous)
    rp = plan.retrieval_plan
    followup = _has_any(user_input, FOLLOWUP_CUES) and not _looks_like_topic_shift(user_input, plan)

    base = prev if followup else DialogMusicState()
    hard = HardConstraints(
        artist_entities=_merge_unique(
            base.hard_constraints.artist_entities if followup else [],
            rp.hard_constraints.artist_entities,
        ),
        song_entities=_merge_unique(
            base.hard_constraints.song_entities if followup else [],
            rp.hard_constraints.song_entities,
        ),
        language=rp.hard_constraints.language or (base.hard_constraints.language if followup else None),
        region=rp.hard_constraints.region or (base.hard_constraints.region if followup else None),
        instrumental=rp.hard_constraints.instrumental or (base.hard_constraints.instrumental if followup else False),
    )

    soft = SoftIntent(
        goal=rp.soft_intent.goal or (base.soft_intent.goal if followup else ""),
        trajectory=rp.soft_intent.trajectory or (base.soft_intent.trajectory if followup else ""),
        avoid=_merge_unique(base.soft_intent.avoid if followup else [], rp.soft_intent.avoid),
        vibe=rp.soft_intent.vibe or (base.soft_intent.vibe if followup else ""),
    )
    hints = IntentHints(
        genres=_merge_unique(base.hints.genres if followup else [], rp.hints.genres),
        mood=rp.hints.mood or (base.hints.mood if followup else None),
        scenario=rp.hints.scenario or (base.hints.scenario if followup else None),
    )

    # Lightweight deterministic overrides for very common follow-up language.
    text = _norm(user_input)
    if re.search(r"中文|国语|华语|mandarin|chinese", text):
        hard.language = "Chinese"
    elif re.search(r"英文|英语|english", text):
        hard.language = "English"
    elif re.search(r"日语|日文|japanese", text):
        hard.language = "Japanese"
    elif re.search(r"韩语|韩文|korean", text):
        hard.language = "Korean"
    elif re.search(r"粤语|cantonese", text):
        hard.language = "Cantonese"

    if re.search(r"无人声|没人声|纯音乐|instrumental|no vocals?", text):
        hard.instrumental = True

    return DialogMusicState(
        hard_constraints=hard,
        soft_intent=soft,
        hints=hints,
        last_intent_type=plan.intent_type,
        last_query=user_input,
        last_vector_acoustic_query=rp.vector_acoustic_query or base.last_vector_acoustic_query,
        turn_count=prev.turn_count + 1,
    )


def apply_dialog_state_to_plan(plan: MusicQueryPlan, dialog_state: DialogMusicState) -> MusicQueryPlan:
    """Copy deterministic state back into the executable retrieval plan."""
    updated = plan.model_copy(deep=True)
    rp = updated.retrieval_plan
    rp.hard_constraints = dialog_state.hard_constraints.model_copy(deep=True)
    rp.soft_intent = dialog_state.soft_intent.model_copy(deep=True)
    rp.hints = dialog_state.hints.model_copy(deep=True)
    # Re-validate to sync layered fields back to legacy fields.
    updated.retrieval_plan = type(rp).model_validate(rp.model_dump())
    return updated
