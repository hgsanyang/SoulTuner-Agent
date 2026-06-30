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
    "他的歌",
    "她的歌",
    "他们的歌",
    "那首歌",
    "那个歌手",
    "same vibe",
    "that vibe",
    "similar",
    "like that",
)

PRIVATE_MEMORY_REFERENCE_CUES = (
    "上个月一直循环",
    "之前一直循环",
    "我以前说不喜欢",
    "后来又说不喜欢",
    "我上次说的那首",
    "last month",
    "i used to loop",
    "i said i disliked",
)


class ClarificationRequest(BaseModel):
    """A high-precision clarification response."""

    required: bool = False
    reason: str = ""
    question: str = ""
    options: list[str] = Field(default_factory=list)


class DialogStateDelta(BaseModel):
    """Deterministic state update report for the current turn."""

    followup: bool = False
    topic_shift: bool = False
    confidence: float = 1.0
    reason: str = ""
    inherited: list[str] = Field(default_factory=list)
    added: dict[str, Any] = Field(default_factory=dict)
    replaced: dict[str, Any] = Field(default_factory=dict)
    removed: list[str] = Field(default_factory=list)


class DialogMusicState(BaseModel):
    """Session-local structured music state."""

    hard_constraints: HardConstraints = Field(default_factory=HardConstraints)
    soft_intent: SoftIntent = Field(default_factory=SoftIntent)
    hints: IntentHints = Field(default_factory=IntentHints)
    last_intent_type: str = ""
    last_query: str = ""
    last_vector_acoustic_query: str = ""
    last_delta: DialogStateDelta = Field(default_factory=DialogStateDelta)
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


def _non_empty(value: Any) -> bool:
    if isinstance(value, list):
        return bool(value)
    return value not in (None, "", False, [])


def _state_value_map(state: DialogMusicState) -> dict[str, Any]:
    hard = state.hard_constraints
    soft = state.soft_intent
    hints = state.hints
    return {
        "hard_constraints.artist_entities": list(hard.artist_entities),
        "hard_constraints.song_entities": list(hard.song_entities),
        "hard_constraints.language": hard.language,
        "hard_constraints.region": hard.region,
        "hard_constraints.instrumental": hard.instrumental,
        "soft_intent.goal": soft.goal,
        "soft_intent.trajectory": soft.trajectory,
        "soft_intent.avoid": list(soft.avoid),
        "soft_intent.vibe": soft.vibe,
        "hints.genres": list(hints.genres),
        "hints.mood": hints.mood,
        "hints.scenario": hints.scenario,
    }


def _plan_value_map(plan: MusicQueryPlan) -> dict[str, Any]:
    return _state_value_map(
        DialogMusicState(
            hard_constraints=plan.retrieval_plan.hard_constraints,
            soft_intent=plan.retrieval_plan.soft_intent,
            hints=plan.retrieval_plan.hints,
        )
    )


def _build_delta_report(
    previous: DialogMusicState,
    plan: MusicQueryPlan,
    updated: DialogMusicState,
    *,
    followup: bool,
    topic_shift: bool,
    reason: str,
) -> DialogStateDelta:
    prev_values = _state_value_map(previous)
    plan_values = _plan_value_map(plan)
    new_values = _state_value_map(updated)

    inherited: list[str] = []
    added: dict[str, Any] = {}
    replaced: dict[str, Any] = {}
    removed: list[str] = []

    for key, new_value in new_values.items():
        prev_value = prev_values.get(key)
        plan_value = plan_values.get(key)
        if followup and _non_empty(prev_value) and new_value == prev_value and not _non_empty(plan_value):
            inherited.append(key)
        elif not _non_empty(prev_value) and _non_empty(new_value):
            added[key] = new_value
        elif _non_empty(prev_value) and _non_empty(new_value) and new_value != prev_value:
            replaced[key] = new_value
        elif _non_empty(prev_value) and not _non_empty(new_value):
            removed.append(key)

    return DialogStateDelta(
        followup=followup,
        topic_shift=topic_shift,
        confidence=0.8 if followup and inherited else 1.0,
        reason=reason,
        inherited=inherited,
        added=added,
        replaced=replaced,
        removed=removed,
    )


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
    has_state = state.turn_count > 0
    if _has_any(user_input, PRIVATE_MEMORY_REFERENCE_CUES) and not has_state:
        return ClarificationRequest(
            required=True,
            reason="private_memory_reference_without_state",
            question="我现在没有足够可靠的历史记录来判断你说的是哪一首。你可以告诉我歌名、歌手，或描述一下那首歌的感觉吗？",
            options=[
                "告诉我歌名或歌手",
                "描述那首歌的氛围",
                "先按相近感觉推荐",
            ],
        )
    if has_state:
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
        if re.search(r"工作|写代码|coding|focus|专注", text):
            hints.scenario = "工作"
            extracted = True
        if re.search(r"放松|relax|calm", text):
            hints.mood = "放松"
            extracted = True
        if re.search(r"治愈|healing", text):
            hints.mood = "治愈"
            soft.vibe = soft.vibe or "healing, gentle, warm"
            extracted = True
        if re.search(r"悲伤|难过|想哭|sad|cry", text):
            hints.mood = "悲伤"
            soft.vibe = soft.vibe or "sad, melancholy, gentle"
            extracted = True
        if re.search(r"钢琴|piano", text):
            hints.genres = _merge_unique(hints.genres, ["piano"])
            extracted = True
        if re.search(r"流行|pop", text):
            hints.genres = _merge_unique(hints.genres, ["pop"])
            extracted = True
        if re.search(r"摇滚|rock", text):
            hints.genres = _merge_unique(hints.genres, ["rock"])
            extracted = True
        if re.search(r"安静|quiet|低动态", text):
            soft.vibe = soft.vibe or "quiet, low energy"
            extracted = True
        artist_match = re.search(r"([\w\u4e00-\u9fff·・\-. ]{2,32})的歌", text)
        if artist_match:
            artist = str(artist_match.group(1)).strip(" ，,。.!！")
            if artist and not re.search(r"我|你|他|她|他们|她们|大家|一些|几首", artist):
                hard.artist_entities = _merge_unique(hard.artist_entities, [artist])
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


def apply_plan_delta_with_report(
    previous: DialogMusicState | dict[str, Any] | None,
    plan: MusicQueryPlan,
    user_input: str,
) -> tuple[DialogMusicState, DialogStateDelta]:
    """Apply the current plan as a deterministic delta over prior state."""
    prev = load_dialog_state(previous)
    rp = plan.retrieval_plan
    topic_shift = _looks_like_topic_shift(user_input, plan)
    followup = _has_any(user_input, FOLLOWUP_CUES) and not topic_shift

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
    if re.search(r"人声再少|人声少|别打扰|不打扰|写代码|focus|coding", text):
        soft.vibe = soft.vibe or "unobtrusive, sparse vocals, focus-friendly"
        soft.avoid = _merge_unique(soft.avoid, ["prominent vocals", "distracting vocals"])
    if re.search(r"别.*(苦情|抒情)|不要.*(苦情|抒情)|not.*sad ballad|not the sad ballads", text):
        soft.avoid = _merge_unique(soft.avoid, ["sad ballad", "melancholy ballad", "苦情", "抒情大歌"])
    if re.search(r"不要.*伤|别.*伤|别太丧|not sad|less sad", text):
        soft.avoid = _merge_unique(soft.avoid, ["sad", "melancholy", "悲伤", "丧"])
    if re.search(r"拉起来|振作|有精神|upbeat|uplift", text):
        soft.vibe = soft.vibe or "uplifting, hopeful, brighter energy"
    if re.search(r"危险感|danger|dark", text):
        soft.vibe = soft.vibe or "dark, tense, dangerous"
        soft.avoid = _merge_unique(soft.avoid, ["healing", "gentle", "治愈"])

    updated = DialogMusicState(
        hard_constraints=hard,
        soft_intent=soft,
        hints=hints,
        last_intent_type=plan.intent_type,
        last_query=user_input,
        last_vector_acoustic_query=rp.vector_acoustic_query or base.last_vector_acoustic_query,
        turn_count=prev.turn_count + 1,
    )
    delta = _build_delta_report(
        prev,
        plan,
        updated,
        followup=followup,
        topic_shift=topic_shift,
        reason="followup_delta" if followup else "new_topic_delta",
    )
    updated.last_delta = delta
    return updated, delta


def apply_plan_delta(
    previous: DialogMusicState | dict[str, Any] | None,
    plan: MusicQueryPlan,
    user_input: str,
) -> DialogMusicState:
    """Backward-compatible wrapper returning only the updated state."""
    updated, _delta = apply_plan_delta_with_report(previous, plan, user_input)
    return updated


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


def has_retrievable_dialog_state(dialog_state: DialogMusicState | dict[str, Any] | None) -> bool:
    """Return True when a stored state can safely drive a music retrieval turn."""
    state = load_dialog_state(dialog_state)
    hard = state.hard_constraints
    soft = state.soft_intent
    hints = state.hints
    return any(
        [
            hard.artist_entities,
            hard.song_entities,
            hard.language,
            hard.region,
            hard.instrumental,
            soft.goal,
            soft.trajectory,
            soft.avoid,
            soft.vibe,
            hints.genres,
            hints.mood,
            hints.scenario,
            state.last_vector_acoustic_query,
        ]
    )


def coerce_followup_general_chat_to_retrieval(
    plan: MusicQueryPlan,
    dialog_state: DialogMusicState | dict[str, Any] | None,
    user_input: str,
) -> MusicQueryPlan:
    """Keep resolved follow-up turns on the recommendation path.

    LLM planners sometimes classify short follow-ups such as "same vibe, but
    Chinese" as general_chat because the current text does not explicitly say
    "recommend songs".  Once DST has resolved that follow-up into concrete
    music state, routing to chat would drop valid constraints and return no
    songs.  This guardrail is intentionally narrow: it only fires for a
    general_chat plan, a follow-up cue, and an already retrievable state.
    """
    state = load_dialog_state(dialog_state)
    if plan.intent_type != "general_chat":
        return plan
    if not (state.last_delta.followup or _has_any(user_input, FOLLOWUP_CUES)):
        return plan
    if not has_retrievable_dialog_state(state):
        return plan

    updated = plan.model_copy(deep=True)
    updated.intent_type = "hybrid_search"
    updated.parameters = {
        "query": user_input,
        "entities": _merge_unique(
            state.hard_constraints.artist_entities,
            state.hard_constraints.song_entities,
        ),
    }
    updated.context = updated.context or "基于对话状态继续推荐"
    updated.reasoning = "DST 已解析多轮继承，闲聊判定收束为混合检索"

    rp = updated.retrieval_plan
    rp.use_graph = bool(
        state.hard_constraints.artist_entities
        or state.hard_constraints.song_entities
        or state.hard_constraints.language
        or state.hard_constraints.region
        or state.hard_constraints.instrumental
        or state.hints.genres
        or state.hints.mood
        or state.hints.scenario
    )
    rp.use_vector = True
    rp.use_web_search = False
    if not rp.vector_acoustic_query:
        vector_parts: list[str] = []
        for part in [
            state.soft_intent.goal,
            state.soft_intent.trajectory,
            state.soft_intent.vibe,
            ", ".join(state.soft_intent.avoid),
            ", ".join(state.hints.genres),
            state.hints.mood,
            state.hints.scenario,
            state.last_vector_acoustic_query,
            user_input,
        ]:
            value = str(part or "").strip()
            if value and value not in vector_parts:
                vector_parts.append(value)
        rp.vector_acoustic_query = " ; ".join(vector_parts) or user_input

    updated.retrieval_plan = type(rp).model_validate(rp.model_dump())
    return updated
