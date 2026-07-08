"""Structured dialogue state storage for multi-turn music intent.

The LLM is the highest-priority interpreter of complex user intent and
cross-turn context.  This module stores the explicit state produced by the
planner, validates LLM-generated deltas, and applies them deterministically
after the model has made the semantic decision.  It must not replace the
planner with phrase-based follow-up or vibe interpretation rules.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from schemas.query_plan import HardConstraints, IntentHints, MusicQueryPlan, SoftIntent


UNRESOLVED_REFERENCE_CUES = (
    "类似的",
    "类似听感",
    "同样的氛围",
    "同样的感觉",
    "刚才那首",
    "刚才第一首",
    "第一首",
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
    unresolved_paths: list[str] = Field(default_factory=list)


ALLOWED_DELTA_PATHS = {
    "hard_constraints.artist_entities",
    "hard_constraints.song_entities",
    "hard_constraints.language",
    "hard_constraints.region",
    "hard_constraints.instrumental",
    "soft_intent.goal",
    "soft_intent.trajectory",
    "soft_intent.avoid",
    "soft_intent.vibe",
    "hints.genres",
    "hints.mood",
    "hints.scenario",
}


class DeltaOperation(BaseModel):
    """One validated mutation against the session-local music state."""

    op: Literal["add", "replace", "remove", "clear_topic"]
    path: str = ""
    value: Any = None

    @model_validator(mode="after")
    def validate_path(self) -> "DeltaOperation":
        if self.op == "clear_topic":
            self.path = ""
            return self
        if self.path not in ALLOWED_DELTA_PATHS:
            raise ValueError(f"Unsupported dialogue delta path: {self.path}")
        return self


class PlanDelta(BaseModel):
    """The only structure an LLM may produce for an established follow-up."""

    operations: list[DeltaOperation] = Field(default_factory=list)
    resolved_references: list[str] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    ambiguity_reasons: list[str] = Field(default_factory=list)
    clarification: ClarificationRequest | None = None
    planner_mode: Literal["deterministic", "delta_llm", "full_fallback"] = "delta_llm"


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
    operations: list[dict[str, Any]] = Field(default_factory=list)
    resolved_references: list[str] = Field(default_factory=list)
    ambiguity_reasons: list[str] = Field(default_factory=list)
    planner_mode: str = "full_planner"


class DialogMusicState(BaseModel):
    """Session-local structured music state."""

    hard_constraints: HardConstraints = Field(default_factory=HardConstraints)
    soft_intent: SoftIntent = Field(default_factory=SoftIntent)
    hints: IntentHints = Field(default_factory=IntentHints)
    last_intent_type: str = ""
    last_query: str = ""
    last_vector_acoustic_query: str = ""
    last_complete_plan: dict[str, Any] = Field(default_factory=dict)
    last_result_titles: list[str] = Field(default_factory=list)
    last_result_artists: list[str] = Field(default_factory=list)
    pending_clarification: ClarificationRequest | None = None
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


_NON_ARTIST_ENTITY_CUES = (
    "想听",
    "推荐",
    "来点",
    "给我",
    "一些",
    "几首",
    "适合",
    "感觉",
    "氛围",
    "那种",
    "这种",
    "同样",
    "类似",
    "保留",
    "今天",
    "晚上",
    "夜里",
    "凌晨",
    "下雨",
    "雨天",
    "通勤",
    "路上",
    "散步",
    "开车",
    "写代码",
    "听一点",
    "一点",
)


def _is_plausible_artist_entity(value: Any) -> bool:
    """Reject broad context phrases before they become hard artist filters."""
    text = str(value or "").strip(" ，,。.!！?？")
    if not text:
        return False
    folded = _norm(text)
    if len(text) > 24:
        return False
    if re.search(r"\s{2,}", text):
        return False
    if re.search(r"[，,。.!！?？；;：:]", text):
        return False
    if any(cue in folded for cue in _NON_ARTIST_ENTITY_CUES):
        return False
    if re.search(r"(的)?(歌|歌曲|音乐)$", folded):
        return False
    if re.search(r"^(我|你|他|她|他们|她们|大家|某个|一个|一种)", folded):
        return False
    return True


def _clean_artist_entities(values: list[str]) -> list[str]:
    return [value for value in _merge_unique([], values) if _is_plausible_artist_entity(value)]


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


def is_followup_turn(
    user_input: str,
    dialog_state: DialogMusicState | dict[str, Any] | None,
) -> bool:
    """Return whether a previous state exists for the LLM delta planner.

    This intentionally does not inspect follow-up keywords. If a semantic
    inheritance decision is needed, the LLM delta planner decides it.
    """
    state = load_dialog_state(dialog_state)
    if state.pending_clarification is not None:
        return True
    _ = user_input
    return state.turn_count > 0


def _get_state_path(state: DialogMusicState, path: str) -> Any:
    section, field = path.split(".", 1)
    return getattr(getattr(state, section), field)


def _set_state_path(state: DialogMusicState, path: str, value: Any) -> None:
    section, field = path.split(".", 1)
    container = getattr(state, section)
    current = getattr(container, field)
    if isinstance(current, list):
        values = value if isinstance(value, list) else [value]
        setattr(container, field, _merge_unique([], values))
    elif isinstance(current, bool):
        setattr(container, field, bool(value))
    else:
        setattr(container, field, value)


def _clear_state_path(state: DialogMusicState, path: str) -> None:
    current = _get_state_path(state, path)
    if isinstance(current, list):
        _set_state_path(state, path, [])
    elif isinstance(current, bool):
        _set_state_path(state, path, False)
    elif path.startswith("soft_intent."):
        _set_state_path(state, path, "")
    else:
        _set_state_path(state, path, None)


def _operation_report(
    previous: DialogMusicState,
    updated: DialogMusicState,
    delta: PlanDelta,
) -> DialogStateDelta:
    prev_values = _state_value_map(previous)
    new_values = _state_value_map(updated)
    added: dict[str, Any] = {}
    replaced: dict[str, Any] = {}
    removed: list[str] = []
    inherited: list[str] = []
    touched = {operation.path for operation in delta.operations if operation.path}

    for path, new_value in new_values.items():
        old_value = prev_values.get(path)
        if path not in touched and _non_empty(old_value) and old_value == new_value:
            inherited.append(path)
        elif not _non_empty(old_value) and _non_empty(new_value):
            added[path] = new_value
        elif _non_empty(old_value) and _non_empty(new_value) and old_value != new_value:
            replaced[path] = new_value
        elif _non_empty(old_value) and not _non_empty(new_value):
            removed.append(path)

    return DialogStateDelta(
        followup=True,
        topic_shift=any(operation.op == "clear_topic" for operation in delta.operations),
        confidence=delta.confidence,
        reason="delta_applied",
        inherited=inherited,
        added=added,
        replaced=replaced,
        removed=removed,
        operations=[operation.model_dump() for operation in delta.operations],
        resolved_references=list(delta.resolved_references),
        ambiguity_reasons=list(delta.ambiguity_reasons),
        planner_mode=delta.planner_mode,
    )


def apply_plan_delta_operations(
    previous: DialogMusicState | dict[str, Any] | None,
    delta: PlanDelta,
    user_input: str,
) -> tuple[DialogMusicState, DialogStateDelta]:
    """Apply a whitelisted PlanDelta without asking an LLM to rewrite state."""
    prev = load_dialog_state(previous)
    updated = prev.model_copy(deep=True)

    for operation in delta.operations:
        if operation.op == "clear_topic":
            updated.hard_constraints = HardConstraints()
            updated.soft_intent = SoftIntent()
            updated.hints = IntentHints()
            updated.last_vector_acoustic_query = ""
            updated.last_complete_plan = {}
            continue

        current = _get_state_path(updated, operation.path)
        if operation.op == "add":
            if isinstance(current, list):
                values = operation.value if isinstance(operation.value, list) else [operation.value]
                _set_state_path(updated, operation.path, _merge_unique(current, values))
            elif not _non_empty(current):
                _set_state_path(updated, operation.path, operation.value)
            elif str(operation.value or "").strip() and str(operation.value) not in str(current):
                _set_state_path(updated, operation.path, f"{current}; {operation.value}")
        elif operation.op == "replace":
            _set_state_path(updated, operation.path, operation.value)
        elif operation.op == "remove":
            if isinstance(current, list) and operation.value not in (None, "", []):
                removals = operation.value if isinstance(operation.value, list) else [operation.value]
                folded = {_norm(value) for value in removals}
                _set_state_path(
                    updated,
                    operation.path,
                    [value for value in current if _norm(value) not in folded],
                )
            else:
                _clear_state_path(updated, operation.path)

    updated.last_query = user_input
    updated.turn_count = prev.turn_count + 1
    updated.pending_clarification = None
    report = _operation_report(prev, updated, delta)
    updated.last_delta = report
    return updated, report


def build_deterministic_plan_delta(
    user_input: str,
    dialog_state: DialogMusicState | dict[str, Any] | None,
) -> PlanDelta | None:
    """Compatibility hook; complex intent deltas are intentionally LLM-first.

    Earlier versions used regex rules here for follow-up phrases such as
    "quieter" or "rainy". That made context inheritance fast, but it also
    encoded brittle taste assumptions in Python. Keep the public function so
    old callers do not break, but let the IntentDeltaPlanner/LLM own all
    semantic delta decisions.
    """
    _ = user_input, dialog_state
    return None


def clarification_from_delta(
    delta: PlanDelta,
    *,
    confidence_threshold: float,
) -> ClarificationRequest:
    """Honor only high-precision clarification reasons."""
    clarification = delta.clarification
    high_precision = {
        "unresolved_reference",
        "missing_key_slot",
        "severe_conflict",
        "entity_ambiguity",
    }
    reasons = {_norm(reason).replace(" ", "_") for reason in delta.ambiguity_reasons}
    if clarification and clarification.required and (
        _norm(clarification.reason).replace(" ", "_") in high_precision
        or reasons.intersection(high_precision)
    ):
        return clarification
    if delta.confidence < confidence_threshold and reasons.intersection(high_precision):
        return ClarificationRequest(
            required=True,
            reason=next(iter(reasons.intersection(high_precision))),
            question="我还不能可靠地确定你想保留或替换哪一部分。你想按哪个方向继续？",
            options=["保留上一轮氛围", "只按这句话重新推荐", "告诉我一首参考歌"],
        )
    return ClarificationRequest()


def compile_dialog_state_to_plan(
    dialog_state: DialogMusicState,
    user_input: str,
) -> MusicQueryPlan:
    """Compile session state into the existing executable MusicQueryPlan."""
    if dialog_state.last_complete_plan:
        try:
            plan = MusicQueryPlan.model_validate(dialog_state.last_complete_plan)
        except Exception:
            plan = MusicQueryPlan(intent_type="hybrid_search")
    else:
        plan = MusicQueryPlan(intent_type="hybrid_search")

    updated = plan.model_copy(deep=True)
    updated.parameters = {
        "query": user_input,
        "entities": _merge_unique(
            dialog_state.hard_constraints.artist_entities,
            dialog_state.hard_constraints.song_entities,
        ),
    }
    updated.context = "基于结构化对话状态继续推荐"
    updated.reasoning = "A7 PlanDelta 经白名单校验后确定性应用"
    updated.retrieval_plan.hard_constraints = dialog_state.hard_constraints.model_copy(deep=True)
    updated.retrieval_plan.soft_intent = dialog_state.soft_intent.model_copy(deep=True)
    updated.retrieval_plan.hints = dialog_state.hints.model_copy(deep=True)
    updated.retrieval_plan.use_graph = True
    updated.retrieval_plan.use_vector = True
    updated.retrieval_plan.use_web_search = False

    has_entities = bool(
        dialog_state.hard_constraints.artist_entities
        or dialog_state.hard_constraints.song_entities
    )
    has_soft = bool(
        dialog_state.soft_intent.goal
        or dialog_state.soft_intent.trajectory
        or dialog_state.soft_intent.vibe
        or dialog_state.soft_intent.avoid
    )
    updated.intent_type = "hybrid_search" if has_entities or has_soft else "graph_search"

    vector_parts: list[str] = []
    for part in (
        dialog_state.soft_intent.goal,
        dialog_state.soft_intent.trajectory,
        dialog_state.soft_intent.vibe,
        ", ".join(dialog_state.soft_intent.avoid),
        ", ".join(dialog_state.hints.genres),
        dialog_state.hints.mood,
        dialog_state.hints.scenario,
        dialog_state.last_vector_acoustic_query,
        user_input,
    ):
        value = str(part or "").strip()
        if value and value not in vector_parts:
            vector_parts.append(value)
    updated.retrieval_plan.vector_acoustic_query = " ; ".join(vector_parts)
    updated.retrieval_plan = type(updated.retrieval_plan).model_validate(
        updated.retrieval_plan.model_dump()
    )
    return updated


def update_dialog_result_anchors(
    dialog_state: DialogMusicState | dict[str, Any] | None,
    recommendations: list[dict[str, Any]],
    *,
    limit: int = 12,
) -> DialogMusicState:
    """Persist only compact session-local result anchors for later references."""
    state = load_dialog_state(dialog_state).model_copy(deep=True)
    titles: list[str] = []
    artists: list[str] = []
    for item in recommendations:
        song = item.get("song", item) if isinstance(item, dict) else {}
        if not isinstance(song, dict):
            continue
        title = str(song.get("title") or "").strip()
        artist = str(song.get("artist") or "").strip()
        if title and title not in titles:
            titles.append(title)
        if artist and artist not in artists:
            artists.append(artist)
        if len(titles) >= limit and len(artists) >= limit:
            break
    state.last_result_titles = titles[:limit]
    state.last_result_artists = artists[:limit]
    return state


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
    if _has_any(text, UNRESOLVED_REFERENCE_CUES):
        explicit_new_entity = False
    reset_words = ("换个话题", "重新开始", "新歌单", "不要管上面", "from scratch", "new topic")
    return explicit_new_entity or _has_any(text, reset_words)


def should_clarify_before_planning(
    user_input: str,
    dialog_state: DialogMusicState | dict[str, Any] | None,
) -> ClarificationRequest:
    """Ask only for unresolved references that cannot be grounded."""
    state = load_dialog_state(dialog_state)
    if state.pending_clarification is not None:
        return ClarificationRequest()
    has_state = state.turn_count > 0
    text = _norm(user_input)
    severe_conflict = (
        re.search(r"安静|助眠|睡前|quiet|sleep", text)
        and re.search(r"炸裂|蹦迪|狂暴|extremely loud|raging party", text)
    )
    instrumental_voice_conflict = (
        re.search(r"完全无歌词|无歌词|纯音乐|器乐|无人声|instrumental|without vocals|no vocals|no lyrics", text)
        and re.search(r"中文说唱|说唱|rap|突出人声|人声|vocal|vocals", text)
    )
    if severe_conflict:
        return ClarificationRequest(
            required=True,
            reason="severe_conflict",
            question="“安静助眠”和“炸裂蹦迪”是两个相反方向。你希望这次更偏哪一边？",
            options=["安静助眠", "有节奏但不吵", "直接来高能量"],
            unresolved_paths=["soft_intent.vibe", "hints.scenario"],
        )
    if instrumental_voice_conflict:
        return ClarificationRequest(
            required=True,
            reason="severe_conflict",
            question="“完全无歌词/纯音乐”和“说唱/突出人声”会互相冲突。你这次更想保留哪一个方向？",
            options=["完全无歌词", "中文说唱人声", "保留节奏但弱化人声"],
            unresolved_paths=["hard_constraints.instrumental", "soft_intent.vibe", "hints.genres"],
        )
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
        asks_song = _has_any(text, ("那首歌", "刚才那首", "上一首"))
        asks_artist = _has_any(text, ("那个歌手", "他的歌", "她的歌"))
        if asks_song and not (state.last_result_titles or state.hard_constraints.song_entities):
            return ClarificationRequest(
                required=True,
                reason="missing_key_slot",
                question="你指的是哪一首歌？告诉我歌名，或从刚才的结果里点一首作为参考。",
                options=["告诉我歌名", "描述那首歌的感觉", "按上一轮氛围继续"],
                unresolved_paths=["hard_constraints.song_entities"],
            )
        if asks_artist and not (state.last_result_artists or state.hard_constraints.artist_entities):
            return ClarificationRequest(
                required=True,
                reason="missing_key_slot",
                question="你指的是哪位歌手？告诉我名字后，我会保留其余音乐偏好继续推荐。",
                options=["告诉我歌手名", "按上一轮氛围继续", "重新开始推荐"],
                unresolved_paths=["hard_constraints.artist_entities"],
            )
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


def clarification_from_plan_conflict(plan: MusicQueryPlan) -> ClarificationRequest:
    """Ask when the LLM-produced plan contains mutually exclusive constraints."""

    rp = plan.retrieval_plan
    hard = rp.hard_constraints
    evidence = _norm(
        " ".join(
            str(part or "")
            for part in (
                rp.graph_genre_filter,
                rp.graph_mood_filter,
                rp.graph_scenario_filter,
                " ".join(rp.hints.genres or []),
                rp.hints.mood,
                rp.hints.scenario,
                rp.soft_intent.goal,
                rp.soft_intent.trajectory,
                rp.soft_intent.vibe,
                " ".join(rp.soft_intent.avoid or []),
                rp.vector_acoustic_query,
                " ".join(rp.vector_acoustic_queries or []),
            )
        )
    )
    voice_forward = _has_any(
        evidence,
        (
            "说唱",
            "rap",
            "hip-hop",
            "hip hop",
            "人声",
            "vocal",
            "vocals",
            "voice-forward",
            "voice forward",
        ),
    )
    if hard.instrumental and voice_forward:
        return ClarificationRequest(
            required=True,
            reason="severe_conflict",
            question="“完全无歌词/纯音乐”和“中文说唱/突出人声”会互相冲突。你这次更想保留哪一个方向？",
            options=["完全无歌词", "中文说唱人声", "保留节奏但弱化人声"],
            unresolved_paths=["hard_constraints.instrumental", "soft_intent.vibe", "hints.genres"],
        )
    return ClarificationRequest()


def load_dialog_state(raw: DialogMusicState | dict[str, Any] | None) -> DialogMusicState:
    if isinstance(raw, DialogMusicState):
        return raw
    if isinstance(raw, dict) and raw:
        return DialogMusicState.model_validate(raw)
    return DialogMusicState()


def infer_dialog_state_from_history(chat_history: list[Any] | None) -> DialogMusicState:
    """Build a minimal legacy seed state without semantic regex extraction.

    ``dialog_state`` should be produced by the planner delta path.  Legacy
    chat history is kept only to indicate that a prior turn exists; semantic
    inheritance remains LLM-first.
    """
    if not chat_history:
        return DialogMusicState()

    user_turns = 0
    for message in chat_history:
        role = getattr(message, "type", None) or getattr(message, "role", None)
        content = getattr(message, "content", None)
        if isinstance(message, dict):
            role = message.get("role", role)
            content = message.get("content", content)
        if role not in ("user", "human"):
            continue
        if _norm(content):
            user_turns += 1

    return DialogMusicState(turn_count=user_turns)


def apply_plan_delta_with_report(
    previous: DialogMusicState | dict[str, Any] | None,
    plan: MusicQueryPlan,
    user_input: str,
) -> tuple[DialogMusicState, DialogStateDelta]:
    """Apply the current plan as a deterministic delta over prior state."""
    prev = load_dialog_state(previous)
    rp = plan.retrieval_plan
    topic_shift = _looks_like_topic_shift(user_input, plan)
    followup = prev.turn_count > 0 and not topic_shift

    # Full planner output is already expected to be a complete LLM-produced
    # plan. Do not fill missing fields from previous state here; that would
    # reintroduce deterministic inheritance outside the LLM.
    base = DialogMusicState()
    hard = HardConstraints(
        artist_entities=_clean_artist_entities(
            _merge_unique(
                base.hard_constraints.artist_entities if followup else [],
                rp.hard_constraints.artist_entities,
            )
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

    updated = DialogMusicState(
        hard_constraints=hard,
        soft_intent=soft,
        hints=hints,
        last_intent_type=plan.intent_type,
        last_query=user_input,
        last_vector_acoustic_query=rp.vector_acoustic_query or base.last_vector_acoustic_query,
        last_complete_plan=plan.model_dump(),
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
    hard = dialog_state.hard_constraints.model_copy(deep=True)
    hard.artist_entities = _clean_artist_entities(hard.artist_entities)
    rp.hard_constraints = hard
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
    _ = user_input
    if not state.last_delta.followup:
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
