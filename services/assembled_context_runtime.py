"""Deterministic runtime assembly for planner and general-chat context.

The model never decides which memory source wins. Code applies the shared
precedence policy, bounds every text channel, and records the exact structured
context used for the turn.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

from retrieval.gssc_context_builder import TokenCounter, build_context
from schemas.agent_context import (
    AssembledContext,
    CurrentContext,
    RecommendedAnchor,
    RetrievedMemory,
    SessionContext,
)
from schemas.dialog_state import load_dialog_state


@dataclass(frozen=True)
class PromptContextBundle:
    """The bounded renderings plus their auditable structured representation."""

    assembled: AssembledContext
    chat_history: str
    planner_preferences: str
    chat_memory: str

    def state_payload(self) -> dict[str, Any]:
        return {
            "assembled_context": self.assembled.model_dump(mode="json"),
            "prompt_context": {
                "chat_history": self.chat_history,
                "planner_preferences": self.planner_preferences,
                "chat_memory": self.chat_memory,
            },
        }


def _current_context(session_id: str, listening: Mapping[str, Any] | None) -> CurrentContext:
    source = dict(listening or {})
    return CurrentContext(
        local_time=str(source.get("local_time") or ""),
        timezone=str(source.get("timezone") or ""),
        local_hour=source.get("local_hour"),
        day_of_week=source.get("day_of_week"),
        day_type=source.get("day_type"),
        day_part=source.get("day_part"),
        explicit_scene=str(source.get("scene") or ""),
        session_id=session_id,
        turn_at_ms=source.get("ts_ms"),
    )


def _session_context(dialog_state: Mapping[str, Any] | None, previous_plan: str) -> SessionContext:
    state = load_dialog_state(dict(dialog_state or {}))
    delta = state.last_delta
    refinements: list[str] = []
    for label, values in (("新增", delta.added), ("替换", delta.replaced)):
        for key, value in values.items():
            rendered = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
            refinements.append(f"{label} {key}: {rendered}")
    anchors = [
        RecommendedAnchor(title=title, artist=artist, rank=index + 1)
        for index, (title, artist) in enumerate(
            zip(state.last_result_titles, state.last_result_artists, strict=False)
        )
        if title or artist
    ]
    return SessionContext(
        refinements=refinements[:12],
        rejected=list(state.soft_intent.avoid)[:12],
        last_plan_summary=previous_plan,
        last_recommendations=anchors[:10],
    )


def _retrieved_memories(memory_context: Mapping[str, Any] | None) -> list[RetrievedMemory]:
    records = list((memory_context or {}).get("retrieved_records") or [])
    result: list[RetrievedMemory] = []
    for index, item in enumerate(records[:8]):
        if not isinstance(item, Mapping):
            continue
        field = str(item.get("field") or "").strip()
        value = str(item.get("value") or "").strip()
        if not value:
            continue
        result.append(
            RetrievedMemory(
                memory_id=str(item.get("memory_id") or item.get("record_id") or f"memory-{index}"),
                statement=f"{field}={value}" if field else value,
                layer=str(item.get("layer") or ""),
                occurred_at_ms=item.get("created_at_ms"),
                relevance=max(0.0, min(1.0, float(item.get("relevance") or 0.0))),
            )
        )
    return result


async def assemble_prompt_context(
    *,
    user_input: str,
    user_id: str,
    session_id: str,
    chat_history: str,
    explicit_profile: str,
    long_memory: str,
    memory_context: Mapping[str, Any] | None = None,
    dialog_state: Mapping[str, Any] | None = None,
    previous_plan: str = "",
    listening_context: Mapping[str, Any] | None = None,
    total_budget: int = 0,
    token_counter: TokenCounter | None = None,
) -> PromptContextBundle:
    """Assemble one canonical context used by every language-model route."""

    bounded = await build_context(
        explicit_profile=explicit_profile,
        graphzep_facts=long_memory,
        chat_history=chat_history,
        user_input=user_input,
        total_budget=total_budget,
        user_id=user_id,
        conversation_id=session_id,
        token_counter=token_counter,
    )
    profile = bounded.get("explicit_profile", "").strip()
    memory = bounded.get("graphzep_facts", "").strip()
    memory_parts: list[str] = []
    if profile:
        memory_parts.append(f"【用户明确画像】\n{profile}")
    if memory and memory != "暂无用户长期记忆":
        memory_parts.append(f"【与当前请求相关的长期记忆】\n{memory}")
    rendered_memory = "\n".join(memory_parts) or "暂无可用用户记忆"
    assembled = AssembledContext(
        current=_current_context(session_id, listening_context),
        session=_session_context(dialog_state, previous_plan),
        explicit_profile=profile,
        retrieved_memories=_retrieved_memories(memory_context),
    )
    return PromptContextBundle(
        assembled=assembled,
        chat_history=bounded.get("chat_history", ""),
        planner_preferences=rendered_memory,
        chat_memory=rendered_memory,
    )
