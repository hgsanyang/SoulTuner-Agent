"""Planner-owned routing helpers for the unified public conversation surface.

The SoulTuner LoRA owns semantic intent.  Python validates the returned
contract and maps it to UI actions, but never guesses intent from user-facing
phrases.  This keeps the Creation Space aligned with the production Agent:
paraphrases and follow-ups are interpreted from assembled dialogue context
instead of a growing keyword list.
"""

from __future__ import annotations

from typing import Any


def planner_turn_kind(plan: dict[str, Any] | None) -> str:
    """Map one validated Planner decision to a UI action.

    No user text is inspected here.  The only inputs are structured fields
    emitted by the Planner and accepted by the guard.
    """

    current = plan or {}
    if str(current.get("response_mode") or "answer") == "clarify":
        return "clarification"
    if str(current.get("task_mode") or "recommendation") != "dialogue":
        return "recommendation"
    mode = str(current.get("dialogue_mode") or "chat")
    if mode == "information":
        return "information"
    return "conversation"


def resolved_reference(
    rows: list[dict[str, Any]] | None,
    selected_song_id: str | None,
) -> dict[str, Any]:
    """Return the currently playing result, falling back to the first row."""

    current = list(rows or [])
    selected = str(selected_song_id or "").strip()
    if selected:
        for row in current:
            if str(row.get("song_id") or "") == selected:
                return row
    return current[0] if current else {}


def novel_result_window(
    candidates: list[dict[str, Any]] | None,
    previous_rows: list[dict[str, Any]] | None,
    limit: int,
    *,
    preserve_previous: bool = False,
) -> list[dict[str, Any]]:
    """Prefer unseen candidates after a Planner-owned recommendation action.

    This is presentation policy, not intent classification: the validated
    Planner has already decided that the turn requests recommendations.  A
    follow-up therefore gets a genuinely new window without looking for any
    particular word in the user's message.  Explicit song constraints may opt
    out so an exact requested title is never filtered away.
    """

    requested = max(0, int(limit))
    current = list(candidates or [])
    if requested == 0 or preserve_previous or not previous_rows:
        return current[:requested]
    previous_ids = {
        str(row.get("song_id") or "").strip()
        for row in previous_rows
        if str(row.get("song_id") or "").strip()
    }
    unseen = [row for row in current if str(row.get("song_id") or "").strip() not in previous_ids]
    repeated = [row for row in current if str(row.get("song_id") or "").strip() in previous_ids]
    return (unseen + repeated)[:requested]


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
