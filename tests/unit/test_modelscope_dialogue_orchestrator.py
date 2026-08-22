from __future__ import annotations

from deploy.modelscope_space import dialogue_orchestrator as dialogue


def test_ui_action_is_derived_only_from_validated_planner_fields() -> None:
    assert dialogue.planner_turn_kind({"task_mode": "recommendation"}) == "recommendation"
    assert dialogue.planner_turn_kind(
        {"task_mode": "dialogue", "dialogue_mode": "chat"}
    ) == "conversation"
    assert dialogue.planner_turn_kind(
        {"task_mode": "dialogue", "dialogue_mode": "information"}
    ) == "information"
    assert dialogue.planner_turn_kind(
        {"task_mode": "recommendation", "response_mode": "clarify"}
    ) == "clarification"


def test_currently_playing_song_is_the_reference_anchor() -> None:
    rows = [
        {"song_id": "one", "title": "First"},
        {"song_id": "two", "title": "Second"},
    ]

    assert dialogue.resolved_reference(rows, "two")["title"] == "Second"
    assert dialogue.resolved_reference(rows, "missing")["title"] == "First"


def test_history_is_bounded_and_only_keeps_visible_dialogue_roles() -> None:
    history = [
        {"role": "system", "content": "hidden"},
        *[{"role": "user" if index % 2 == 0 else "assistant", "content": f"turn-{index}"} for index in range(30)],
    ]

    bounded = dialogue.bounded_history(history)

    assert len(bounded) == 20
    assert bounded[0]["content"] == "turn-10"
    assert all(item["role"] in {"user", "assistant"} for item in bounded)
