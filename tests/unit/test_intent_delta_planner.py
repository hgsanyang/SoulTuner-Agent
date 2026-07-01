import asyncio

from agent.intent.delta_planner import IntentDeltaPlanner, _parse_delta_json
from schemas.dialog_state import DialogMusicState
from schemas.query_plan import HardConstraints, SoftIntent


def test_delta_json_parser_accepts_fenced_payload():
    delta = _parse_delta_json(
        """```json
        {"operations":[{"op":"replace","path":"hard_constraints.language","value":"Chinese"}],
         "confidence":0.9,"planner_mode":"delta_llm"}
        ```"""
    )

    assert delta.operations[0].path == "hard_constraints.language"
    assert delta.confidence == 0.9


def test_delta_planner_uses_deterministic_fast_path_without_llm():
    def fail_if_called():
        raise AssertionError("LLM should not be called for a deterministic follow-up")

    planner = IntentDeltaPlanner(fail_if_called)
    state = DialogMusicState(
        hard_constraints=HardConstraints(language="English"),
        soft_intent=SoftIntent(vibe="dreamy"),
        turn_count=1,
    )

    delta = asyncio.run(
        planner.plan(user_input="同样的感觉，换中文并更安静", dialog_state=state)
    )

    assert delta.planner_mode == "deterministic"
    assert len(delta.operations) == 2
