import asyncio

import agent.intent.delta_planner as delta_planner_module
from agent.intent.delta_planner import IntentDeltaPlanner, _parse_delta_json
from config.settings import settings
from schemas.dialog_state import DialogMusicState, DeltaOperation, PlanDelta
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


def test_delta_operation_normalizes_slash_paths():
    operation = DeltaOperation(op="replace", path="$.soft_intent/avoid", value="loud")

    assert operation.path == "soft_intent.avoid"


def test_delta_planner_uses_llm_for_complex_followup(monkeypatch):
    monkeypatch.setattr(settings, "intent_llm_provider", "dashscope")
    monkeypatch.setattr(settings, "llm_default_provider", "dashscope")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    called = {"llm": False}

    async def fake_post_json(*_args, **_kwargs):
        return {
            "choices": [
                {
                    "message": {
                        "content": PlanDelta(
                            operations=[
                                DeltaOperation(op="replace", path="hard_constraints.language", value="Chinese"),
                                DeltaOperation(op="add", path="soft_intent.vibe", value="quieter"),
                            ],
                            confidence=0.82,
                        ).model_dump_json()
                    }
                }
            ]
        }

    def llm_factory():
        called["llm"] = True
        return type("FakeLLM", (), {"model_name": "qwen3.7-plus"})()

    monkeypatch.setattr(delta_planner_module, "_post_json", fake_post_json)

    planner = IntentDeltaPlanner(llm_factory)
    state = DialogMusicState(
        hard_constraints=HardConstraints(language="English"),
        soft_intent=SoftIntent(vibe="dreamy"),
        turn_count=1,
    )

    delta = asyncio.run(
        planner.plan(user_input="同样的感觉，换中文并更安静", dialog_state=state)
    )

    assert called["llm"] is True
    assert delta.planner_mode == "delta_llm"
    assert [operation.path for operation in delta.operations] == [
        "hard_constraints.language",
        "soft_intent.vibe",
    ]
