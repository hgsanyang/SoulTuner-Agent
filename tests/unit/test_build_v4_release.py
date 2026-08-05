import json

from data.sft.build_v4_release import _normalise_failure_recovery
from schemas.planner_decision_v3 import PlannerDecisionV3, compile_v3_to_tool_plan
from schemas.tool_plan import ToolName


SYSTEM = "Return exactly one PlannerDecisionV3 JSON object."


def _plan(*, mode: str, tool_name: str | None, arguments: dict, clarify: bool = False) -> str:
    calls = []
    if tool_name:
        calls.append(
            {
                "id": "step",
                "name": tool_name,
                "arguments": arguments,
                "reason": "fixture",
            }
        )
    return json.dumps(
        {
            "version": "1.1",
            "origin": "replanner" if clarify else "planner",
            "request_mode": mode,
            "tool_calls": calls,
            "needs_clarification": clarify,
            "clarification_question": "请提供更明确的版本信息。" if clarify else "",
            "confidence": 1.0,
            "decision_summary": "recover safely",
            "max_replans": 1,
        },
        ensure_ascii=False,
    )


def _row(*, kind: str, initial: str, recovery: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": "old recovery contract"},
            {"role": "user", "content": '{"current_query":"test"}'},
            {"role": "assistant", "content": initial},
            {"role": "tool", "content": '[{"status":"empty"}]'},
            {"role": "assistant", "content": recovery},
        ],
        "meta": {
            "request_kind": kind,
            "trajectory_kind": "failure_recovery",
            "observation_origin": "harness_execution",
        },
    }


def _decision(row: dict) -> PlannerDecisionV3:
    assistant = [message for message in row["messages"] if message["role"] == "assistant"]
    return PlannerDecisionV3.model_validate_json(assistant[-1]["content"])


def test_failure_recovery_becomes_one_v3_target_with_observation_context():
    row = _row(
        kind="recommendation",
        initial=_plan(
            mode="recommendation",
            tool_name="search_graph",
            arguments={"artist_entities": ["The Blue Nile"], "limit": 20},
        ),
        recovery=_plan(
            mode="recommendation",
            tool_name="search_external_music",
            arguments={"requirements": "The Blue Nile songs", "entities": ["The Blue Nile"], "limit": 8},
        ),
    )

    output = _normalise_failure_recovery(row, system_prompt=SYSTEM)
    decision = _decision(output)

    assert [message["role"] for message in output["messages"]] == ["system", "user", "assistant"]
    assert "[OBSERVED TOOL RESULT]" in output["messages"][1]["content"]
    assert decision.tool_names == ["web"]
    assert decision.hard.artist == ["The Blue Nile"]


def test_acquisition_recovery_keeps_reversible_ingest_lane():
    row = _row(
        kind="acquisition",
        initial=_plan(
            mode="acquisition",
            tool_name="search_external_music",
            arguments={"requirements": "find the song", "entities": [], "limit": 8},
        ),
        recovery=_plan(
            mode="acquisition",
            tool_name="search_external_music",
            arguments={"requirements": "find an alternate version", "entities": [], "limit": 8},
        ),
    )

    decision = _decision(_normalise_failure_recovery(row, system_prompt=SYSTEM))
    compiled = compile_v3_to_tool_plan(decision)

    assert decision.tool_names == ["web", "ingest"]
    assert ToolName.STAGE_INGEST in {call.name for call in compiled.tool_calls}


def test_clarifying_recovery_runs_no_tools():
    row = _row(
        kind="acquisition",
        initial=_plan(
            mode="acquisition",
            tool_name="search_external_music",
            arguments={"requirements": "ambiguous title", "entities": [], "limit": 8},
        ),
        recovery=_plan(mode="acquisition", tool_name=None, arguments={}, clarify=True),
    )

    decision = _decision(_normalise_failure_recovery(row, system_prompt=SYSTEM))

    assert decision.response_mode == "clarify"
    assert decision.tool_names == []
