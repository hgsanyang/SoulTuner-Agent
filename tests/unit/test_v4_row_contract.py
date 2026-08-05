from __future__ import annotations

import json

from data.sft.v4_contract import row_contract_errors
from schemas.planner_decision_v3 import PlannerDecisionV3


def _row() -> dict:
    decision = PlannerDecisionV3(
        request_kind="library",
        tool_names=["library"],
        decision_summary="查看收藏",
    )
    return {
        "messages": [
            {"role": "system", "content": "planner"},
            {"role": "user", "content": "看看我的收藏"},
            {"role": "assistant", "content": decision.model_dump_json()},
        ],
        "meta": {
            "seed_source": "curated_seed",
            "episode_id": "library_1",
            "turn_id": 0,
            "request_kind": "library",
            "trajectory_kind": "library_state",
            "observation_origin": "none",
            "teacher": {"model": "teacher", "version": "1"},
            "reviewer": {"model": "reviewer", "version": "1"},
            "reviewer_verdict": "accept",
            "extension_field": "allowed",
        },
    }


def test_real_row_meta_is_bound_to_the_provenance_contract():
    row = _row()
    assert row_contract_errors(row) == []
    del row["meta"]["reviewer_verdict"]
    assert any("reviewer_verdict" in error for error in row_contract_errors(row))


def test_row_contract_checks_that_the_decision_compiles_without_dropping_library():
    row = _row()
    payload = json.loads(row["messages"][-1]["content"])
    payload["tool_names"] = []
    row["messages"][-1]["content"] = json.dumps(payload)
    assert any(error.startswith("decision:") for error in row_contract_errors(row))

