from __future__ import annotations

from collections import Counter

import pytest

from data.sft.build_v4_balanced_sealed import build_candidates
from data.sft.review_v4_sealed import replace_reviewed
from data.sft.v4_contract import row_contract_errors
from schemas.planner_decision import DecisionHard, DecisionSoft
from schemas.planner_decision_v3 import PlannerDecisionV3


SYSTEM = "Return one PlannerDecisionV3 JSON object."


def _teacher_row(index: int, kind: str) -> dict:
    artist = f"Unseen Artist {index:03d}"
    if kind == "recommendation":
        decision = PlannerDecisionV3(
            request_kind="recommendation",
            tool_names=["graph", "dense"],
            hard=DecisionHard(artist=[artist]),
            soft=DecisionSoft(goal="discover unfamiliar music"),
            acoustic_queries=[
                f"Distinct atmospheric music with restrained dynamics, variation {index}."
            ],
            decision_summary="recommend an unseen artist",
        )
    else:
        decision = PlannerDecisionV3(
            request_kind="information",
            tool_names=["graph", "web"],
            hard=DecisionHard(artist=[artist]),
            decision_summary="verify artist information",
        )
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"[当前输入] request {kind} {artist}"},
            {"role": "assistant", "content": decision.model_dump_json(exclude_none=True)},
        ],
        "meta": {"request_kind": kind},
        "lineage": {"entity": artist},
    }


def test_balanced_sealed_recommendations_cover_five_behaviours():
    teacher = [
        *(_teacher_row(index, "recommendation") for index in range(125)),
        *(_teacher_row(index + 125, "information") for index in range(125)),
    ]

    rows = build_candidates(teacher, per_kind=125)
    counts = Counter(row["meta"]["request_kind"] for row in rows)
    recommendation = [row for row in rows if row["meta"]["request_kind"] == "recommendation"]
    trajectories = Counter(row["meta"]["trajectory_kind"] for row in recommendation)

    assert counts == {
        "recommendation": 125,
        "information": 125,
        "acquisition": 125,
        "library": 125,
        "conversation": 125,
    }
    assert trajectories == {
        "single_turn": 25,
        "multi_turn_inheritance": 25,
        "memory_vs_current_request": 25,
        "clarification_positive": 25,
        "clarification_negative": 25,
    }
    for row in recommendation:
        errors = row_contract_errors(row)
        assert errors
        assert all("reviewer_verdict" in error for error in errors)


def test_partial_blind_review_replacement_preserves_all_five_classes():
    base = [
        {"meta": {"request_kind": kind, "episode_id": f"old-{kind}-{index}"}}
        for kind in ("recommendation", "information", "acquisition", "library", "conversation")
        for index in range(2)
    ]
    replacement = [
        {"meta": {"request_kind": "recommendation", "episode_id": f"new-{index}"}}
        for index in range(2)
    ]

    combined = replace_reviewed(
        base,
        replacement,
        kinds=("recommendation",),
        per_kind=2,
    )

    assert len(combined) == 10
    assert not any(row["meta"]["episode_id"].startswith("old-recommendation") for row in combined)


def test_partial_replacement_fails_closed_when_a_class_is_short():
    base = [
        {"meta": {"request_kind": kind, "episode_id": kind}}
        for kind in ("recommendation", "information", "acquisition", "library", "conversation")
    ]

    with pytest.raises(ValueError, match="unbalance"):
        replace_reviewed(base, [], kinds=("recommendation",), per_kind=1)
