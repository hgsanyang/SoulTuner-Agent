from data.sft.generate_v3_gap_seeds import build_seeds
from data.sft.review_v3_ambiguous import (
    FALSE_CLARIFICATION_REASONS,
    VALID_CLARIFICATION_IDS,
    WEB_INFORMATION_IDS,
    WEB_RECOMMENDATION_IDS,
    _review_row,
)
from schemas.planner_decision_v3 import PlannerDecisionV3
from schemas.tool_plan import ToolName, ToolPlan


def _ambiguous_row(episode_id: str) -> dict:
    is_clarification = (
        episode_id in VALID_CLARIFICATION_IDS
        or episode_id in FALSE_CLARIFICATION_REASONS
    )
    decision = {
        "request_kind": "recommendation" if is_clarification else "information",
        "response_mode": "clarify" if is_clarification else "answer",
        "tool_names": [] if is_clarification else ["web"],
        "clarification": "你更偏向哪一种？" if is_clarification else None,
        "decision_summary": "frozen review contract fixture",
    }
    return {
        "episode_id": episode_id,
        "turn_id": 0,
        "current_query": f"fixture query for {episode_id}",
        "chat_history": "",
        "teacher_decision_v3": decision,
        "migration": {"legacy_intent": "clarification" if is_clarification else "web_search"},
        "provenance": {"source_type": "unit_test_fixture"},
    }


def test_ambiguous_review_table_has_exactly_71_explicit_decisions():
    reviewed_ids = (
        WEB_INFORMATION_IDS
        | WEB_RECOMMENDATION_IDS
        | VALID_CLARIFICATION_IDS
        | set(FALSE_CLARIFICATION_REASONS)
    )
    assert len(reviewed_ids) == 71
    assert len(WEB_INFORMATION_IDS) == 31
    assert len(WEB_RECOMMENDATION_IDS) == 17
    assert len(VALID_CLARIFICATION_IDS) == 16
    assert len(FALSE_CLARIFICATION_REASONS) == 7


def test_review_logic_keeps_false_clarifications_out_of_trainable_data():
    reviewed_ids = (
        WEB_INFORMATION_IDS
        | WEB_RECOMMENDATION_IDS
        | VALID_CLARIFICATION_IDS
        | set(FALSE_CLARIFICATION_REASONS)
    )
    reviews = []
    trainable = []
    for episode_id in sorted(reviewed_ids):
        review, resolved = _review_row(_ambiguous_row(episode_id))
        reviews.append(review)
        if resolved is not None:
            trainable.append(resolved)

    excluded = {
        row["episode_id"]
        for row in reviews
        if not row["training_eligible"]
    }
    included = {row["episode_id"] for row in trainable}

    assert len(reviews) == 71
    assert len(trainable) == 64
    assert excluded == set(FALSE_CLARIFICATION_REASONS)
    assert not (excluded & included)
    assert all(
        row["training_governance"]["training_eligible"] is True
        for row in trainable
    )
    for row in trainable:
        PlannerDecisionV3.model_validate(row["teacher_decision_v3"])


def test_curated_gap_seeds_are_schema_valid_and_balanced():
    rows = build_seeds()
    counts = {"conversation": 0, "acquisition": 0, "library": 0}
    for row in rows:
        decision = PlannerDecisionV3.model_validate(row["teacher_decision_v3"])
        plan = ToolPlan.model_validate(row["tool_plan"])
        counts[decision.request_kind] += 1
        assert row["training_governance"] == {
            "training_eligible": True,
            "data_purpose": "planner_v3_sft",
            "source_type": "curated_v3_gate_seed",
            "seed_version": "planner_v3_gap_seeds_2026_07_28",
        }
        if decision.request_kind == "library":
            assert [call.name for call in plan.tool_calls] == [ToolName.READ_LIBRARY]
        elif decision.request_kind == "acquisition":
            assert [call.name for call in plan.tool_calls] == [
                ToolName.SEARCH_EXTERNAL_MUSIC,
                ToolName.STAGE_INGEST,
            ]
        else:
            assert plan.tool_calls == []

    assert counts == {"conversation": 8, "acquisition": 8, "library": 8}
