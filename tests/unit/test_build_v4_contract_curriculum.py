from __future__ import annotations

import json

from data.sft.build_v4_contract_curriculum import build_rows, entity_pool
from schemas.planner_decision_v3 import PlannerDecisionV3


def _source() -> list[dict]:
    return [
        {
            "messages": [
                {"role": "system", "content": "planner-system"},
                {"role": "user", "content": "[当前输入] 播放 The Cure 的 Lovesong"},
                {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "request_kind": "recommendation",
                            "response_mode": "answer",
                            "tool_names": ["graph"],
                            "hard": {
                                "artist": ["The Cure"],
                                "song": ["Lovesong"],
                            },
                        }
                    ),
                },
            ],
            "meta": {"episode_id": "source-1"},
        },
        {
            "messages": [
                {"role": "system", "content": "planner-system"},
                {"role": "user", "content": "[当前输入] 来点朴树"},
                {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "request_kind": "recommendation",
                            "response_mode": "answer",
                            "tool_names": ["graph"],
                            "hard": {"artist": ["朴树"]},
                        }
                    ),
                },
            ],
            "meta": {"episode_id": "source-2"},
        },
    ]


def test_entity_pool_preserves_original_names_and_song_pairs():
    artists, pairs = entity_pool(_source())
    assert artists == ["The Cure", "朴树"]
    assert pairs == [("The Cure", "Lovesong")]


def test_curriculum_is_balanced_unique_and_schema_valid():
    rows = build_rows(
        _source(),
        counts={
            "conversation": 3,
            "library": 3,
            "acquisition": 3,
            "information": 3,
        },
    )
    assert len(rows) == 12
    assert len({row["meta"]["episode_id"] for row in rows}) == 12
    assert len({row["messages"][1]["content"] for row in rows}) == 12
    assert {row["meta"]["request_kind"] for row in rows} == {
        "conversation",
        "library",
        "acquisition",
        "information",
    }
    for row in rows:
        decision = PlannerDecisionV3.model_validate_json(
            row["messages"][-1]["content"]
        )
        assert decision.request_kind == row["meta"]["request_kind"]
        assert row["meta"]["reviewer_verdict"] == "accept"
        assert row["messages"][0]["content"] == "planner-system"
