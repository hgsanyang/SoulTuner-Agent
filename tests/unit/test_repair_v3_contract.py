from __future__ import annotations

import json

import pytest

from data.sft.repair_v3_contract import (
    ACOUSTIC_QUERY_REPAIRS,
    CJK_QUERY_REPLACEMENTS,
    REMOVE_DENSE_REPAIRS,
    repair_row,
)
from scripts.validate_sft_dataset import CJK, check_contract


def _row(key: str, payload: dict) -> dict:
    episode, turn = key.rsplit(":", 1)
    return {
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "[当前输入] test"},
            {"role": "assistant", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        "meta": {"episode_id": episode, "turn_id": int(turn)},
    }


def _decision(**updates) -> dict:
    value = {
        "request_kind": "recommendation",
        "response_mode": "answer",
        "tool_names": ["graph", "dense"],
        "hard": {},
        "soft": {},
        "hints": {},
        "metadata": {},
        "acoustic_queries": [],
        "decision_summary": "test",
    }
    value.update(updates)
    return value


@pytest.mark.parametrize("key", sorted(ACOUSTIC_QUERY_REPAIRS))
def test_missing_dense_queries_are_added_without_turning_tension_into_clarification(key):
    repaired, changed = repair_row(_row(key, _decision()))
    payload = json.loads(repaired["messages"][2]["content"])

    assert changed is True
    assert payload["response_mode"] == "answer"
    assert "dense" in payload["tool_names"]
    assert payload["acoustic_queries"] == ACOUSTIC_QUERY_REPAIRS[key]
    assert not check_contract([repaired], "test")[0]


@pytest.mark.parametrize("key", sorted(REMOVE_DENSE_REPAIRS))
def test_dense_is_removed_when_the_request_has_no_sonic_vector_target(key):
    repaired, changed = repair_row(
        _row(key, _decision(tool_names=["graph", "dense", "web"]))
    )
    payload = json.loads(repaired["messages"][2]["content"])

    assert changed is True
    assert payload["tool_names"] == ["graph", "web"]
    assert payload["acoustic_queries"] == []
    assert not check_contract([repaired], "test")[0]


@pytest.mark.parametrize(("key", "fragments"), sorted(CJK_QUERY_REPLACEMENTS.items()))
def test_mixed_language_acoustic_queries_become_english(key, fragments):
    old, new = fragments
    repaired, changed = repair_row(
        _row(key, _decision(acoustic_queries=[f"prefix {old} suffix"]))
    )
    payload = json.loads(repaired["messages"][2]["content"])

    assert changed is True
    assert new in payload["acoustic_queries"][0]
    assert not CJK.search(payload["acoustic_queries"][0])
    assert not check_contract([repaired], "test")[0]
