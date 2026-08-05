from __future__ import annotations

import json

from data.sft.collect_v4_sealed_teacher import (
    collection_messages,
    parse_response,
    request_payload,
)


def _seed(index: int = 1) -> dict:
    return {
        "episode_id": f"sealed_{index:05d}",
        "turn_id": 0,
        "current_query": "推荐 Nala Sinephro 的代表作。",
        "entity": {"artist": "Nala Sinephro", "song": "Space 1"},
    }


def _decision() -> dict:
    return {
        "request_kind": "recommendation",
        "response_mode": "answer",
        "tool_names": ["graph", "web"],
        "hard": {"artist": ["Nala Sinephro"]},
    }


def test_parse_accepts_schema_valid_answer_and_rejects_false_clarification():
    seed = _seed()
    accepted, failures = parse_response(
        json.dumps(
            {"items": [{"seed_id": seed["episode_id"], "decision": _decision()}]}
        ),
        [seed],
    )
    assert len(accepted) == 1
    assert not failures

    clarify = {
        "request_kind": "recommendation",
        "response_mode": "clarify",
        "tool_names": [],
        "clarification": "你指哪位？",
    }
    accepted, failures = parse_response(
        json.dumps(
            {"items": [{"seed_id": seed["episode_id"], "decision": clarify}]}
        ),
        [seed],
    )
    assert not accepted
    assert failures["false_clarification"] == 1


def test_collection_prompt_keeps_graph_and_library_roles_distinct():
    system = collection_messages([_seed()])[0]["content"]
    assert "graph 是本地歌曲/歌手/知识卡" in system
    assert "library 只表示用户个人" in system


def test_request_is_deterministic_non_thinking_json():
    payload = request_payload([_seed()])
    assert payload["temperature"] == 0.0
    assert payload["enable_thinking"] is False
    assert payload["response_format"] == {"type": "json_object"}
