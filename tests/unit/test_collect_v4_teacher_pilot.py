from __future__ import annotations

import json

import pytest

from data.sft.collect_v4_teacher_pilot import (
    build_messages,
    build_request_payload,
    build_seeds,
    parse_batch,
    write_private,
)


def _decision(kind: str, mode: str = "answer") -> dict:
    if mode == "clarify":
        return {
            "request_kind": kind,
            "response_mode": "clarify",
            "tool_names": [],
            "clarification": "你指的是哪一个版本？",
        }
    lanes = {
        "recommendation": ["graph"],
        "information": ["graph"],
        "acquisition": ["ingest"],
        "library": ["library"],
        "conversation": [],
    }[kind]
    return {
        "request_kind": kind,
        "response_mode": "answer",
        "tool_names": lanes,
    }


def test_seed_curriculum_is_unique_and_keeps_clarification_ratio():
    seeds = build_seeds(10)
    assert len({seed.seed_id for seed in seeds}) == len(seeds)
    positive = sum(
        seed.trajectory_kind == "clarification_positive" for seed in seeds
    )
    negative = sum(
        seed.trajectory_kind == "clarification_negative" for seed in seeds
    )
    assert positive == 10
    assert negative == 15


def test_batch_parser_accepts_only_expected_mode_and_kind():
    seeds = build_seeds(2)[:3]
    items = []
    for seed in seeds:
        items.append(
            {
                "seed_id": seed.seed_id,
                "decision": _decision(seed.expected_kind, seed.expected_mode),
            }
        )
    accepted, failures = parse_batch(json.dumps({"items": items}), seeds)
    assert len(accepted) == 3
    assert not failures

    items[0]["decision"] = _decision("information")
    accepted, failures = parse_batch(json.dumps({"items": items}), seeds)
    assert len(accepted) == 2
    assert failures["kind_mismatch"] == 1


def test_request_disables_thinking_and_uses_json_mode():
    seeds = build_seeds(2)[:2]
    payload = build_request_payload(seeds)
    assert payload["model"] == "qwen3.7-plus"
    assert payload["temperature"] == 0.0
    assert payload["enable_thinking"] is False
    assert payload["response_format"] == {"type": "json_object"}
    system = build_messages(seeds)[0]["content"]
    assert "graph 查询本地歌曲、歌手、知识卡" in system
    assert "library 只查询当前用户" in system
    assert "acquisition 必须含 ingest" in system


def test_private_writer_refuses_public_paths(tmp_path):
    with pytest.raises(ValueError, match="private"):
        write_private(tmp_path / "pilot.jsonl", [], {})
