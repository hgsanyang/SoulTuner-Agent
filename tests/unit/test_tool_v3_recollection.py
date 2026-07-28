import json

import pytest

from data.sft.recollect_v3_false_clarifications import (
    DASHSCOPE_MODEL,
    build_request_payload,
    call_dashscope,
    ensure_private_output,
    recollect_rows,
    select_quarantined_rows,
    write_private_jsonl,
)
from data.sft.review_v3_ambiguous import FALSE_CLARIFICATION_REASONS


def _source_rows():
    return [
        {
            "episode_id": episode_id,
            "turn_id": 0,
            "current_query": f"query for {episode_id}",
            "chat_history": "",
            "previous_plan": "",
            "profile_snapshot": "",
            "retrieved_memories": [],
            "provenance": {"source_type": "pilot_synthetic"},
            "teacher_decision_v3": {
                "request_kind": "recommendation",
                "response_mode": "clarify",
                "tool_names": [],
                "clarification": "old question",
            },
            "migration": {"ambiguous": True},
        }
        for episode_id in sorted(FALSE_CLARIFICATION_REASONS)
    ]


def _answer_json():
    return json.dumps(
        {
            "request_kind": "recommendation",
            "response_mode": "answer",
            "tool_names": ["dense"],
            "acoustic_queries": [
                "natural instruments with an intimate timbre"
            ],
            "decision_summary": "直接检索可满足的听感",
        },
        ensure_ascii=False,
    )


def test_select_quarantined_rows_requires_exactly_the_frozen_seven():
    selected = select_quarantined_rows(_source_rows())
    assert len(selected) == 7
    assert {row["episode_id"] for row in selected} == set(
        FALSE_CLARIFICATION_REASONS
    )

    with pytest.raises(ValueError, match="source mismatch"):
        select_quarantined_rows(_source_rows()[:-1])


def test_request_payload_locks_strong_model_thinking_and_json_schema():
    payload = build_request_payload(_source_rows()[0])
    assert payload["model"] == DASHSCOPE_MODEL == "qwen3.7-plus"
    assert payload["enable_thinking"] is False
    assert payload["temperature"] == 0.0
    assert payload["response_format"] == {"type": "json_object"}
    assert "PlannerDecisionV3" in payload["messages"][0]["content"]
    assert "DASHSCOPE_API_KEY" not in json.dumps(
        payload,
        ensure_ascii=False,
    )


def test_recollection_accepts_only_schema_valid_answers():
    rows = _source_rows()
    invalid_id = rows[0]["episode_id"]
    clarify_id = rows[1]["episode_id"]

    def fake_model_call(row):
        if row["episode_id"] == invalid_id:
            return {"content": '{"response_mode":"answer","unexpected":true}'}
        if row["episode_id"] == clarify_id:
            return {
                "content": json.dumps(
                    {
                        "request_kind": "recommendation",
                        "response_mode": "clarify",
                        "tool_names": [],
                        "clarification": "still asking",
                    }
                )
            }
        return {"content": _answer_json()}

    accepted, summary = recollect_rows(rows, fake_model_call)
    accepted_ids = {row["episode_id"] for row in accepted}

    assert len(accepted) == 5
    assert invalid_id not in accepted_ids
    assert clarify_id not in accepted_ids
    assert summary["accepted"] == 5
    assert summary["still_quarantined"] == 2
    assert all(
        row["teacher_decision_v3"]["response_mode"] == "answer"
        for row in accepted
    )
    assert all(
        row["training_governance"]["training_eligible"] is True
        for row in accepted
    )


def test_private_writer_is_atomic_and_refuses_public_paths(tmp_path):
    public = tmp_path / "result.jsonl"
    with pytest.raises(ValueError, match="directory named private"):
        ensure_private_output(public)

    target = tmp_path / "private" / "result.jsonl"
    write_private_jsonl(target, [{"ok": True}])
    assert json.loads(target.read_text(encoding="utf-8")) == {"ok": True}
    assert not target.with_suffix(".jsonl.tmp").exists()


def test_dashscope_call_does_not_print_or_embed_key(capsys):
    captured = {}

    class _Response:
        @staticmethod
        def raise_for_status():
            return None

        @staticmethod
        def json():
            return {
                "choices": [{"message": {"content": _answer_json()}}],
                "usage": {"prompt_tokens": 10},
            }

    class _Client:
        @staticmethod
        def post(url, headers, json):
            captured["url"] = url
            captured["headers"] = headers
            captured["json"] = json
            return _Response()

    secret = "private-test-secret"
    response = call_dashscope(
        _source_rows()[0],
        api_key=secret,
        timeout_seconds=5,
        client=_Client(),
    )
    output = capsys.readouterr()

    assert response["content"] == _answer_json()
    assert captured["headers"]["Authorization"] == f"Bearer {secret}"
    assert secret not in json.dumps(captured["json"], ensure_ascii=False)
    assert secret not in output.out
    assert secret not in output.err
