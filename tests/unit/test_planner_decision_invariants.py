"""Invariants for the distillation contract + student scorer gate (GPT 2nd review)."""

from __future__ import annotations

import json

import pytest

from schemas.planner_decision import PlannerDecisionV2


def _base(**kw):
    d = {"intent": "hybrid_search", "tool_names": ["graph", "dense"],
         "acoustic_queries": ["warm mellow acoustic"]}
    d.update(kw)
    return d


def test_valid_decisions_pass():
    PlannerDecisionV2.model_validate(_base())
    PlannerDecisionV2.model_validate({"intent": "graph_search", "tool_names": ["graph"]})
    PlannerDecisionV2.model_validate({"intent": "vector_search", "tool_names": ["dense"],
                                      "acoustic_queries": ["dreamy"]})
    PlannerDecisionV2.model_validate({"intent": "clarification", "clarification": "你想要哪种?"})
    PlannerDecisionV2.model_validate({"intent": "general_chat"})
    # empty tool_names is a valid SCHEMA state (inference-time compiler fallback);
    # a training TARGET for a recall intent must still name tools — that stricter
    # rule is enforced by verify_episodes V10, not by the schema.
    PlannerDecisionV2.model_validate({"intent": "graph_search"})


def test_clarification_requires_text():
    with pytest.raises(ValueError):
        PlannerDecisionV2.model_validate({"intent": "clarification"})


def test_clarification_forbids_lanes():
    with pytest.raises(ValueError):
        PlannerDecisionV2.model_validate({"intent": "clarification", "clarification": "?",
                                          "tool_names": ["graph"]})


def test_clarification_text_only_for_clarification():
    with pytest.raises(ValueError):
        PlannerDecisionV2.model_validate(_base(clarification="不该出现"))


def test_general_chat_no_lanes():
    with pytest.raises(ValueError):
        PlannerDecisionV2.model_validate({"intent": "general_chat", "tool_names": ["dense"]})


def test_unknown_tool_rejected():
    with pytest.raises(ValueError):
        PlannerDecisionV2.model_validate(_base(tool_names=["graph", "telepathy"]))


def test_recall_intent_requires_its_lane():
    # hybrid/vector must include dense when lanes are named
    with pytest.raises(ValueError):
        PlannerDecisionV2.model_validate({"intent": "vector_search", "tool_names": ["graph"],
                                          "acoustic_queries": ["x"]})
    with pytest.raises(ValueError):
        PlannerDecisionV2.model_validate({"intent": "web_search", "tool_names": ["dense"],
                                          "acoustic_queries": ["x"]})


def test_scorer_gate_flags_incomplete(tmp_path):
    from data.sft.score_student import score

    gold = {"messages": [{"role": "user", "content": "放周杰伦"},
                          {"role": "assistant", "content": json.dumps({"intent": "graph_search",
                                                                        "tool_names": ["graph"]})}],
            "meta": {"episode_id": "e1", "turn_id": 0}}
    ev = tmp_path / "eval.jsonl"
    ev.write_text(json.dumps(gold, ensure_ascii=False) + "\n"
                  + json.dumps({**gold, "meta": {"episode_id": "e2", "turn_id": 0}}, ensure_ascii=False) + "\n",
                  encoding="utf-8")
    # prediction file missing e2 -> coverage incomplete
    pred = tmp_path / "pred.jsonl"
    pred.write_text(json.dumps({**gold, "prediction": gold["messages"][-1]["content"]}, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    report = score(ev, pred)
    assert report["coverage"]["missing"] == 1
    assert report["coverage"]["complete"] is False


def test_scorer_detects_duplicate_gold(tmp_path):
    from data.sft.score_student import score

    row = {"messages": [{"role": "user", "content": "同一条"},
                        {"role": "assistant", "content": json.dumps({"intent": "graph_search",
                                                                     "tool_names": ["graph"]})}],
           "meta": {"episode_id": "e1", "turn_id": 0}}
    ev = tmp_path / "eval.jsonl"
    ev.write_text(json.dumps(row, ensure_ascii=False) + "\n" + json.dumps(row, ensure_ascii=False) + "\n",
                  encoding="utf-8")
    pred = tmp_path / "pred.jsonl"
    pred.write_text(json.dumps({**row, "prediction": row["messages"][-1]["content"]}, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    report = score(ev, pred)
    assert report["coverage"]["gold_duplicate"] == 1
    assert report["coverage"]["complete"] is False


def test_scorer_accepts_complete_v3_predictions(tmp_path):
    from data.sft.score_student import score

    decision = {
        "request_kind": "library",
        "response_mode": "answer",
        "tool_names": ["library"],
    }
    row = {
        "messages": [
            {"role": "user", "content": "查看我的收藏"},
            {"role": "assistant", "content": json.dumps(decision)},
        ],
        "meta": {"episode_id": "v3-library", "turn_id": 0},
    }
    ev = tmp_path / "eval-v3.jsonl"
    pred = tmp_path / "pred-v3.jsonl"
    ev.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    pred.write_text(
        json.dumps(
            {**row, "prediction": json.dumps(decision, ensure_ascii=False)},
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    report = score(ev, pred)

    assert report["coverage"]["complete"] is True
    assert report["schema_valid"] == 1.0
    assert report["compilable"] == 1.0
    assert report["request_kind_acc"] == 1.0
    assert report["lane_f1"] == 1.0


def test_scorer_cli_nonzero_exit_on_incomplete(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    rows = [
        {"messages": [{"role": "user", "content": f"q{i}"},
                      {"role": "assistant", "content": json.dumps({"intent": "graph_search",
                                                                   "tool_names": ["graph"]})}],
         "meta": {"episode_id": f"e{i}", "turn_id": 0}}
        for i in range(2)
    ]
    ev = tmp_path / "eval.jsonl"
    ev.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    pred = tmp_path / "pred.jsonl"  # only 1 of 2 -> incomplete
    pred.write_text(json.dumps({**rows[0], "prediction": rows[0]["messages"][-1]["content"]},
                               ensure_ascii=False) + "\n", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "data.sft.score_student", "--eval", str(ev), "--pred", str(pred)],
        cwd=str(root), capture_output=True, text=True,
    )
    assert result.returncode == 1, f"expected non-zero exit, got {result.returncode}: {result.stdout}"
