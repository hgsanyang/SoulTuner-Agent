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


def _v3_gold(episode: str, turn: int, text: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": "决策器"},
            {"role": "user", "content": text},
            {"role": "assistant", "content": json.dumps(
                {"request_kind": "recommendation", "response_mode": "answer",
                 "tool_names": ["dense"], "acoustic_queries": ["warm piano"]},
                ensure_ascii=False)},
        ],
        "meta": {"episode_id": episode, "turn_id": turn},
    }


def test_predictions_without_meta_still_align_to_gold(tmp_path):
    """真实形状：`swift infer --result_path` 写出的行没有 meta。

    gold 有 meta 就按 episode 编索引，预测没有就退回按题目文本编索引 ——
    两边用了不同的钥匙，一条都配不上，而且**不抛异常**：覆盖率读成 0.0，
    每个指标都是 0.0，看起来和"模型什么都没学会"一模一样。
    20 条冒烟测试实测就是这个结果，而那个 adapter 的输出其实完全正常。
    """
    from data.sft.score_student import score

    golds = [_v3_gold("e1", 0, "找点周五晚上的歌"), _v3_gold("e2", 0, "推荐纯音乐")]
    ev = tmp_path / "eval.jsonl"
    ev.write_text("".join(json.dumps(g, ensure_ascii=False) + "\n" for g in golds), encoding="utf-8")

    # swift 的真实输出键：没有 meta，只有 messages + response
    preds = [{"messages": g["messages"][:-1], "response": g["messages"][-1]["content"],
              "dataset": None, "labels": None, "logprobs": None} for g in golds]
    pf = tmp_path / "pred.jsonl"
    pf.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in preds), encoding="utf-8")

    report = score(ev, pf)
    assert report["coverage"]["matched"] == 2, "没有 meta 的预测必须仍能对上 gold"
    assert report["coverage"]["coverage"] == 1.0
    assert report["coverage"]["complete"] is True
    assert report["request_kind_acc"] == 1.0, "对齐修好后，正确的预测就该得满分"


def test_alignment_still_works_when_both_sides_carry_meta(tmp_path):
    """带 meta 的老路径不能被改坏。"""
    from data.sft.score_student import score

    gold = _v3_gold("e9", 3, "随便来点")
    ev = tmp_path / "eval.jsonl"
    ev.write_text(json.dumps(gold, ensure_ascii=False) + "\n", encoding="utf-8")
    pf = tmp_path / "pred.jsonl"
    pf.write_text(json.dumps(
        {**gold, "response": gold["messages"][-1]["content"]}, ensure_ascii=False) + "\n",
        encoding="utf-8")
    report = score(ev, pf)
    assert report["coverage"]["matched"] == 1
    assert report["coverage"]["complete"] is True


def test_an_empty_thinking_prefix_does_not_destroy_the_score(tmp_path):
    """`--enable_thinking false` 靠预填一对空 <think></think> 生效。

    带着这个前缀 `json.loads` 会失败，schema_valid 归零，而发布门要求它等于 1.0。
    前缀要剥掉，但必须**如实报告**剥过。
    """
    from data.sft.score_student import score

    gold = _v3_gold("e1", 0, "找歌")
    ev = tmp_path / "eval.jsonl"
    ev.write_text(json.dumps(gold, ensure_ascii=False) + "\n", encoding="utf-8")
    pf = tmp_path / "pred.jsonl"
    pf.write_text(json.dumps({
        "messages": gold["messages"][:-1],
        "response": "<think>\n\n</think>\n\n" + gold["messages"][-1]["content"],
    }, ensure_ascii=False) + "\n", encoding="utf-8")

    report = score(ev, pf)
    assert report["schema_valid"] == 1.0, "空 think 前缀不该让合法 JSON 判成非法"
    assert report["thinking_wrapped"] == 1, "剥掉了就要报出来"
    assert report["thinking_nonempty"] == 0, "块是空的，不是真的在思考"


def test_a_thinking_block_with_content_is_reported_as_such(tmp_path):
    """真泄漏和空前缀必须区分开，否则这道门等于没有。"""
    from data.sft.score_student import score

    gold = _v3_gold("e1", 0, "找歌")
    ev = tmp_path / "eval.jsonl"
    ev.write_text(json.dumps(gold, ensure_ascii=False) + "\n", encoding="utf-8")
    pf = tmp_path / "pred.jsonl"
    pf.write_text(json.dumps({
        "messages": gold["messages"][:-1],
        "response": "<think>用户想要安静的音乐，我应该……</think>\n" + gold["messages"][-1]["content"],
    }, ensure_ascii=False) + "\n", encoding="utf-8")

    report = score(ev, pf)
    assert report["thinking_nonempty"] == 1, "有内容的 think 块是真泄漏，必须报出来"


def _lane_row(episode: str, kind: str, tools: list[str]) -> dict:
    return {
        "messages": [
            {"role": "system", "content": "决策器"},
            {"role": "user", "content": f"问题 {episode}"},
            {"role": "assistant", "content": json.dumps(
                {"request_kind": kind, "response_mode": "answer", "tool_names": tools},
                ensure_ascii=False)},
        ],
        "meta": {"episode_id": episode, "turn_id": 0},
    }


def _pred_row(gold: dict, tools: list[str]) -> dict:
    decision = json.loads(gold["messages"][-1]["content"])
    decision["tool_names"] = tools
    return {
        "messages": gold["messages"][:-1],
        "response": json.dumps(decision, ensure_ascii=False),
    }


def test_a_category_whose_gold_never_asks_for_lanes_reports_f1_as_not_applicable(tmp_path):
    """conversation 的 gold 全是空工具集。空集∩空集对 tp/fp/fn 零贡献，
    于是 100 行完全正确的表现完全看不见，整类 F1 被唯一错的那行决定为 0.0。
    正确的说法是"这个指标在这里没有定义"，并另报逐行 exact match。"""
    from data.sft.score_student import score

    golds = [_lane_row(f"c{i}", "conversation", []) for i in range(101)]
    preds = [_pred_row(g, []) for g in golds[:100]]
    preds.append(_pred_row(golds[100], ["web", "ingest"]))   # 唯一的非空误报

    ev = tmp_path / "eval.jsonl"
    ev.write_text("".join(json.dumps(g, ensure_ascii=False) + "\n" for g in golds), encoding="utf-8")
    pf = tmp_path / "pred.jsonl"
    pf.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in preds), encoding="utf-8")

    kind = score(ev, pf)["by_request_kind"]["conversation"]

    assert kind["lane_f1"] is None, "没有 gold-positive lane 时不能给出 F1 数值"
    assert kind["lane_f1_status"] == "not_applicable"
    assert kind["lane_gold_positive_rows"] == 0
    assert kind["lane_gold_positive_labels"] == 0
    assert kind["tool_set_exact_match_numerator"] == 100
    assert kind["tool_set_exact_match_denominator"] == 101
    assert kind["tool_set_exact_match"] == 0.9901


def test_two_empty_tool_sets_agreeing_is_never_counted_as_a_true_positive(tmp_path):
    """把空集命中伪造成 TP 会让 F1 依赖于"有多少行本来就不需要工具"。"""
    from data.sft.score_student import score

    golds = [_lane_row(f"e{i}", "conversation", []) for i in range(5)]
    preds = [_pred_row(g, []) for g in golds]
    ev = tmp_path / "eval.jsonl"
    ev.write_text("".join(json.dumps(g, ensure_ascii=False) + "\n" for g in golds), encoding="utf-8")
    pf = tmp_path / "pred.jsonl"
    pf.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in preds), encoding="utf-8")

    report = score(ev, pf)
    assert report["lane_gold_positive_labels"] == 0
    assert report["by_request_kind"]["conversation"]["lane_f1"] is None
    assert report["tool_set_exact_match_numerator"] == 5, "全对要如实体现在 exact match 上"


def test_a_category_with_real_gold_lanes_still_gets_a_measured_f1(tmp_path):
    """D3 只豁免"无从判定"，不豁免"判得差"。"""
    from data.sft.score_student import score

    golds = [_lane_row(f"r{i}", "recommendation", ["graph", "web"]) for i in range(4)]
    preds = [_pred_row(golds[0], ["graph", "web"]),
             _pred_row(golds[1], ["graph"]),
             _pred_row(golds[2], ["web"]),
             _pred_row(golds[3], ["graph", "web"])]
    ev = tmp_path / "eval.jsonl"
    ev.write_text("".join(json.dumps(g, ensure_ascii=False) + "\n" for g in golds), encoding="utf-8")
    pf = tmp_path / "pred.jsonl"
    pf.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in preds), encoding="utf-8")

    kind = score(ev, pf)["by_request_kind"]["recommendation"]
    assert kind["lane_f1_status"] == "measured"
    assert isinstance(kind["lane_f1"], float) and 0.0 < kind["lane_f1"] < 1.0
    assert kind["lane_gold_positive_rows"] == 4
    assert kind["lane_gold_positive_labels"] == 8
    assert kind["tool_set_exact_match_numerator"] == 2


def test_clarification_supports_are_reported_separately(tmp_path):
    """precision 的分母是预测数，recall 的分母是 gold 数，两者不是一回事。"""
    from data.sft.score_student import score

    golds = [_lane_row(f"k{i}", "recommendation", ["graph"]) for i in range(4)]
    # 第 0 条 gold 要求澄清
    g0 = json.loads(golds[0]["messages"][-1]["content"])
    g0["response_mode"] = "clarify"
    g0["clarification"] = "想要安静一点还是热闹一点？"
    g0["tool_names"] = []          # clarify 不得带工具通道
    golds[0]["messages"][-1]["content"] = json.dumps(g0, ensure_ascii=False)

    preds = []
    for i, g in enumerate(golds):
        d = json.loads(g["messages"][-1]["content"])
        if i in (0, 1, 2):        # 预测 3 次澄清，其中只有第 0 次是对的
            d["response_mode"] = "clarify"
            d["clarification"] = "想要什么风格？"
            d["tool_names"] = []          # clarify 不得带工具通道
        else:
            d["response_mode"] = "answer"
            d.pop("clarification", None)
        preds.append({"messages": g["messages"][:-1],
                      "response": json.dumps(d, ensure_ascii=False)})

    ev = tmp_path / "eval.jsonl"
    ev.write_text("".join(json.dumps(g, ensure_ascii=False) + "\n" for g in golds), encoding="utf-8")
    pf = tmp_path / "pred.jsonl"
    pf.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in preds), encoding="utf-8")

    report = score(ev, pf)
    assert report["clarification_precision_support"] == 3, "预测了 3 次澄清"
    assert report["clarification_recall_support"] == 1, "gold 里只有 1 条该澄清"
    assert report["clarification_precision_support"] != report["clarification_recall_support"]
