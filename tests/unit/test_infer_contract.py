"""Fail-closed guards around `swift infer`, tested against the shape that broke.

The 9B evaluation handed ms-swift 226 rows and got 452 back: the same questions,
in the same order, answered twice with different generations. swift reported
`num_samples: 226`, so nothing it printed said anything was wrong. Splitting the
file in half and scoring each was a workaround, and the wrong shape of one — it
silently picks which of two different answers counts. These tests pin the
refusal instead.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

# --------------------------------------------- D7: infer contract guards ---
# 9B 实测：226 行输入 → 452 行输出（同样 226 题、同样顺序、两次不同生成），
# 而 swift 自报 num_samples=226、进度条也只走 226 次。当时是靠打分器的
# coverage.complete=false 才发现的。拆成两半分别打分是绕过，不是修复 ——
# 它在替人决定"两个不同答案里哪个算数"。这里改成直接拒绝并说明原因。

from data.sft.check_infer_contract import check_contract, main as contract_main, reserve


def _rows_file(path: Path, episodes: list[str]) -> Path:
    path.write_text(
        "".join(
            json.dumps({
                "messages": [
                    {"role": "user", "content": f"问题 {e}"},
                    {"role": "assistant", "content": "{}"},
                ],
                "meta": {"episode_id": e, "turn_id": 0},
            }, ensure_ascii=False) + "\n"
            for e in episodes
        ),
        encoding="utf-8",
    )
    return path


def test_the_exact_452_over_226_shape_is_refused(tmp_path):
    """真实发生过的形状：每一行都被回答了两次。必须非零退出，不得拆半继续。"""
    episodes = [f"e{i}" for i in range(226)]
    src = _rows_file(tmp_path / "in.jsonl", episodes)
    pred = _rows_file(tmp_path / "pred.jsonl", episodes + episodes)

    code = contract_main(["--input", str(src), "--pred", str(pred)])
    assert code == 8, "452/226 必须失败退出"

    report = check_contract(src, pred)
    assert report["one_to_one"] is False
    assert report["pred_rows"] == 452 and report["input_rows"] == 226
    assert report["repeated"] == 226
    assert any("did not answer it once" in p for p in report["problems"]), \
        "报错要说清楚为什么不能靠删行解决"


def test_one_to_one_predictions_pass(tmp_path):
    episodes = [f"e{i}" for i in range(10)]
    src = _rows_file(tmp_path / "in.jsonl", episodes)
    pred = _rows_file(tmp_path / "pred.jsonl", episodes)
    assert contract_main(["--input", str(src), "--pred", str(pred)]) == 0
    report = check_contract(src, pred)
    assert report["one_to_one"] is True
    assert len(report["input_sha256"]) == 64
    assert len(report["pred_sha256"]) == 64
    assert report["input_sha256"] == hashlib.sha256(src.read_bytes()).hexdigest()
    assert report["pred_sha256"] == hashlib.sha256(pred.read_bytes()).hexdigest()


def test_a_short_prediction_file_is_refused(tmp_path):
    """被腰斩的推理会留下一个看起来正常的文件。"""
    src = _rows_file(tmp_path / "in.jsonl", [f"e{i}" for i in range(10)])
    pred = _rows_file(tmp_path / "pred.jsonl", [f"e{i}" for i in range(7)])
    assert contract_main(["--input", str(src), "--pred", str(pred)]) == 8
    report = check_contract(src, pred)
    assert report["missing"] == 3


def test_predictions_for_questions_that_were_not_asked_are_refused(tmp_path):
    src = _rows_file(tmp_path / "in.jsonl", ["a", "b"])
    pred = _rows_file(tmp_path / "pred.jsonl", ["a", "b", "zzz"])
    report = check_contract(src, pred)
    assert report["unmatched"] == 1
    assert report["one_to_one"] is False


def test_an_existing_result_path_is_never_written_to(tmp_path):
    """复用结果路径正是"被杀掉的运行 + 新运行"拼成怪文件的来源。"""
    target = tmp_path / "preds.jsonl"
    assert reserve(target) == [], "不存在的路径应当放行"
    target.write_text("{}\n", encoding="utf-8")
    assert reserve(target), "已存在的路径必须拒绝"
    assert contract_main(["--reserve", str(target)]) == 8


def test_the_decoding_parameters_are_written_into_the_run_record(tmp_path):
    """六周后回看一个分数，必须能知道它是用什么解码参数跑出来的。"""
    episodes = ["a", "b"]
    src = _rows_file(tmp_path / "in.jsonl", episodes)
    pred = _rows_file(tmp_path / "pred.jsonl", episodes)
    out = tmp_path / "record.json"
    code = contract_main([
        "--input", str(src), "--pred", str(pred),
        "--record", '{"do_sample": false, "seed": 42, "max_new_tokens": 1024}',
        "--json", str(out),
    ])
    assert code == 0
    record = json.loads(out.read_text(encoding="utf-8"))
    assert record["inference_params"] == {
        "do_sample": False, "seed": 42, "max_new_tokens": 1024}
    assert record["one_to_one"] is True
