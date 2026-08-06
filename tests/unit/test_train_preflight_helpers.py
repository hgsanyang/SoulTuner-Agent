"""The training preflight is the only thing standing between a typo and a
billed AMD instance, so its three helpers are tested here rather than being
discovered on the cloud box.

Each test is a situation the V4 preflight is actually exposed to: a manifest
whose recorded digest no longer matches the file, an ms-swift release that
renamed the LoRA flag, and a thinking block that survives in a form ``grep
'<think>'`` cannot see.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from data.sft.check_swift_flags import _run_help, check_flags, parse_supported_flags
from data.sft.verify_frozen_manifest import check_manifest
from data.sft.verify_infer_output import scan_row, verify

V4 = Path(__file__).resolve().parents[2] / "data" / "sft" / "v4"


# ---------------------------------------------------------------- manifest ---

def _write_split(root: Path, name: str, rows: int) -> dict:
    path = root / f"{name}.jsonl"
    decision = json.dumps(
        {
            "request_kind": "recommendation",
            "response_mode": "answer",
            "tool_names": ["graph"],
        }
    )
    path.write_text(
        "".join(
            json.dumps(
                {
                    "messages": [
                        {"role": "system", "content": "return PlannerDecisionV3"},
                        {"role": "user", "content": f"request {name} {i}"},
                        {"role": "assistant", "content": decision},
                    ],
                    "meta": {
                        "seed_source": "curated_seed",
                        "episode_id": f"{name}-{i}",
                        "turn_id": 0,
                        "request_kind": "recommendation",
                        "trajectory_kind": "single_turn",
                        "observation_origin": "none",
                        "teacher": {"model": "fixture", "version": "1"},
                        "reviewer": {"model": "fixture", "version": "1"},
                        "reviewer_verdict": "accept",
                    },
                }
            )
            + "\n"
            for i in range(rows)
        ),
        encoding="utf-8",
    )
    return {
        "path": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "rows": rows,
        "counts_by_request_kind": {"recommendation": rows},
        "counts_by_trajectory": {"single_turn": rows},
    }


@pytest.fixture
def frozen_build(tmp_path: Path) -> tuple[Path, dict]:
    manifest = {
        "manifest_version": "1.0",
        "dataset_version": "v4.0.0",
        "created_at": "2026-08-05T12:00:00Z",
        "generator_commit": "e9eb3fa",
        "splits": {
            "train": _write_split(tmp_path, "train", 8),
            "regression": _write_split(tmp_path, "regression", 4),
            "sealed": _write_split(tmp_path, "sealed", 2),
        },
        "sealed_policy": {
            "entity_disjoint": True,
            "template_disjoint": True,
            "episode_namespace": "sealed_v4",
            "seed_pool": "neo4j_unseen_entities_2026Q3",
            "measured": {
                "shared_episodes": 0,
                "shared_artists": 0,
                "shared_songs": 0,
                "shared_templates": 0,
                "max_near_dupe_jaccard": 0.16,
            },
            "release_gates": {
                "overall_vs_teacher_pp": -3.0,
                "per_kind_max_regression_pp": 3.0,
                "schema_validity": 1.0,
                "lane_authority_violations": 0,
                "sealed_vs_regression_max_gap_pp": 8.0,
            },
        },
        "validator": {
            "tool": "scripts/validate_sft_dataset.py",
            "commit": "e9eb3fa",
            "hard_findings": 0,
            "report_path": "dataset_gate.json",
        },
    }
    path = tmp_path / "MANIFEST.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path, manifest


def _rewrite(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest), encoding="utf-8")


def test_a_consistent_manifest_passes(frozen_build, tmp_path):
    path, _ = frozen_build
    code, report = check_manifest(path, root=tmp_path)
    assert code == 0, report["problems"]
    assert report["manifest_sha256"]
    assert set(report["splits"]) == {"train", "regression", "sealed"}


def test_a_digest_that_no_longer_matches_the_file_is_caught(frozen_build, tmp_path):
    """The whole point of recording a sha256: someone appends one row later."""
    path, manifest = frozen_build
    (tmp_path / "train.jsonl").open("a", encoding="utf-8").write('{"i": 999}\n')
    code, report = check_manifest(path, root=tmp_path)
    assert code == 6
    assert any("sha256 mismatch" in p for p in report["problems"])


def test_a_row_count_that_contradicts_the_class_counts_is_caught(frozen_build, tmp_path):
    path, manifest = frozen_build
    manifest["splits"]["train"]["counts_by_request_kind"] = {"recommendation": 3}
    _rewrite(path, manifest)
    code, report = check_manifest(path, root=tmp_path)
    assert code == 6
    assert any("counts_by_request_kind sums to 3" in p for p in report["problems"])


def test_a_schema_invalid_manifest_is_rejected(frozen_build, tmp_path):
    path, manifest = frozen_build
    del manifest["generator_commit"]
    _rewrite(path, manifest)
    code, report = check_manifest(path, root=tmp_path)
    assert code == 6
    assert report["schema_errors"]


def test_a_sealed_split_sharing_artists_cannot_be_waved_through(frozen_build, tmp_path):
    path, manifest = frozen_build
    manifest["sealed_policy"]["measured"]["shared_artists"] = 64
    _rewrite(path, manifest)
    code, report = check_manifest(path, root=tmp_path)
    assert code == 6
    assert any("shared_artists" in p for p in report["problems"])


def test_training_on_a_file_the_manifest_does_not_describe_is_rejected(frozen_build, tmp_path):
    """A manifest waved at a run that trains on something else is worse than none."""
    path, _ = frozen_build
    other = tmp_path / "somethingelse.jsonl"
    other.write_text('{"i": 0}\n', encoding="utf-8")
    code, report = check_manifest(path, root=tmp_path, expect_train=other)
    assert code == 6
    assert any("is not the manifest's 'train' split" in p for p in report["problems"])
    code, _ = check_manifest(path, root=tmp_path, expect_train=tmp_path / "train.jsonl")
    assert code == 0


def test_a_missing_manifest_is_unusable_not_a_pass(tmp_path):
    code, report = check_manifest(tmp_path / "nope.json")
    assert code == 4
    assert report["problems"]


def test_the_shipped_schema_is_the_one_the_checker_uses():
    assert (V4 / "MANIFEST.schema.json").is_file()


# ------------------------------------------------------------ swift flags ---

HELP = """
usage: swift sft [-h] [--model MODEL] [--train_type TRAIN_TYPE]
  --model MODEL
  --train_type TRAIN_TYPE
  --dataset DATASET
  --seed SEED
"""


def test_help_output_is_parsed_into_flag_names():
    assert {"--model", "--train_type", "--dataset", "--seed"} <= parse_supported_flags(HELP)


def test_a_renamed_lora_flag_is_reported_with_the_alias_this_build_accepts():
    code, report = check_flags(
        "sft", ["--model", "--tuner_type"], help_reader=lambda _: HELP
    )
    assert code == 10
    assert report["missing"]["--tuner_type"] == ["--train_type"]


def test_all_present_flags_pass():
    code, report = check_flags(
        "sft", ["--model", "--dataset", "--seed"], help_reader=lambda _: HELP
    )
    assert code == 0
    assert not report["missing"]


def test_a_cli_that_cannot_be_queried_is_unusable_not_a_pass():
    def boom(_):
        raise FileNotFoundError("`swift` is not on PATH")

    code, report = check_flags("sft", ["--model"], help_reader=boom)
    assert code == 4
    assert any("could not run" in p for p in report["problems"])


def test_empty_help_is_unusable_not_a_pass():
    code, _ = check_flags("sft", ["--model"], help_reader=lambda _: "")
    assert code == 4


def test_sft_help_always_uses_the_real_pipeline_parser():
    complete = "usage: sft [-h] [--model MODEL] [--dataset DATASET]"
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout=complete, stderr="")

    with (
        patch("data.sft.check_swift_flags.shutil.which", return_value="/bin/swift"),
        patch("data.sft.check_swift_flags.subprocess.run", side_effect=fake_run),
    ):
        help_text = _run_help("sft")

    assert len(calls) == 1
    assert "from swift.pipelines import sft_main; sft_main()" in calls[0]
    assert {"--model", "--dataset"} <= parse_supported_flags(help_text)


# ── 下面四条锁的是"调用方真正能传什么" ─────────────────────────────────────
# 上面那批测试全部直接给函数传 `--model`。但 argparse 的 `--flags` 是 nargs="+"，
# 它把 `--model` 读成下一个选项而不是值，所以 shell 只能传裸名。检查器却按
# `startswith("--")` 过滤，两边永远对不上：每次调用检查 0 个参数并报 OK。
# 这个空门在云端实例开始计费之后才暴露，而它本可以被下面任意一条测试拦住。


def test_bare_names_are_what_the_shell_can_pass_and_must_be_checked():
    code, report = check_flags(
        "sft", ["model", "dataset", "seed"], help_reader=lambda _: HELP
    )
    assert code == 0
    assert report["checked"] == ["--dataset", "--model", "--seed"]


def test_a_nonexistent_flag_fails_even_when_passed_bare():
    """空门的证否测试：编一个任何 ms-swift 都没有的名字，必须失败。"""
    code, report = check_flags(
        "sft", ["definitely_not_a_real_flag_xyz"], help_reader=lambda _: HELP
    )
    assert code == 10
    assert "--definitely_not_a_real_flag_xyz" in report["missing"]


def test_an_empty_flag_list_is_unusable_not_a_pass():
    """检查 0 个参数不是"全部通过"，是配置坏了。"""
    code, report = check_flags("sft", [], help_reader=lambda _: HELP)
    assert code == 4
    assert any("no flags were configured" in p for p in report["problems"])
    code, _ = check_flags("sft", ["", "-", "--"], help_reader=lambda _: HELP)
    assert code == 4


def test_the_training_script_passes_flags_in_a_form_the_checker_accepts():
    """把两个文件绑在一起：脚本怎么写的，就用那个形式跑一遍检查器。

    这条测试存在的理由是上一版两边各自"看着都对"——脚本传带横线的名字，
    检查器也只认带横线的——但中间隔着 argparse，谁都没真的跑通过一次。
    """
    import shlex

    script = (
        Path(__file__).resolve().parents[2] / "data" / "sft" / "train_planner_student.sh"
    ).read_text(encoding="utf-8")

    # 把续行拼成单行，再用 shlex 切 —— 不用正则去猜 shell 的换行规则。
    joined = script.replace("\\\n", " ")
    lines = [ln for ln in joined.splitlines() if "check_swift_flags" in ln]
    assert lines, "训练脚本里找不到 check_swift_flags 调用"

    for line in lines:
        tokens = shlex.split(line)
        assert "--flags" in tokens, f"调用缺少 --flags: {line[:80]}"
        names = tokens[tokens.index("--flags") + 1:]
        assert names, "--flags 后面一个参数都没有"
        # argparse 会把带横线的项吃成选项，脚本必须传裸名
        dashed = [n for n in names if n.startswith("-")]
        assert not dashed, f"这些名字带了横线，argparse 会当成选项而不是值: {dashed}"
        code, report = check_flags("sft", names, help_reader=lambda _: HELP)
        assert report["checked"], "脚本传的参数被过滤成空了"
        assert code in (0, 10), f"检查器无法处理脚本的调用形式: {report['problems']}"


def test_training_code_and_dataset_generator_are_recorded_separately():
    """A harness-only fix must not require regenerating an unchanged dataset."""
    script = (
        Path(__file__).resolve().parents[2] / "data" / "sft" / "train_planner_student.sh"
    ).read_text(encoding="utf-8")
    assert 'git_succeeds("cat-file", "-e", f"{generator_commit}^{{commit}}")' in script
    assert 'git_succeeds("merge-base", "--is-ancestor", generator_commit, git_sha)' in script
    assert '"dataset_generator_commit": generator_commit' in script
    assert "generator_commit != git_sha" not in script


# ------------------------------------------------------------ infer output ---

CLEAN = json.dumps(
    {"request_kind": "recommendation", "response_mode": "answer", "tool_names": ["graph"]},
    ensure_ascii=False,
)


def _pred(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "eval_predictions.jsonl"
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    return path


def test_a_clean_run_passes_and_reports_the_schema_parse_rate(tmp_path):
    code, report = verify(_pred(tmp_path, [{"response": CLEAN}] * 3))
    assert code == 0
    assert report["rows"] == 3
    assert report["schema_parse_rate"] == 1.0


@pytest.mark.parametrize(
    "row",
    [
        {"response": "<think>hmm</think>" + CLEAN},
        # the template consumed the opening tag; grep '<think>' sees nothing
        {"response": "hmm</think>" + CLEAN},
        {"response": "◁think▷hmm◁/think▷" + CLEAN},
        # the scratchpad came back beside the answer instead of inside it
        {"response": CLEAN, "reasoning_content": "hmm"},
        {"messages": [{"role": "assistant", "content": "<thinking>x</thinking>"}]},
    ],
    ids=["open_and_close", "close_only", "fullwidth", "sidecar_field", "in_messages"],
)
def test_every_way_a_thinking_block_leaks_is_caught(tmp_path, row):
    code, report = verify(_pred(tmp_path, [row]))
    assert code == 9, report
    assert report["rows_with_thinking"] == 1


def test_invalid_json_is_not_reported_as_a_thinking_failure(tmp_path):
    """A truncated payload and a leaked scratchpad look identical downstream;
    conflating them turns a max_new_tokens bug into a phantom model regression."""
    code, report = verify(_pred(tmp_path, [{"response": '{"request_kind": "recomm'}]))
    assert code == 0
    assert report["rows_with_thinking"] == 0
    assert report["schema_parse_rate"] == 0.0


def test_an_empty_prediction_file_is_unusable_not_a_pass(tmp_path):
    path = tmp_path / "eval_predictions.jsonl"
    path.write_text("", encoding="utf-8")
    code, report = verify(path)
    assert code == 4
    assert any("empty" in p for p in report["problems"])


def test_a_missing_prediction_file_is_unusable_not_a_pass(tmp_path):
    code, _ = verify(tmp_path / "absent.jsonl")
    assert code == 4


def test_the_word_think_in_prose_is_not_a_thinking_block(tmp_path):
    row = {"response": json.dumps({"request_kind": "conversation", "response_mode": "answer",
                                   "decision_summary": "I think we should just chat"},
                                  ensure_ascii=False)}
    assert not scan_row(row)
    code, _ = verify(_pred(tmp_path, [row]))
    assert code == 0


# --------------------------------------------------------------- projection ---
# 冻结的 train 分片里 lineage 不同形（5700 行两键 / 411 行三键）。Arrow struct 强类型，
# datasets 从文件头推出两键结构后读到三键行就 cast 失败，训练在加载数据集时崩溃。
# 训练只读 messages，所以派生一份只含 messages 的副本；冻结字节不动，manifest 仍成立。

from data.sft.project_chatml import main as project_main  # noqa: E402


def _chatml(tmp_path, rows):
    path = tmp_path / "frozen.jsonl"
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    return path


def _row(i, lineage):
    return {
        "messages": [
            {"role": "user", "content": f"request {i}"},
            {"role": "assistant", "content": '{"request_kind": "recommendation"}'},
        ],
        "meta": {"episode_id": f"e{i}"},
        "lineage": lineage,
    }


def test_a_heterogeneous_lineage_survives_projection(tmp_path):
    """真实形状：多数行两键，少数行三键。投影后两种行都在，且顺序不变。"""
    source = _chatml(tmp_path, [
        _row(0, {"builder": "b", "builder_version": "1"}),
        _row(1, {"builder": "b", "builder_version": "1", "clarification_trope": "vague"}),
        _row(2, {"builder": "b", "builder_version": "1"}),
    ])
    target = tmp_path / "projected.jsonl"
    assert project_main(["--source", str(source), "--target", str(target),
                         "--skip-load-check"]) == 0

    out = [json.loads(x) for x in target.read_text(encoding="utf-8").splitlines()]
    assert len(out) == 3
    assert all(set(r) == {"messages"} for r in out), "投影必须只保留 messages"
    src = [json.loads(x) for x in source.read_text(encoding="utf-8").splitlines()]
    assert [r["messages"] for r in out] == [r["messages"] for r in src], "对话内容或顺序变了"


def test_the_frozen_source_is_never_modified(tmp_path):
    """冻结文件的 SHA-256 必须原样不动，否则 manifest 立刻失效。"""
    source = _chatml(tmp_path, [_row(0, {"builder": "b", "builder_version": "1"})])
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    project_main(["--source", str(source), "--target", str(tmp_path / "p.jsonl"),
                 "--skip-load-check"])
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before


def test_a_row_without_messages_fails_closed(tmp_path):
    """少一行对话比加载报错糟得多，所以宁可失败也不静默跳过。"""
    source = _chatml(tmp_path, [{"meta": {"episode_id": "no-messages"}}])
    assert project_main(["--source", str(source), "--target", str(tmp_path / "p.jsonl"),
                         "--skip-load-check"]) == 4


def test_a_missing_source_is_not_a_pass(tmp_path):
    assert project_main(["--source", str(tmp_path / "nope.jsonl"),
                         "--target", str(tmp_path / "p.jsonl")]) == 4


def test_the_report_records_both_digests(tmp_path):
    source = _chatml(tmp_path, [_row(0, {"builder": "b", "builder_version": "1"})])
    target = tmp_path / "p.jsonl"
    report_path = tmp_path / "report.json"
    project_main(["--source", str(source), "--target", str(target),
                  "--json", str(report_path), "--skip-load-check"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert report["target_sha256"] == hashlib.sha256(target.read_bytes()).hexdigest()
    assert report["rows"] == report["kept"] == 1


def test_the_training_script_feeds_swift_the_projection_not_the_frozen_file():
    """把脚本和投影绑在一起：swift 必须吃派生副本，manifest 必须校验原文件。

    这条存在的理由和前面那条 flag 门的端到端测试一样——两边各自看着都对，
    但没人验证过它们接得上。
    """
    script = (
        Path(__file__).resolve().parents[2] / "data" / "sft" / "train_planner_student.sh"
    ).read_text(encoding="utf-8")
    assert '--dataset "$SWIFT_TRAIN_FILE"' in script
    assert '--val_dataset "$SWIFT_VAL_FILE"' in script
    assert '--expect-train "$TRAIN_FILE"' in script, "manifest 必须校验冻结原文件"
    assert "data.sft.project_chatml" in script


# --- 投影的三项加固：原子写入 / 空行不错位 / 真加载 -------------------------

def test_a_blank_line_in_the_source_does_not_shift_the_comparison(tmp_path):
    """原来 zip() 配的是原始行：源里一个空行会白白消耗掉目标的一行，
    之后每一对都错位，而校验照样报通过。"""
    from data.sft.project_chatml import verify

    source = tmp_path / "src.jsonl"
    source.write_text(
        json.dumps({"messages": [{"role": "user", "content": "A"}], "lineage": {}}) + "\n"
        + "\n"                                                    # ← 空行
        + json.dumps({"messages": [{"role": "user", "content": "B"}], "lineage": {}}) + "\n",
        encoding="utf-8",
    )
    target = tmp_path / "dst.jsonl"
    assert project_main(["--source", str(source), "--target", str(target),
                         "--skip-load-check"]) == 0
    assert verify(source, target) == []

    out = [json.loads(x) for x in target.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert [r["messages"][0]["content"] for r in out] == ["A", "B"]


def test_a_shifted_projection_is_caught(tmp_path):
    """证否：手工造一个错位的目标文件，校验必须发现。"""
    from data.sft.project_chatml import verify

    source = tmp_path / "src.jsonl"
    source.write_text(
        "".join(json.dumps({"messages": [{"role": "user", "content": c}]}) + "\n"
                for c in "ABC"), encoding="utf-8")
    bad = tmp_path / "bad.jsonl"
    bad.write_text(
        "".join(json.dumps({"messages": [{"role": "user", "content": c}]}) + "\n"
                for c in "BCA"), encoding="utf-8")
    assert verify(source, bad), "错位的投影必须被发现"


def test_no_partial_file_is_left_when_projection_fails(tmp_path):
    """半截的投影下一次跑起来看着完全正常——那是最坏的失败方式。"""
    source = tmp_path / "src.jsonl"
    source.write_text(
        json.dumps({"messages": [{"role": "user", "content": "ok"}]}) + "\n"
        + json.dumps({"meta": {"no": "messages"}}) + "\n", encoding="utf-8")
    target = tmp_path / "dst.jsonl"
    assert project_main(["--source", str(source), "--target", str(target),
                         "--skip-load-check"]) == 4
    assert not target.exists(), "失败后不该留下目标文件"
    assert list(tmp_path.glob("*.partial")) == [], "不该留下临时文件"


def test_the_projection_is_actually_loadable_by_datasets(tmp_path):
    """这条是整个投影存在的理由：SHA 和行数全绿的冻结文件恰恰是加载不了的那个。"""
    pytest.importorskip("datasets")
    from data.sft.project_chatml import load_check

    source = tmp_path / "src.jsonl"
    source.write_text(
        json.dumps({"messages": [{"role": "user", "content": "A"}],
                    "lineage": {"builder": "b", "builder_version": "1"}}) + "\n"
        + json.dumps({"messages": [{"role": "user", "content": "B"}],
                      "lineage": {"builder": "b", "builder_version": "1",
                                  "clarification_trope": "vague"}}) + "\n",
        encoding="utf-8")
    target = tmp_path / "dst.jsonl"
    assert project_main(["--source", str(source), "--target", str(target)]) == 0
    assert load_check(target, 2) == []


def test_the_load_check_reports_a_row_count_mismatch(tmp_path):
    pytest.importorskip("datasets")
    from data.sft.project_chatml import load_check

    target = tmp_path / "dst.jsonl"
    target.write_text(json.dumps({"messages": []}) + "\n", encoding="utf-8")
    assert load_check(target, 999), "行数对不上必须报出来"


# ------------------------------------------------------------ deps gate ------

from data.sft.check_training_deps import collect, read_pins  # noqa: E402


def test_pins_are_parsed_from_the_amd_requirements_file():
    pins = read_pins(Path(__file__).resolve().parents[2] / "data" / "sft" / "requirements-amd.txt")
    assert pins.get("flash-linear-attention") == "0.5.2", \
        "AMD 依赖清单必须钉死 fla 版本——9B preflight 就是缺它才崩的"


def test_a_missing_pinned_package_fails_and_says_what_to_do(tmp_path):
    req = tmp_path / "r.txt"
    req.write_text("definitely-not-installed-xyz==1.2.3\n", encoding="utf-8")
    _, problems = collect(read_pins(req))
    assert any("definitely-not-installed-xyz" in p for p in problems)
    assert any("will not install or upgrade" in p for p in problems), \
        "必须明说这个脚本不会自己装——自动升级会把 ROCm torch 换成 PyPI 版"


def test_a_wrong_version_is_not_waved_through(tmp_path):
    """装了但版本不对，比没装更危险：看起来一切正常。"""
    req = tmp_path / "r.txt"
    req.write_text("pytest==0.0.1\n", encoding="utf-8")
    _, problems = collect(read_pins(req))
    assert any("pytest is" in p and "expected 0.0.1" in p for p in problems)


def test_an_empty_requirements_file_still_records_the_runtime(tmp_path):
    req = tmp_path / "empty.txt"
    req.write_text("# nothing pinned\n", encoding="utf-8")
    report, _ = collect(read_pins(req))
    assert "torch" in report["recorded"], "即使没有 pin，也必须记录运行时版本"


def test_the_runner_binds_the_commit_through_an_env_var_not_a_hardcoded_sha():
    """脚本不该知道自己未来的 SHA——那正是上一版每次修复都要手改守卫的原因。"""
    import re
    script = (
        Path(__file__).resolve().parents[2] / "data" / "sft" / "run_planner_v4.sh"
    ).read_text(encoding="utf-8")
    assert 'EXPECTED_TRAINING_COMMIT:?' in script, "必须是必填环境变量"
    assert 'RUN_FULL="${RUN_FULL:-0}"' in script, "RUN_FULL 必须默认 0"
    for guard in ("HEAD is", "worktree is dirty", "is not an ancestor of the training code"):
        assert guard in script, f"守卫缺失: {guard}"
    # 不允许出现看起来像被硬编码的 40 位 commit（生成提交那个除外，它是数据身份）
    shas = set(re.findall(r"\b[0-9a-f]{40}\b", script)) - {
        "48d87edc3fe52d52031cbb3ad78633fc5a4e54d4"}
    assert not shas, f"训练提交不该硬编码在脚本里: {shas}"


# ------------------------------------------------- canonical prompt eval ---
# 冻结分片的 system prompt 不一致：train/regression 是 662 字符的
# STUDENT_SYSTEM_PROMPT_V3，sealed 是 77 字符的另一条。那是两条代码路径的意外，
# 不是刻意的 prompt 鲁棒性设计。但 sealed 分数因此是"模型没见过的 prompt 下"的
# 泛化，不是部署条件下的表现。派生一份换成 canonical prompt 的 eval 输入，
# gold 一字不改，两个分数的差就是 prompt 敏感度。

from data.sft.derive_canonical_prompt_eval import (  # noqa: E402
    canonical_prompt,
    main as derive_main,
)

LONG = "你是音乐智能体的决策器。" + "契约说明。" * 60
SHORT = "你是 SoulTuner 的 Planner。"


def _eval_row(i, system, extra=None):
    row = {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": f"帮我找歌 {i}"},
            {"role": "assistant", "content": '{"request_kind": "recommendation"}'},
        ],
        "meta": {"episode_id": f"e{i}", "turn_id": 1},
    }
    if extra:
        row["lineage"] = extra
    return row


def _jsonl(tmp_path, name, rows):
    path = tmp_path / name
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    return path


def test_the_sealed_prompt_is_replaced_and_the_gold_is_not(tmp_path):
    """换的是问法，不是答案。assistant 一个字节都不能动。"""
    reference = _jsonl(tmp_path, "train.jsonl", [_eval_row(i, LONG) for i in range(3)])
    source = _jsonl(tmp_path, "sealed.jsonl", [
        _eval_row(0, SHORT, {"builder": "b", "entity": "x"}),
        _eval_row(1, SHORT, {"collector": "c", "seed_kind": "k"}),
    ])
    target = tmp_path / "sealed_canonical.jsonl"
    assert derive_main(["--source", str(source), "--target", str(target),
                        "--reference", str(reference)]) == 0

    out = [json.loads(x) for x in target.read_text(encoding="utf-8").splitlines()]
    src = [json.loads(x) for x in source.read_text(encoding="utf-8").splitlines()]
    assert [r["messages"][0]["content"] for r in out] == [LONG, LONG]
    assert [r["messages"][2] for r in out] == [r["messages"][2] for r in src], "gold 被改了"
    assert [r["messages"][1] for r in out] == [r["messages"][1] for r in src], "user 被改了"
    assert [r["meta"] for r in out] == [r["meta"] for r in src], "meta 丢了，打分就对不上"
    assert [r["lineage"] for r in out] == [r["lineage"] for r in src]


def test_meta_survives_because_scoring_aligns_on_it(tmp_path):
    """score_student 用 meta.episode_id#turn_id 对齐 gold 与预测。
    丢了它，两边各自退回按 user 文本哈希，覆盖率直接归零而不是报错。"""
    reference = _jsonl(tmp_path, "train.jsonl", [_eval_row(0, LONG)])
    source = _jsonl(tmp_path, "sealed.jsonl", [_eval_row(7, SHORT)])
    target = tmp_path / "d.jsonl"
    derive_main(["--source", str(source), "--target", str(target),
                 "--reference", str(reference)])
    out = json.loads(target.read_text(encoding="utf-8").splitlines()[0])
    assert out["meta"] == {"episode_id": "e7", "turn_id": 1}


def test_the_frozen_source_is_never_rewritten(tmp_path):
    reference = _jsonl(tmp_path, "train.jsonl", [_eval_row(0, LONG)])
    source = _jsonl(tmp_path, "sealed.jsonl", [_eval_row(0, SHORT)])
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    derive_main(["--source", str(source), "--target", str(tmp_path / "d.jsonl"),
                 "--reference", str(reference)])
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before


def test_deriving_onto_the_source_is_refused(tmp_path):
    """目标就是冻结文件时必须拒绝——那会当场毁掉 manifest 身份。"""
    reference = _jsonl(tmp_path, "train.jsonl", [_eval_row(0, LONG)])
    source = _jsonl(tmp_path, "sealed.jsonl", [_eval_row(0, SHORT)])
    assert derive_main(["--source", str(source), "--target", str(source),
                        "--reference", str(reference)]) == 4


def test_a_reference_with_two_prompts_has_no_canonical_answer(tmp_path):
    """参考分片自己就有两种 prompt 时，选哪条都是掷硬币冒充测量，必须失败。"""
    reference = _jsonl(tmp_path, "mixed.jsonl", [_eval_row(0, LONG), _eval_row(1, SHORT)])
    source = _jsonl(tmp_path, "sealed.jsonl", [_eval_row(0, SHORT)])
    assert derive_main(["--source", str(source), "--target", str(tmp_path / "d.jsonl"),
                        "--reference", str(reference)]) == 4
    with pytest.raises(ValueError, match="2 distinct system prompts"):
        canonical_prompt(reference)


def test_a_row_that_does_not_start_with_system_fails_closed(tmp_path):
    """不能猜哪条是 system——猜错就是把 user 内容换成了 prompt。"""
    reference = _jsonl(tmp_path, "train.jsonl", [_eval_row(0, LONG)])
    bad = {"messages": [{"role": "user", "content": "hi"},
                        {"role": "assistant", "content": "{}"}]}
    source = _jsonl(tmp_path, "sealed.jsonl", [bad])
    assert derive_main(["--source", str(source), "--target", str(tmp_path / "d.jsonl"),
                        "--reference", str(reference)]) == 4


def test_no_partial_file_survives_a_failure(tmp_path):
    """半截的 eval 输入下一次跑起来看着完全正常，会静默只评一部分。"""
    reference = _jsonl(tmp_path, "train.jsonl", [_eval_row(0, LONG)])
    source = _jsonl(tmp_path, "sealed.jsonl", [_eval_row(0, SHORT),
                                               {"messages": [{"role": "user", "content": "x"}]}])
    target = tmp_path / "d.jsonl"
    assert derive_main(["--source", str(source), "--target", str(target),
                        "--reference", str(reference)]) == 4
    assert not target.exists(), "失败后不该留下目标文件"
    assert not list(tmp_path.glob("*.partial")), "临时文件必须清掉"


def test_the_report_records_both_prompts_and_digests(tmp_path):
    reference = _jsonl(tmp_path, "train.jsonl", [_eval_row(0, LONG)])
    source = _jsonl(tmp_path, "sealed.jsonl", [_eval_row(0, SHORT), _eval_row(1, SHORT)])
    target = tmp_path / "d.jsonl"
    report_path = tmp_path / "r.json"
    derive_main(["--source", str(source), "--target", str(target),
                 "--reference", str(reference), "--json", str(report_path)])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["rows"] == 2
    assert report["system_replaced"] == 2
    assert report["system_already_canonical"] == 0
    assert report["canonical_prompt_chars"] == len(LONG)
    assert report["source_sha256"] != report["target_sha256"]
    assert len(report["source_system_prompts"]) == 1


def test_an_already_canonical_split_is_reported_as_such_not_replaced(tmp_path):
    """regression 已经是 canonical prompt；派生它应当是恒等操作并如实说明。"""
    reference = _jsonl(tmp_path, "train.jsonl", [_eval_row(0, LONG)])
    source = _jsonl(tmp_path, "regression.jsonl", [_eval_row(0, LONG), _eval_row(1, LONG)])
    target = tmp_path / "d.jsonl"
    report_path = tmp_path / "r.json"
    assert derive_main(["--source", str(source), "--target", str(target),
                        "--reference", str(reference), "--json", str(report_path)]) == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["system_already_canonical"] == 2
    assert report["system_replaced"] == 0
