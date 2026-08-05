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
