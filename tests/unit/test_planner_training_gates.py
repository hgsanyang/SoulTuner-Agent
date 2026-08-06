import json
from pathlib import Path
import tarfile

from data.sft.benchmark_planner_endpoint import summarise
from data.sft.check_planner_release import check_release
from data.sft.compare_planner_scores import compare, compare_split_gap
from data.sft import export_v4_training_bundle


def _score(value: float = 1.0) -> dict:
    return {
        "coverage": {"complete": True},
        "schema_valid": value,
        "compilable": value,
        "request_kind_acc": value,
        "lane_f1": value,
        "clarification_precision": value,
        "clarification_recall": value,
        "hyde_present_when_dense": value,
        "lane_authority_violations": 0,
        "by_request_kind": {
            "recommendation": {
                "rows": 100,
                "schema_valid": value,
                "request_kind_acc": value,
                "lane_f1": value,
            }
        },
    }


def test_contrast_gate_accepts_small_quality_delta():
    baseline = _score(0.95)
    candidate = _score(0.93)
    baseline["schema_valid"] = 1.0
    candidate["schema_valid"] = 1.0
    baseline["by_request_kind"]["recommendation"]["schema_valid"] = 1.0
    candidate["by_request_kind"]["recommendation"]["schema_valid"] = 1.0

    report = compare(baseline, candidate, tolerance=0.03)

    assert report["passed"]


def test_contrast_gate_rejects_category_regression_and_invalid_schema():
    baseline = _score(1.0)
    candidate = _score(1.0)
    candidate["schema_valid"] = 0.99
    candidate["by_request_kind"]["recommendation"]["lane_f1"] = 0.8

    report = compare(baseline, candidate, tolerance=0.03)

    assert not report["passed"]
    assert any("schema_valid must be 1.0" in finding for finding in report["findings"])
    assert any("recommendation.lane_f1" in finding for finding in report["findings"])


def test_endpoint_benchmark_reports_percentiles_without_prompt_text():
    records = [
        {"sample_id": "a#0", "ok": True, "schema_valid": True, "latency_ms": 100},
        {"sample_id": "b#0", "ok": True, "schema_valid": True, "latency_ms": 200},
        {"sample_id": "c#0", "ok": True, "schema_valid": True, "latency_ms": 300},
    ]

    report = summarise(records, max_p50_ms=250)

    assert report["gate"]["passed"]
    assert report["latency_ms"]["p50"] == 200
    assert "messages" not in report


def test_sealed_split_gap_rejects_unseen_entity_collapse():
    regression = _score(0.95)
    sealed = _score(0.95)
    sealed["lane_f1"] = 0.80

    report = compare_split_gap(regression, sealed, tolerance=0.08)

    assert not report["passed"]
    assert any("lane_f1" in finding for finding in report["findings"])


def test_endpoint_benchmark_fails_on_schema_error_or_slow_p50():
    records = [
        {"sample_id": "a#0", "ok": True, "schema_valid": True, "latency_ms": 3000},
        {"sample_id": "b#0", "ok": False, "schema_valid": False, "latency_ms": 100, "error": "HTTPError"},
    ]

    report = summarise(records, max_p50_ms=2000)

    assert not report["gate"]["passed"]
    assert report["failures"] == [{"sample_id": "b#0", "error": "HTTPError"}]


def test_frozen_release_gate_rejects_a_pair_that_is_jointly_weak():
    manifest = {
        "dataset_version": "v4.0.0",
        "sealed_policy": {
            "release_gates": {
                "overall_vs_teacher_pp": -3.0,
                "per_kind_max_regression_pp": 3.0,
                "schema_validity": 1.0,
                "lane_authority_violations": 0,
                "sealed_vs_regression_max_gap_pp": 8.0,
            }
        },
    }
    regression = _score(0.90)
    sealed = _score(0.90)
    regression["schema_valid"] = sealed["schema_valid"] = 1.0
    regression["compilable"] = sealed["compilable"] = 1.0

    report = check_release(manifest, regression, sealed)

    assert not report["passed"]
    assert any("request_kind_acc" in finding for finding in report["findings"])


def test_frozen_release_gate_checks_clarification_when_the_split_has_support():
    manifest = {
        "dataset_version": "v4.0.0",
        "sealed_policy": {
            "release_gates": {
                "overall_vs_teacher_pp": -3.0,
                "per_kind_max_regression_pp": 3.0,
                "schema_validity": 1.0,
                "lane_authority_violations": 0,
                "sealed_vs_regression_max_gap_pp": 8.0,
            }
        },
    }
    regression = _score()
    sealed = _score()
    regression["clarification_gold_cases"] = 3
    sealed["clarification_gold_cases"] = 22
    sealed["clarification_recall"] = 0.90

    report = check_release(manifest, regression, sealed)

    assert not report["passed"]
    assert any("sealed.clarification_recall" in finding for finding in report["findings"])


def test_bundle_export_resolves_relative_paths_from_project_root(tmp_path, monkeypatch):
    private_dir = tmp_path / "data" / "teacher" / "private" / "v4"
    private_dir.mkdir(parents=True)
    split_path = private_dir / "train.jsonl"
    split_path.write_text("{}\n", encoding="utf-8")
    manifest_path = private_dir / "MANIFEST.json"
    manifest_path.write_text(
        json.dumps({"splits": {"train": {"path": "data/teacher/private/v4/train.jsonl"}}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(export_v4_training_bundle, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        export_v4_training_bundle,
        "check_manifest",
        lambda manifest, root: (export_v4_training_bundle.EXIT_OK, {"problems": []}),
    )

    report = export_v4_training_bundle.export_bundle(
        Path("data/teacher/private/v4/MANIFEST.json"),
        Path("data/teacher/private/v4/export.tar.gz"),
    )

    archive_path = tmp_path / report["archive"] if not Path(report["archive"]).is_absolute() else Path(report["archive"])
    with tarfile.open(archive_path, "r:gz") as archive:
        assert archive.getnames() == [
            "data/teacher/private/v4/MANIFEST.json",
            "data/teacher/private/v4/train.jsonl",
        ]


# --- 小样本类别：报告而不是硬门 ----------------------------------------------
# regression 里 acquisition/library/conversation 各只有 1 条。n=1 时 per-kind
# 指标只能是 0.0 或 1.0：判错一次直接卡住发布，判对也证明不了 3pp 的测量能力。
# 这类类别改为 canary 报告；sealed 五类各 100 条，继续按统计硬门执行。

def _frozen_manifest():
    return {
        "dataset_version": "v4.0.0",
        "sealed_policy": {
            "release_gates": {
                "overall_vs_teacher_pp": -3.0,
                "per_kind_max_regression_pp": 3.0,
                "schema_validity": 1.0,
                "lane_authority_violations": 0,
                "sealed_vs_regression_max_gap_pp": 8.0,
            }
        },
    }


def _healthy(value: float = 1.0, *, supports: dict[str, int] | None = None) -> dict:
    score = _score(value)
    score["schema_valid"] = 1.0
    score["compilable"] = 1.0
    for kind, support in (supports or {}).items():
        score.setdefault("by_request_kind", {}).setdefault(kind, {})
        score["by_request_kind"][kind].update(
            {"request_kind_acc": value, "lane_f1": value, "support": support}
        )
    return score


def test_a_single_row_category_in_regression_is_a_canary_not_a_hard_gate():
    regression = _healthy(supports={"acquisition": 1})
    regression["by_request_kind"]["acquisition"]["lane_f1"] = 0.0   # n=1 判错一次
    sealed = _healthy()

    report = check_release(_frozen_manifest(), regression, sealed)

    assert not any("acquisition" in f for f in report["findings"]), \
        "n=1 的类别不该卡住发布"
    assert any("acquisition" in c for c in report["low_support_canaries"]), \
        "但必须报出来，不能装作没看见"


def test_a_well_supported_regression_category_still_fails_hard():
    regression = _healthy(supports={"recommendation": 218})
    regression["by_request_kind"]["recommendation"]["lane_f1"] = 0.5
    sealed = _healthy()

    report = check_release(_frozen_manifest(), regression, sealed)

    assert not report["passed"]
    assert any("recommendation" in f and "support=218" in f for f in report["findings"])


def test_sealed_is_always_a_hard_gate_regardless_of_support():
    """sealed 是唯一允许卡发布的切分；即便某类支持数被误记成很小也不豁免。"""
    sealed = _healthy(supports={"library": 3})
    sealed["by_request_kind"]["library"]["request_kind_acc"] = 0.4
    report = check_release(_frozen_manifest(), _healthy(), sealed)

    assert not report["passed"]
    assert any("sealed.library" in f for f in report["findings"])
    assert not any("sealed.library" in c for c in report["low_support_canaries"])


def test_the_report_states_every_category_support():
    regression = _healthy(supports={"acquisition": 1, "recommendation": 218})
    report = check_release(_frozen_manifest(), regression, _healthy())

    supports = report["support_by_request_kind"]["regression"]
    assert supports["acquisition"] == 1 and supports["recommendation"] == 218, \
        "读报告的人必须能看到每类的样本量，否则无从判断一个数字值多少"
    assert report["thresholds"]["min_support_for_statistical_gate"] == 20
