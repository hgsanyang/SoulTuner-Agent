from data.sft.benchmark_planner_endpoint import summarise
from data.sft.check_planner_release import check_release
from data.sft.compare_planner_scores import compare, compare_split_gap


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
