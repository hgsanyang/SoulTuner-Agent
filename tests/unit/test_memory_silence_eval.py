"""Validate the memory silence evaluator on the open fixture (MV3-5)."""

from pathlib import Path

from tests.eval.evaluate_memory_silence import DEFAULT_CASES, evaluate

REPO = Path(__file__).resolve().parents[2]


def test_silence_fixture_is_open_synthetic():
    import json

    data = json.loads(DEFAULT_CASES.read_text(encoding="utf-8"))
    assert data["sealed"] is False
    assert "no user, production" in data["provenance"]


def test_silence_evaluator_meets_dod_on_fixture():
    report = evaluate(DEFAULT_CASES)
    s = report["summary"]
    # every hand-authored case should be decided correctly
    assert s["passed"] == s["cases"]
    assert s["injection_precision"] >= 0.90
    assert s["over_injection_rate"] <= 0.05
    assert s["silence_appropriateness"] == 1.0


def test_adversarial_cases_stay_silent():
    report = evaluate(DEFAULT_CASES)
    by_id = {row["id"]: row for row in report["rows"]}
    for silent_case in (
        "silent_irrelevant_memory",
        "silent_wrong_scene",
        "silent_current_constraint_override",
        "silent_expired_episode",
    ):
        assert by_id[silent_case]["selected"] == [], silent_case


def test_contradicted_case_injects_only_the_newer_memory():
    report = evaluate(DEFAULT_CASES)
    by_id = {row["id"]: row for row in report["rows"]}
    row = by_id["silent_contradicted_memory"]
    assert row["selected"] == ["m_new"]
    assert row["silence_decision"]["suppressed_contradicted"] == 1
