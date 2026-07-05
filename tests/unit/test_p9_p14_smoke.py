from scripts.p9_p14_smoke import run_checks


def test_p9_p14_smoke_checks_are_non_failing():
    report = run_checks()

    names = {row["name"] for row in report["checks"]}
    assert "context_pressure_cases" in names
    assert "catalog_gap_release_fallback" in names
    assert "slate_feedback_log" in names
    assert "tag_policy_cap" in names
    assert "ui_slate_feedback_api" in names
    assert report["summary"]["failed"] == 0

