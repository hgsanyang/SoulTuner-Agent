from scripts.p7_smoke import run_checks


def test_p7_smoke_offline_checks_are_non_failing(tmp_path, monkeypatch):
    monkeypatch.setenv("MUSIC_FEEDBACK_DIR", str(tmp_path / "feedback"))
    report = run_checks()

    names = {row["name"] for row in report["checks"]}
    assert "public_demo_write_guard" in names
    assert "public_gradio_demo_sample" in names
    assert "dense_backend_config" in names
    assert "ranking_policy_readiness" in names
    assert report["summary"]["failed"] == 0
