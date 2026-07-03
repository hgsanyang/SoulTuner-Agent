from services.ranking_policy import summarize_policy_readiness


def test_readiness_collects_feedback_before_minimum_signals():
    report = summarize_policy_readiness(
        num_exposures=2,
        num_events=3,
        num_slate_feedback=1,
        min_events=10,
    )

    assert report["stage"] == "collect_feedback"
    assert report["can_replay"] is False
    assert report["labeled_signals"] == 4


def test_readiness_allows_replay_after_minimum_signals():
    report = summarize_policy_readiness(
        num_exposures=4,
        num_events=8,
        num_slate_feedback=3,
        min_events=10,
    )

    assert report["stage"] == "replay_ready"
    assert report["can_replay"] is True
    assert report["can_promote"] is False


def test_readiness_marks_gate_passed_candidate_promotable():
    report = summarize_policy_readiness(
        num_exposures=4,
        num_events=20,
        num_slate_feedback=0,
        candidate={"gate_passed": True, "global_status": "accepted"},
    )

    assert report["stage"] == "candidate_ready"
    assert report["can_promote"] is True


def test_readiness_prefers_active_policy_stage():
    report = summarize_policy_readiness(
        num_exposures=4,
        num_events=20,
        num_slate_feedback=0,
        active={"status": "active", "gate_passed": True, "global_status": "accepted"},
        candidate={"gate_passed": True, "global_status": "accepted"},
    )

    assert report["stage"] == "active_policy"
    assert report["has_active_policy"] is True
