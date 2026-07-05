from services.feedback_diagnostics import summarize_feedback_quality


def _exposure(exposure_id: str = "e1"):
    return {
        "exposure_id": exposure_id,
        "ts": 1000,
        "items": [
            {
                "title": "A",
                "artist": "Singer",
                "rank": 1,
                "source_ranks": {"dense": 1},
                "semantic_score": 0.9,
                "acoustic_score": 0.8,
                "personal_score": 0.2,
            },
            {
                "title": "B",
                "artist": "Singer",
                "rank": 2,
                "source_ranks": {"graph": 1},
                "semantic_score": 0.2,
                "acoustic_score": 0.3,
                "personal_score": 0.9,
            },
        ],
    }


def test_feedback_quality_reports_blockers_without_logs():
    report = summarize_feedback_quality([], [], [], min_events=2)

    assert report["ready_for_replay"] is False
    assert "no_exposures" in report["blockers"]
    assert "missing_positive_labels" in report["blockers"]
    assert "missing_negative_labels" in report["blockers"]


def test_feedback_quality_counts_matched_rows_and_pairs():
    exposures = [_exposure()]
    events = [
        {"event_id": "p1", "event_type": "like", "title": "A", "artist": "Singer", "exposure_id": "e1", "ts": 1100},
        {"event_id": "n1", "event_type": "dislike", "title": "B", "artist": "Singer", "exposure_id": "e1", "ts": 1200},
    ]

    report = summarize_feedback_quality(exposures, events, [], min_events=2)

    assert report["ready_for_replay"] is True
    assert report["matched_explicit_rows"] == 2
    assert report["positive_rows"] == 1
    assert report["negative_rows"] == 1
    assert report["same_exposure_pairs"] == 1
    assert report["event_type_counts"] == {"like": 1, "dislike": 1}


def test_feedback_quality_reports_unmatched_feedback():
    exposures = [_exposure()]
    events = [
        {"event_id": "n1", "event_type": "skip", "title": "Missing", "artist": "Singer", "exposure_id": "e1", "ts": 1200},
        {"event_id": "n2", "event_type": "skip", "title": "A", "artist": "Singer", "exposure_id": "unknown", "ts": 1200},
    ]
    slate_feedback = [{"feedback_id": "s1", "rating": "too_noisy", "exposure_id": "unknown", "ts": 1300}]

    report = summarize_feedback_quality(exposures, events, slate_feedback, min_events=2)

    assert report["matched_explicit_rows"] == 0
    assert report["explicit_diagnostics"]["song_not_in_exposure"] == 1
    assert report["explicit_diagnostics"]["unknown_exposure_id"] == 1
    assert report["slate_diagnostics"]["unknown_exposure_id"] == 1
    assert report["ready_for_replay"] is False
