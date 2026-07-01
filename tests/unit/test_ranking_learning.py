import json

from services.ranking_learning import (
    build_preference_pairs,
    build_strict_labeled_rows,
    learn_ranking_policy,
)
from services.ranking_policy import (
    ACTIVE_FILE,
    CANDIDATE_FILE,
    PREVIOUS_FILE,
    promote_candidate,
    rollback_policy,
    runtime_policy_for_user,
    write_candidate,
)


def _exposure(exposure_id: str, ts: int, title: str, good: bool, user_id: str = "u1"):
    high = 0.9 if good else 0.1
    low = 0.1 if good else 0.9
    return {
        "exposure_id": exposure_id,
        "ts": ts,
        "user_id": user_id,
        "intent_type": "hybrid_search",
        "items": [
            {
                "title": title,
                "artist": "Singer",
                "rank": 1,
                "source_ranks": {"dense": 1 if good else 20, "lexical": 20 if good else 1},
                "semantic_score": high,
                "acoustic_score": high,
                "personal_score": high,
                "freshness_score": high,
                "longtail_score": high,
                "exposure_penalty": low,
            },
            {
                "title": f"Untouched {title}",
                "artist": "Singer",
                "rank": 2,
                "source_ranks": {"graph": 2},
                "semantic_score": 0.5,
                "acoustic_score": 0.5,
            },
        ],
    }


def _event(exposure_id: str | None, ts: int, title: str, positive: bool, user_id: str = "u1"):
    return {
        "event_id": f"event-{title}",
        "exposure_id": exposure_id,
        "ts": ts,
        "user_id": user_id,
        "event_type": "like" if positive else "skip",
        "title": title,
        "artist": "Singer",
    }


def test_strict_join_never_matches_event_without_exposure_id():
    exposures = [_exposure("e1", 100, "A", True)]
    events = [_event(None, 110, "A", True)]

    rows, diagnostics = build_strict_labeled_rows(exposures, events)

    assert rows == []
    assert diagnostics["missing_exposure_id"] == 1


def test_unobserved_items_are_not_implicit_negatives():
    exposures = [_exposure("e1", 100, "A", True)]
    events = [_event("e1", 110, "A", True)]

    rows, diagnostics = build_strict_labeled_rows(exposures, events)

    assert diagnostics["matched_events"] == 1
    assert len(rows) == 1
    assert rows[0]["title"] == "A"


def test_preference_pairs_require_explicit_positive_and_negative_in_same_slate():
    exposure = _exposure("e1", 100, "A", True)
    exposure["items"][1]["title"] = "B"
    events = [
        _event("e1", 110, "A", True),
        _event("e1", 120, "B", False),
    ]
    rows, _ = build_strict_labeled_rows([exposure], events)

    pairs = build_preference_pairs(rows)

    assert len(pairs) == 1
    assert pairs[0]["positive"]["title"] == "A"
    assert pairs[0]["negative"]["title"] == "B"


def test_v2_learner_uses_chronological_validation_and_accepts_clear_signal():
    exposures = []
    events = []
    for index in range(40):
        positive = index % 2 == 0
        exposure_id = f"e{index}"
        title = f"Song {index}"
        exposures.append(_exposure(exposure_id, index * 1000, title, positive))
        events.append(_event(exposure_id, index * 1000 + 100, title, positive))

    report = learn_ranking_policy(
        exposures,
        events,
        min_events=20,
        per_user_min_events=20,
        validation_ratio=0.25,
    )

    assert report["gate_passed"] is True
    assert report["global"]["status"] == "accepted"
    assert report["global"]["validation_events"] == 10
    assert report["global"]["learned_validation"]["log_loss"] <= report["global"]["baseline_validation"]["log_loss"]
    assert report["users"]["u1"]["status"] == "accepted"


def test_policy_candidate_promotion_and_rollback(tmp_path, monkeypatch):
    monkeypatch.setenv("MUSIC_FEEDBACK_DIR", str(tmp_path))
    accepted = {
        "schema_version": 2,
        "status": "candidate_accepted",
        "gate_passed": True,
        "global": {
            "status": "accepted",
            "runtime_policy": {"rrf_multipliers": {"graph": 1.1}},
        },
        "users": {},
    }
    write_candidate(accepted, tmp_path)
    assert (tmp_path / CANDIDATE_FILE).exists()

    promote_candidate(tmp_path)
    assert runtime_policy_for_user("u1")["rrf_multipliers"]["graph"] == 1.1

    previous = dict(accepted)
    previous["status"] = "active"
    previous["global"] = {
        "status": "accepted",
        "runtime_policy": {"rrf_multipliers": {"graph": 0.9}},
    }
    (tmp_path / PREVIOUS_FILE).write_text(json.dumps(previous), encoding="utf-8")
    rollback_policy(tmp_path)
    active = json.loads((tmp_path / ACTIVE_FILE).read_text(encoding="utf-8"))
    assert active["global"]["runtime_policy"]["rrf_multipliers"]["graph"] == 0.9
