from retrieval.post_recall_adjustments import (
    DAY_MS,
    PostRecallAdjustmentConfig,
    apply_post_recall_adjustments,
    decayed_exposure_count,
    exposure_penalty,
    freshness_score,
    longtail_score,
)


NOW = 1_800_000_000_000


def _candidate(title: str, score: float, affinity: float = 0.0) -> dict:
    return {
        "song": {"title": title, "artist": "artist"},
        "similarity_score": score,
        "_rrf_score": score,
        "_graph_affinity": affinity,
    }


def test_exposure_count_decays_with_time():
    recent = decayed_exposure_count(
        7.0,
        last_exposed_at_ms=NOW,
        now_ms=NOW,
        half_life_days=7,
    )
    old = decayed_exposure_count(
        7.0,
        last_exposed_at_ms=NOW - 14 * DAY_MS,
        now_ms=NOW,
        half_life_days=7,
    )

    assert recent == 6.0
    assert round(old, 3) == 1.5
    assert exposure_penalty(old, pivot=3.0) < exposure_penalty(recent, pivot=3.0)


def test_freshness_and_longtail_are_normalised():
    fresh = freshness_score(NOW - DAY_MS, now_ms=NOW, half_life_days=21)
    stale = freshness_score(NOW - 90 * DAY_MS, now_ms=NOW, half_life_days=21)

    assert 0.9 < fresh <= 1.0
    assert 0.0 <= stale < fresh
    assert longtail_score(0.0) == 1.0
    assert longtail_score(9.0) == 0.1


def test_post_recall_delta_is_bounded_and_explainable():
    candidates = [
        _candidate("fresh cold", 0.70, affinity=1.0),
        _candidate("overexposed", 0.72, affinity=0.0),
    ]
    metadata = {
        "fresh cold": {
            "updated_at": NOW - DAY_MS,
            "ts_beta": 1.0,
            "ts_last_exposed_at": 0,
        },
        "overexposed": {
            "updated_at": NOW - 90 * DAY_MS,
            "ts_beta": 20.0,
            "ts_last_exposed_at": NOW,
        },
    }

    adjusted = apply_post_recall_adjustments(
        candidates,
        metadata_by_title=metadata,
        score_field="similarity_score",
        output_score_field="_post_final_score",
        apply_to_similarity=True,
        now_ms=NOW,
    )
    by_title = {item["song"]["title"]: item for item in adjusted}

    assert by_title["fresh cold"]["_post_recall_delta"] > 0
    assert by_title["overexposed"]["_post_recall_delta"] < 0
    assert abs(by_title["fresh cold"]["_post_recall_delta"]) <= 0.08
    assert "post_recall_adjustments" in by_title["fresh cold"]["song"]


def test_post_recall_delta_limit_prevents_adjustment_from_dominating():
    candidates = [
        _candidate("strong base", 0.90, affinity=0.0),
        _candidate("weak but fresh", 0.60, affinity=1.0),
    ]
    metadata = {
        "weak but fresh": {
            "updated_at": NOW,
            "ts_beta": 1.0,
        }
    }
    config = PostRecallAdjustmentConfig(delta_limit=0.04)

    adjusted = apply_post_recall_adjustments(
        candidates,
        metadata_by_title=metadata,
        apply_to_similarity=True,
        config=config,
        now_ms=NOW,
    )

    by_title = {item["song"]["title"]: item for item in adjusted}
    assert by_title["weak but fresh"]["similarity_score"] <= 0.64
    assert by_title["strong base"]["similarity_score"] >= 0.86
