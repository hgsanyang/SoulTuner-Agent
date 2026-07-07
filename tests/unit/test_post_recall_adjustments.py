from retrieval.post_recall_adjustments import (
    DAY_MS,
    PostRecallAdjustmentConfig,
    apply_post_recall_adjustments,
    decayed_exposure_count,
    exposure_penalty,
    freshness_score,
    longtail_score,
    semantic_fit_scores,
)


NOW = 1_800_000_000_000


def _candidate(title: str, score: float, affinity: float = 0.0, **song_extra) -> dict:
    song = {"title": title, "artist": "artist"}
    song.update(song_extra)
    return {
        "song": song,
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


def test_semantic_fit_uses_llm_plan_terms_not_query_triggers():
    calm_song = {"genres": ["Lo-Fi"], "moods": ["Peaceful"], "scenarios": ["Rainy Day"]}
    energetic_song = {"genres": ["Dance"], "moods": ["Energetic"], "scenarios": ["Driving"]}

    query_only = semantic_fit_scores(calm_song, query_text="下雨天，柔软安静一点")
    calm = semantic_fit_scores(calm_song, hints={"genres": ["Lo-Fi"], "mood": "Peaceful", "scenario": "Rainy Day"})
    conflict = semantic_fit_scores(energetic_song, soft_intent={"avoid": ["Dance", "Driving", "Energetic"]})

    assert query_only["active"] is False
    assert calm["positive"] > 0
    assert calm["conflict"] == 0
    assert conflict["conflict"] > 0
    assert {"Dance", "Driving", "Energetic"} & set(conflict["conflict_hits"])


def test_post_recall_semantic_context_nudges_near_ties_but_stays_bounded():
    candidates = [
        _candidate("dance conflict", 0.82, genres=["Dance"], moods=["Energetic"], scenarios=["Driving"]),
        _candidate("rain fit", 0.80, genres=["Lo-Fi"], moods=["Peaceful"], scenarios=["Rainy Day"]),
        _candidate("content best", 0.98, genres=["Dance"], moods=["Energetic"], scenarios=["Driving"]),
    ]

    adjusted = apply_post_recall_adjustments(
        candidates,
        hints={"genres": ["Lo-Fi"], "mood": "Peaceful", "scenario": "Rainy Day"},
        soft_intent={"avoid": ["Dance", "Energetic", "Driving"]},
        apply_to_similarity=True,
        now_ms=NOW,
    )
    by_title = {item["song"]["title"]: item for item in adjusted}

    assert by_title["rain fit"]["_post_semantic_positive_score"] > 0
    assert by_title["dance conflict"]["_post_semantic_conflict_score"] > 0
    assert by_title["rain fit"]["similarity_score"] > by_title["dance conflict"]["similarity_score"]
    assert by_title["content best"]["similarity_score"] > by_title["rain fit"]["similarity_score"]
    assert max(abs(item["_post_recall_delta"]) for item in adjusted) <= 0.08


def test_post_recall_semantic_context_does_not_infer_conflicts_from_plain_query():
    candidate = _candidate(
        "road song",
        0.8,
        genres=["Rock"],
        moods=["Energetic"],
        scenarios=["Driving"],
    )

    adjusted = apply_post_recall_adjustments(
        [candidate],
        query_text="开车路上听，温柔但要有公路感",
        now_ms=NOW,
    )

    assert adjusted[0]["_post_semantic_conflict_hits"] == []
