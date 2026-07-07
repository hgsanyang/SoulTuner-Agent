from retrieval.hybrid_retrieval import rerank_with_soft_constraints


def _candidate(title, genres=None, moods=None, scenarios=None, score=0.8):
    return {
        "song": {
            "title": title,
            "artist": "artist",
            "genres": genres or [],
            "moods": moods or [],
            "scenarios": scenarios or [],
            "genre": "/".join((genres or [])[:2] + (moods or [])[:1] + (scenarios or [])[:1]),
        },
        "similarity_score": score,
        "_rrf_score": score,
    }


def test_soft_avoid_is_noop_without_avoid_terms():
    candidates = [
        _candidate("dance", ["Dance", "K-Pop"], ["Energetic"], ["Party"]),
        _candidate("ballad", ["Ballad"], ["Melancholy"], ["Late Night"]),
    ]

    assert rerank_with_soft_constraints(candidates, {}, query_text="韩语抒情慢歌") == candidates


def test_dance_avoid_prefers_korean_ballad_candidates():
    candidates = [
        _candidate("dance pop", ["Dance", "K-Pop"], ["Energetic"], ["Party"], 0.95),
        _candidate("kpop ballad", ["K-Pop", "Ballad"], ["Melancholy"], ["Late Night"], 0.90),
        _candidate("clean ballad", ["Ballad", "Folk"], ["Melancholy", "Romantic"], ["Late Night"], 0.70),
        _candidate("rnb slow", ["R&B", "Soul"], ["Melancholy", "Relaxing"], ["Late Night"], 0.65),
        _candidate("acoustic", ["Acoustic", "Indie"], ["Peaceful", "Romantic"], ["Relaxing"], 0.60),
    ]

    ranked = rerank_with_soft_constraints(
        candidates,
        {"avoid": ["Dance", "Energetic", "Party"]},
        query_text="韩语抒情慢歌，别给我打歌舞曲",
        min_keep=3,
    )

    assert [item["song"]["title"] for item in ranked] == [
        "kpop ballad",
        "clean ballad",
        "rnb slow",
        "acoustic",
    ]
    assert all("dance" not in item.get("_soft_negative_hits", []) for item in ranked)


def test_soft_avoid_does_not_trigger_from_original_query_text():
    candidates = [
        _candidate("dance pop", ["Dance", "K-Pop"], ["Energetic"], ["Party"], 0.95),
        _candidate("clean ballad", ["Ballad", "Folk"], ["Melancholy"], ["Late Night"], 0.60),
        _candidate("rnb slow", ["R&B", "Soul"], ["Melancholy"], ["Late Night"], 0.55),
        _candidate("acoustic", ["Acoustic", "Indie"], ["Peaceful"], ["Relaxing"], 0.50),
    ]

    ranked = rerank_with_soft_constraints(
        candidates,
        {"avoid": ["过于偶像团体"]},
        query_text="韩语抒情慢歌，别给我打歌舞曲",
        min_keep=3,
    )

    assert [item["song"]["title"] for item in ranked] == [
        "dance pop",
        "clean ballad",
        "rnb slow",
        "acoustic",
    ]


def test_pop_and_kpop_tags_are_not_dance_avoid_hits_by_themselves():
    candidates = [
        _candidate("plain pop", ["Pop", "R&B"], ["Romantic"], ["Late Night"], 0.90),
        _candidate("kpop", ["K-Pop", "R&B"], ["Romantic"], ["Late Night"], 0.95),
        _candidate("dance pop", ["Dance", "K-Pop"], ["Energetic"], ["Party"], 0.85),
        _candidate("folk", ["Folk"], ["Peaceful"], ["Relaxing"], 0.50),
    ]

    ranked = rerank_with_soft_constraints(
        candidates,
        {"avoid": ["dance"]},
        query_text="avoid dance beats",
        min_keep=5,
    )

    by_title = {item["song"]["title"]: item for item in ranked}
    assert by_title["plain pop"]["_soft_negative_hits"] == []
    assert by_title["kpop"]["_soft_negative_hits"] == []
    assert set(by_title["dance pop"]["_soft_negative_hits"]) == {"dance"}


def test_soft_avoid_falls_back_to_penalized_order_when_too_few_clean_candidates():
    candidates = [
        _candidate("dance pop", ["Dance", "K-Pop"], ["Energetic"], ["Party"], 0.95),
        _candidate("kpop ballad", ["K-Pop", "Ballad"], ["Melancholy"], ["Late Night"], 0.90),
        _candidate("clean ballad", ["Ballad"], ["Melancholy"], ["Late Night"], 0.50),
    ]

    ranked = rerank_with_soft_constraints(
        candidates,
        {"avoid": ["Dance", "Energetic", "Party"]},
        query_text="韩语抒情慢歌，别给我打歌舞曲",
        min_keep=5,
    )

    assert len(ranked) == 3
    assert ranked[0]["song"]["title"] == "kpop ballad"
    assert ranked[-1]["song"]["title"] == "dance pop"


def test_quiet_soft_request_needs_llm_avoid_to_demote_high_energy_conflicts():
    candidates = [
        _candidate("wild heart", ["Dance", "Rock"], ["Energetic"], ["Driving"], 0.98),
        _candidate("rain room", ["Alternative"], ["Dreamy", "Healing"], ["Rainy Day"], 0.78),
        _candidate("soft folk", ["Folk", "Acoustic"], ["Peaceful"], ["Study"], 0.74),
        _candidate("late night", ["Pop"], ["Melancholy"], ["Late Night"], 0.70),
        _candidate("quiet lofi", ["Lo-Fi"], ["Relaxing"], ["Study"], 0.68),
    ]

    ranked = rerank_with_soft_constraints(
        candidates,
        {"vibe": "柔软安静的雨天感觉"},
        {"scenario": "Rainy Day", "mood": "Calm"},
        query_text="今天是下雨天，需要偏柔软安静的感觉",
        min_keep=3,
    )

    titles = [item["song"]["title"] for item in ranked]
    assert titles[0] == "wild heart"

    fallback_ranked = rerank_with_soft_constraints(
        candidates,
        {"vibe": "柔软安静的雨天感觉", "avoid": ["Dance", "Energetic", "Driving"]},
        {"scenario": "Rainy Day", "mood": "Calm"},
        query_text="今天是下雨天，需要偏柔软安静的感觉",
        min_keep=6,
    )

    by_title = {item["song"]["title"]: item for item in fallback_ranked}
    assert set(by_title["wild heart"]["_soft_conflict_hits"]) >= {"dance", "energetic", "driving"}
    assert by_title["rain room"]["_soft_positive_hits"]


def test_low_dynamic_request_needs_llm_avoid_to_demote_noisy_and_hardcore_tags():
    candidates = [
        _candidate("noisy phonk", ["Phonk", "Hardcore Hip-Hop"], ["Energetic"], ["Workout"], 0.96),
        _candidate("soft room", ["Folk", "Acoustic"], ["Mellow", "Peaceful"], ["Rainy Day"], 0.70),
        _candidate("warm lofi", ["Lo-Fi"], ["Relaxing", "Warm"], ["Study"], 0.68),
        _candidate("gentle night", ["Indie"], ["Soft", "Dreamy"], ["Late Night"], 0.66),
    ]

    ranked = rerank_with_soft_constraints(
        candidates,
        {"vibe": "低动态，不刺耳，温柔一点"},
        {"scenario": "Rainy Day"},
        query_text="下雨天，低动态，不刺耳，温柔一点",
        min_keep=3,
    )

    titles = [item["song"]["title"] for item in ranked]
    assert titles[0] == "noisy phonk"

    fallback_ranked = rerank_with_soft_constraints(
        candidates,
        {"vibe": "低动态，不刺耳，温柔一点", "avoid": ["Phonk", "Hardcore", "Energetic", "Workout"]},
        {"scenario": "Rainy Day"},
        query_text="下雨天，低动态，不刺耳，温柔一点",
        min_keep=8,
    )
    noisy = {item["song"]["title"]: item for item in fallback_ranked}["noisy phonk"]
    assert set(noisy["_soft_conflict_hits"]) >= {"phonk", "hardcore", "energetic", "workout"}
