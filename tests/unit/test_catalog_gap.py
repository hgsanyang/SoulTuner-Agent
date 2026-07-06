from agent.catalog_gap import analyze_catalog_gap, interleave_online_results, unwrap_recommendation_items
from agent.retrieval_fallback import FallbackDecision
from agent.web_discovery import build_web_discovery_query, extract_song_candidates


def _plan(**hard):
    return {
        "hard_constraints": hard,
        "soft_intent": {"goal": "想听温暖的歌", "vibe": "classic"},
        "hints": {"genres": ["Pop"]},
    }


def test_release_year_gap_triggers_discovery_fallback():
    local = [{"song": {"title": "A", "artist": "x", "language": "Chinese"}} for _ in range(12)]

    decision = analyze_catalog_gap(
        local,
        _plan(language="Chinese"),
        "推荐80年代的中文老歌",
        web_enabled=True,
    )

    assert decision.action == "fallback"
    assert decision.discovery_required
    assert "metadata_release_year_missing" in decision.reasons


def test_gap_is_blocked_when_web_is_disabled():
    decision = analyze_catalog_gap(
        [],
        _plan(language="Korean"),
        "推荐几首韩语歌",
        web_enabled=False,
        fallback_decision=FallbackDecision(True, "local_inventory_empty", 0),
    )

    assert decision.action == "blocked"
    assert "打开" in decision.message


def test_normal_catalog_with_web_enabled_mix_in():
    local = [
        {"song": {"title": f"Song {i}", "artist": "A", "preview_url": "u"}}
        for i in range(12)
    ]

    decision = analyze_catalog_gap(local, _plan(), "想听安静一点的歌", web_enabled=True)

    assert decision.action == "mix_in"
    assert decision.target_web_count == 4


def test_background_music_is_not_external_knowledge_fallback():
    local = [
        {"song": {"title": f"Rain {i}", "artist": "A", "preview_url": "u"}}
        for i in range(12)
    ]

    decision = analyze_catalog_gap(
        local,
        _plan(),
        "雨夜看书，想要温柔背景音乐，不要太燃也不要派对感",
        web_enabled=True,
    )

    assert decision.action == "mix_in"
    assert "external_knowledge_required" not in decision.reasons
    assert decision.discovery_required is False


def test_song_background_request_still_uses_external_discovery():
    local = [
        {"song": {"title": f"Song {i}", "artist": "A", "preview_url": "u"}}
        for i in range(12)
    ]

    decision = analyze_catalog_gap(
        local,
        _plan(),
        "查一下晴天这首歌的创作背景和代表作资料",
        web_enabled=True,
    )

    assert decision.action == "fallback"
    assert "external_knowledge_required" in decision.reasons
    assert decision.discovery_required is True


def test_soft_genre_gap_mixes_more_online_candidates():
    local = [
        {"song": {"title": f"Rock {i}", "artist": "A", "preview_url": "u", "genres": ["Rock", "Folk"]}}
        for i in range(12)
    ]

    decision = analyze_catalog_gap(
        local,
        {"hard_constraints": {}, "soft_intent": {}, "hints": {"genres": ["R&B"]}},
        "想听韩语 R&B，别太吵",
        web_enabled=True,
    )

    assert decision.action == "mix_in"
    assert "local_genres_match_insufficient" in decision.reasons
    assert decision.target_web_count == 6
    assert decision.details["tag_evidence"]["genres"]["matched"] == 0


def test_query_inferred_rnb_gap_mixes_online_candidates_without_hint():
    local = [
        {"song": {"title": f"Rock {i}", "artist": "A", "preview_url": "u", "genres": ["Rock", "Folk"]}}
        for i in range(12)
    ]

    decision = analyze_catalog_gap(
        local,
        {"hard_constraints": {}, "soft_intent": {}, "hints": {}},
        "想听韩语 R&B，别太吵",
        web_enabled=True,
    )

    assert decision.action == "mix_in"
    assert "local_genres_match_insufficient" in decision.reasons
    assert decision.details["tag_evidence"]["genres"]["requested"] == ["R&B"]


def test_language_inventory_gap_triggers_fallback_when_evidence_conflicts():
    local = [
        {"song": {"title": f"Mandarin {i}", "artist": "A", "preview_url": "u", "language": "Chinese"}}
        for i in range(12)
    ]

    decision = analyze_catalog_gap(
        local,
        {"hard_constraints": {}, "soft_intent": {}, "hints": {}},
        "想听粤语港乐",
        web_enabled=True,
    )

    assert decision.action == "fallback"
    assert "local_language_match_insufficient" in decision.reasons
    assert decision.details["language_evidence"]["requested"] == "cantonese"


def test_language_gap_does_not_fire_when_catalog_has_match():
    local = [
        {"song": {"title": f"Canto {i}", "artist": "A", "preview_url": "u", "language": "Cantonese"}}
        for i in range(12)
    ]

    decision = analyze_catalog_gap(
        local,
        {"hard_constraints": {}, "soft_intent": {}, "hints": {}},
        "想听粤语港乐",
        web_enabled=True,
    )

    assert "local_language_match_insufficient" not in decision.reasons
    assert decision.details["language_evidence"]["matched"] == 12


def test_soft_genre_gap_is_explained_when_web_disabled():
    local = [
        {"song": {"title": f"Rock {i}", "artist": "A", "preview_url": "u", "genres": ["Rock"]}}
        for i in range(12)
    ]

    decision = analyze_catalog_gap(
        local,
        {"hard_constraints": {}, "soft_intent": {}, "hints": {"genres": ["R&B"]}},
        "想听 R&B",
        web_enabled=False,
    )

    assert decision.action == "blocked"
    assert "local_genres_match_insufficient" in decision.reasons
    assert "打开联网搜索" in decision.message


def test_soft_mood_aliases_do_not_trigger_false_gap():
    local = [
        {"song": {"title": f"Quiet {i}", "artist": "A", "preview_url": "u", "moods": ["Peaceful", "Relaxing"]}}
        for i in range(12)
    ]
    plan = {"hard_constraints": {}, "soft_intent": {}, "hints": {"mood": "平静"}}

    decision = analyze_catalog_gap(
        local,
        plan,
        "想听安静一点的歌",
        web_enabled=True,
    )

    assert decision.action == "mix_in"
    assert decision.reasons == ("online_exploration",)
    assert decision.details["tag_evidence"]["moods"]["matched"] == 12


def test_interleave_online_results_keeps_target_length_and_dedupes():
    local = [{"song": {"title": f"L{i}", "artist": "A"}} for i in range(8)]
    online = [
        {"song": {"title": "W1", "artist": "B", "source": "online_search"}},
        {"song": {"title": "W2", "artist": "B", "source": "online_search"}},
    ]

    merged = interleave_online_results(local, online, target_len=len(local), first_slot=2, stride=3)

    assert len(merged) == len(local)
    assert any(row["song"]["title"] == "W1" for row in merged)
    assert any(row["song"].get("source") == "online_search" for row in merged)


def test_unwrap_recommendation_items_accepts_raw_list_and_tool_output_like_object():
    rows = [{"song": {"title": "A"}}]

    class ToolOutputLike:
        data = rows

    assert unwrap_recommendation_items(rows) == rows
    assert unwrap_recommendation_items(ToolOutputLike()) == rows
    assert unwrap_recommendation_items({"not": "a list"}) == []


def test_extract_song_candidates_from_web_snippets():
    text = """
    1. 崔健的《一无所有》通常被视作 1986 年中文摇滚代表作。
    2. 《恋曲1980》 - 罗大佑，是华语流行音乐里的经典作品。
    """

    candidates = extract_song_candidates(text)

    assert candidates[0].title == "一无所有"
    assert candidates[0].artist == "崔健"
    assert any(candidate.title == "恋曲1980" for candidate in candidates)


def test_build_web_discovery_query_carries_context():
    query = build_web_discovery_query(
        "80年代的中文老歌",
        _plan(language="Chinese"),
        {"reasons": ["metadata_release_year_missing"]},
    )

    assert "80年代" in query
    assert "发行年份" in query
    assert "Chinese" in query
