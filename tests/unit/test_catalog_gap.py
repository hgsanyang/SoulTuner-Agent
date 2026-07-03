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
