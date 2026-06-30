from agent.retrieval_fallback import (
    avoid_terms,
    decide_online_fallback,
    fallback_query,
    filter_results_by_avoid,
    filter_results_by_requested_language,
)


def _plan(*, artists=None, songs=None, language=None, avoid=None):
    return {
        "hard_constraints": {
            "artist_entities": artists or [],
            "song_entities": songs or [],
            "language": language,
        },
        "soft_intent": {"avoid": avoid or []},
    }


def test_empty_explicit_inventory_triggers_web_independent_of_intent_label():
    decision = decide_online_fallback([], _plan(artists=["周杰伦", "Jay Chou"]))
    assert decision.required
    assert decision.reason == "local_inventory_empty"


def test_empty_unconstrained_inventory_does_not_force_web():
    assert not decide_online_fallback([], _plan()).required


def test_missing_exact_song_and_weak_artist_results_trigger_web():
    weak_artist = [{"song": {"title": "A", "artist": "其他歌手"}}]
    assert decide_online_fallback(weak_artist, _plan(artists=["林俊杰"])).reason == "local_artist_match_insufficient"
    wrong_song = [{"song": {"title": "Sunny", "artist": "Bobby Hebb"}}]
    assert decide_online_fallback(wrong_song, _plan(songs=["晴天", "Sunny Day"])).reason == "local_song_match_missing"


def test_similarity_reference_song_does_not_force_web_fallback():
    local_results = [{"song": {"title": "Monday Morning", "artist": "Pulp"}}]
    plan = _plan(songs=["Running Up That Hill"])
    plan["soft_intent"] = {"goal": "找听感相似的歌", "vibe": "类似合成器流行", "avoid": []}

    decision = decide_online_fallback(local_results, plan, "有没有类似听感的歌")

    assert not decision.required


def test_explicit_song_allows_title_suffix_but_not_truncated_alias():
    live_version = [{"song": {"title": "晴天 (Live)", "artist": "歌手"}}]
    assert not decide_online_fallback(live_version, _plan(songs=["晴天"])).required


def test_literal_avoid_filters_title_and_artist_but_not_subjective_words():
    rows = [
        {"song": {"title": "十年", "artist": "陈奕迅"}},
        {"song": {"title": "最佳损友", "artist": "陈奕迅"}},
        {"song": {"title": "晴天", "artist": "周杰伦"}},
        {"song": {"title": "晴天", "artist": "Lucky小爱"}},
    ]
    kept, excluded = filter_results_by_avoid(rows, ["十年", "周杰伦", "不要苦情"])
    assert excluded == 2
    assert [(row["song"]["title"], row["song"]["artist"]) for row in kept] == [
        ("最佳损友", "陈奕迅"),
        ("晴天", "Lucky小爱"),
    ]


def test_conversational_artist_avoid_extracts_literal_core():
    rows = [
        {"song": {"title": "晴天", "artist": "周杰伦.、Asasblue"}},
        {"song": {"title": "晴天", "artist": "Lucky小爱"}},
    ]
    kept, excluded = filter_results_by_avoid(rows, ["周杰伦那首"])

    assert excluded == 1
    assert kept[0]["song"]["artist"] == "Lucky小爱"


def test_raw_query_negative_phrase_is_used_as_avoid_term():
    rows = [
        {"song": {"title": "晴天 (原唱 周杰伦)", "artist": "RyaVocal"}},
        {"song": {"title": "晴天", "artist": "周杰伦.、Asasblue"}},
        {"song": {"title": "晴天", "artist": "Lucky小爱"}},
    ]
    terms = avoid_terms(_plan(songs=["晴天"]), "想听晴天，但不是周杰伦那首")

    kept, excluded = filter_results_by_avoid(
        rows,
        terms,
    )

    assert "周杰伦" in terms
    assert excluded == 2
    assert kept == [{"song": {"title": "晴天", "artist": "Lucky小爱"}}]


def test_sparse_language_inventory_triggers_web_but_sufficient_matches_do_not():
    sparse = [
        {"song": {"title": "A", "artist": "x", "language": "Japanese"}},
        {"song": {"title": "B", "artist": "y", "language": "Unknown"}},
    ]
    sufficient = [
        {"song": {"title": "A", "artist": "x", "language": "Cantonese"}},
        {"song": {"title": "B", "artist": "y", "language": "粤语"}},
        {"song": {"title": "C", "artist": "z", "language": "Cantonese"}},
        {"song": {"title": "D", "artist": "q", "language": "Chinese"}},
    ]

    decision = decide_online_fallback(sparse, _plan(language="Korean"))
    assert decision.required
    assert decision.reason == "local_language_match_insufficient"
    assert not decide_online_fallback(sufficient, _plan(language="Cantonese")).required


def test_korean_web_filter_requires_hangul_and_adds_inferred_language():
    rows = [
        {"name": "그리운 사람", "artists": [{"name": "박흥우"}]},
        {"name": "韩语抒情版", "artists": [{"name": "中文歌手"}]},
    ]

    kept, excluded = filter_results_by_requested_language(rows, "Korean")

    assert excluded == 1
    assert kept == [
        {
            "name": "그리운 사람",
            "artists": [{"name": "박흥우"}],
            "_inferred_language": "Korean",
        }
    ]


def test_song_only_avoid_query_prefers_cover_search():
    assert fallback_query(_plan(songs=["晴天"], avoid=["周杰伦"]), "fallback") == "晴天 翻唱"


def test_language_only_fallback_uses_searchable_catalog_terms():
    assert fallback_query(_plan(language="Korean"), "fallback") == "韩国 抒情歌"
