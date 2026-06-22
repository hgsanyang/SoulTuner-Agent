from agent.retrieval_fallback import (
    decide_online_fallback,
    fallback_query,
    filter_results_by_avoid,
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


def test_song_only_avoid_query_prefers_cover_search():
    assert fallback_query(_plan(songs=["晴天"], avoid=["周杰伦"]), "fallback") == "晴天 翻唱"
