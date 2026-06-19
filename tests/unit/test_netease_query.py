from agent.netease_query import artist_matches, build_netease_query_plan, clean_natural_query


def test_artist_only_query_prefers_chinese_artist_entity():
    plan = build_netease_query_plan(
        user_input="周杰伦的歌",
        fallback_query="周杰伦 Jay Chou",
        retrieval_plan={"graph_artist_entities": ["周杰伦", "Jay Chou"], "graph_song_entities": []},
    )
    assert plan.mode == "search"
    assert plan.query == "周杰伦"
    assert plan.artist_terms == ("周杰伦", "Jay Chou")


def test_artist_song_query_prefers_chinese_artist_and_song():
    plan = build_netease_query_plan(
        user_input="林俊杰的那首江南",
        fallback_query="林俊杰 JJ Lin 江南 Jiang Nan",
        retrieval_plan={"graph_artist_entities": ["林俊杰", "JJ Lin"], "graph_song_entities": ["江南", "Jiang Nan"]},
    )
    assert plan.query == "林俊杰 江南"


def test_recent_new_song_query_uses_new_song_mode():
    plan = build_netease_query_plan(user_input="最近有什么新歌", fallback_query="")
    assert plan.mode == "new_songs"
    assert plan.query == "华语新歌榜"


def test_clean_natural_query_removes_request_suffix():
    assert clean_natural_query("帮我找周杰伦的歌") == "周杰伦"


def test_artist_matches_tolerates_punctuation_and_aliases():
    assert artist_matches("周杰伦-", ("周杰伦", "Jay Chou"))
    assert artist_matches("Jay Chou", ("周杰伦", "Jay Chou"))
    assert not artist_matches("蔡依林", ("周杰伦", "Jay Chou"))
