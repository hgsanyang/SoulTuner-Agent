import asyncio

from agent.netease_query import (
    artist_matches,
    build_netease_query_plan,
    clean_natural_query,
    fetch_json_with_retry,
    parse_play_url_payload,
)


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


def test_parse_play_url_payload_keeps_only_playable_rows_and_trial_flags():
    urls, trials = parse_play_url_payload({
        "data": [
            {"id": 1, "url": "https://audio.example/1.mp3", "freeTrialInfo": None},
            {"id": 2, "url": None, "freeTrialInfo": None},
            {"id": 3, "url": "https://audio.example/3.mp3", "freeTrialInfo": {"end": 30}},
        ]
    })

    assert urls == {
        "1": "https://audio.example/1.mp3",
        "3": "https://audio.example/3.mp3",
    }
    assert trials == {"1": False, "3": True}


def test_parse_play_url_payload_handles_empty_response():
    assert parse_play_url_payload(None) == ({}, {})


def test_fetch_json_with_retry_recovers_from_transient_failure():
    retries = []

    class Response:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def json(self):
            return {"ok": True}

    class FailingResponse(Response):
        async def __aenter__(self):
            raise TimeoutError("temporary")

    class Session:
        def __init__(self):
            self.calls = 0

        def get(self, _url, timeout):
            assert timeout == 15
            self.calls += 1
            return FailingResponse() if self.calls == 1 else Response()

    session = Session()
    result = asyncio.run(fetch_json_with_retry(
        session,
        "http://example.test/search",
        timeout=15,
        retry_delay=0,
        on_retry=lambda attempt, exc: retries.append((attempt, type(exc).__name__)),
    ))

    assert result == {"ok": True}
    assert session.calls == 2
    assert retries == [(1, "TimeoutError")]
