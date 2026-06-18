"""
单元测试：结果导向评测尺子的打分逻辑（tests/eval/evaluate_outcomes.py）。

只测纯逻辑（_unwrap_songs / evaluate_case / 各 check），不拉起 Agent，
因此无需 langchain/neo4j，可在 CI 跑。"尺子"本身也需要一把尺子。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tests.eval.evaluate_outcomes import _unwrap_songs, _is_degraded, evaluate_case


def _song(title, artist, genre="", playable=True, **extra):
    song = {"title": title, "artist": artist, "genre": genre,
            "preview_url": "http://x/a.mp3" if playable else None}
    song.update(extra)
    return {"song": {"title": title, "artist": artist, "genre": genre,
                     "preview_url": "http://x/a.mp3" if playable else None, **extra}}


def _result(songs, intent="graph_search", errors=None):
    return {"recommendations": songs, "intent_type": intent, "errors": errors or []}


# ---------------------------------------------------------------- _unwrap_songs
def test_unwrap_list():
    r = _result([_song("晴天", "周杰伦"), _song("稻香", "周杰伦")])
    songs = _unwrap_songs(r)
    assert len(songs) == 2 and songs[0]["title"] == "晴天"


def test_unwrap_tooloutput():
    class ToolOutput:  # 模拟 retrieve() 的 ToolOutput.data
        def __init__(self, data): self.data = data
    r = {"recommendations": ToolOutput([_song("A", "x")]), "intent_type": "vector_search", "errors": []}
    assert len(_unwrap_songs(r)) == 1


def test_unwrap_drops_titleless():
    r = _result([{"song": {"artist": "x"}}, _song("A", "y")])
    assert len(_unwrap_songs(r)) == 1  # 无 title 的被丢弃


# ---------------------------------------------------------------- _is_degraded
def test_degraded_flag():
    assert _is_degraded(_result([], errors=[{"node": "analyze_intent", "degraded_to": "vector_search"}]))
    assert not _is_degraded(_result([_song("A", "x")]))


# ---------------------------------------------------------------- evaluate_case
def test_artist_case_passes():
    case = {"id": "t", "query": "周杰伦的歌", "checks": {
        "min_results": 3,
        "artist_any_of": ["周杰伦", "Jay Chou"], "artist_match_min_ratio": 0.7,
        "min_playable_ratio": 0.5, "not_degraded": True}}
    songs = [_song("晴天", "周杰伦", "pop"), _song("稻香", "周杰伦", "pop"),
             _song("七里香", "周杰伦", "", playable=False)]
    rep = evaluate_case(case, _result(songs))
    assert rep["case_status"] == "pass", rep["outcomes"]


def test_not_degraded_fails_when_degraded():
    case = {"id": "t", "query": "q", "checks": {"min_results": 1, "not_degraded": True}}
    rep = evaluate_case(case, _result([_song("A", "x")],
                                      errors=[{"node": "analyze_intent", "degraded_to": "vector_search"}]))
    assert rep["case_status"] == "fail"
    assert any(o["name"] == "not_degraded" and o["status"] == "fail" for o in rep["outcomes"])


def test_artist_ratio_fails_when_wrong_artist():
    case = {"id": "t", "query": "周杰伦的歌", "checks": {
        "artist_any_of": ["周杰伦"], "artist_match_min_ratio": 0.7}}
    songs = [_song("A", "陈奕迅"), _song("B", "林俊杰"), _song("晴天", "周杰伦")]
    rep = evaluate_case(case, _result(songs))
    assert rep["case_status"] == "fail"  # 1/3 < 0.7


def test_genre_skips_when_no_genre_data():
    case = {"id": "t", "query": "英文摇滚", "checks": {
        "genre_any_of": ["rock"], "genre_match_min_ratio": 0.4}}
    songs = [_song("A", "x", genre=""), _song("B", "y", genre="")]
    rep = evaluate_case(case, _result(songs))
    # 全是 skip → 无可判定的硬 check → indeterminate（不冤枉为 fail）
    assert rep["case_status"] == "indeterminate"
    assert all(o["status"] == "skip" for o in rep["outcomes"])


def test_language_ratio_passes_with_transmitted_field():
    case = {"id": "t", "query": "安静的日语歌", "checks": {
        "language_any_of": ["Japanese"], "language_match_min_ratio": 0.7}}
    songs = [
        _song("A", "x", language="Japanese"),
        _song("B", "y", language="Japanese"),
        _song("C", "z", language="Japanese"),
    ]
    rep = evaluate_case(case, _result(songs))
    assert rep["case_status"] == "pass"
    assert any(o["name"] == "language_match_min_ratio" and o["status"] == "pass" for o in rep["outcomes"])


def test_language_ratio_skips_when_no_language_data():
    case = {"id": "t", "query": "日语歌", "checks": {
        "language_any_of": ["Japanese"], "language_match_min_ratio": 0.7}}
    rep = evaluate_case(case, _result([_song("A", "x"), _song("B", "y")]))
    assert rep["case_status"] == "indeterminate"
    assert all(o["status"] == "skip" for o in rep["outcomes"])


def test_mood_and_scenario_ratio_use_list_fields():
    case = {"id": "t", "query": "跑步热血", "checks": {
        "mood_any_of": ["Energetic"], "mood_match_min_ratio": 0.5,
        "scenario_any_of": ["Workout"], "scenario_match_min_ratio": 0.5}}
    songs = [
        _song("A", "x", moods=["Energetic"], scenarios=["Workout"]),
        _song("B", "y", moods=["Peaceful"], scenarios=["Study"]),
    ]
    rep = evaluate_case(case, _result(songs))
    assert rep["case_status"] == "pass", rep["outcomes"]


def test_must_exclude_violation_fails():
    case = {"id": "t", "query": "q", "checks": {
        "min_results": 1, "must_exclude": [{"title": "差歌", "artist": "坏歌手"}]}}
    songs = [_song("好歌", "好歌手"), _song("差歌", "坏歌手")]
    rep = evaluate_case(case, _result(songs))
    assert rep["case_status"] == "fail"


def test_must_exclude_template_skips():
    case = {"id": "t", "query": "q", "checks": {
        "min_results": 1, "must_exclude": [{"title": "示例-请替换为你不喜欢的歌名", "artist": "示例歌手"}]}}
    rep = evaluate_case(case, _result([_song("好歌", "好歌手")]))
    # 占位模板 → must_exclude 跳过；min_results 通过 → 整体 pass
    assert rep["case_status"] == "pass"
    assert any(o["name"] == "must_exclude" and o["status"] == "skip" for o in rep["outcomes"])


def test_max_per_artist_fails():
    case = {"id": "t", "query": "q", "checks": {"max_per_artist": 3}}
    songs = [_song(f"s{i}", "同一歌手") for i in range(4)]
    rep = evaluate_case(case, _result(songs))
    assert rep["case_status"] == "fail"


def test_must_include_titles():
    case = {"id": "t", "query": "林俊杰的江南", "checks": {"must_include_titles": ["江南"]}}
    assert evaluate_case(case, _result([_song("江南", "林俊杰")]))["case_status"] == "pass"
    assert evaluate_case(case, _result([_song("曹操", "林俊杰")]))["case_status"] == "fail"


def test_min_playable_ratio():
    case = {"id": "t", "query": "q", "checks": {"min_playable_ratio": 0.6}}
    songs = [_song("A", "x", playable=True), _song("B", "y", playable=True), _song("C", "z", playable=False)]
    assert evaluate_case(case, _result(songs))["case_status"] == "pass"  # 2/3 ≈ 67%
    songs2 = [_song("A", "x", playable=True), _song("B", "y", playable=False), _song("C", "z", playable=False)]
    assert evaluate_case(case, _result(songs2))["case_status"] == "fail"  # 1/3 ≈ 33%


def test_manual_review_not_scored():
    case = {"id": "t", "query": "安静的日语歌", "checks": {
        "min_results": 1, "manual_review": ["确认是日语", "确认安静"]}}
    rep = evaluate_case(case, _result([_song("A", "x")]))
    assert rep["case_status"] == "pass"
    assert rep["manual_review"] == ["确认是日语", "确认安静"]
