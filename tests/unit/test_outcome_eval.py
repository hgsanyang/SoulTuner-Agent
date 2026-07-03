"""
单元测试：结果导向评测尺子的打分逻辑（tests/eval/evaluate_outcomes.py）。

只测纯逻辑（_unwrap_songs / evaluate_case / 各 check），不拉起 Agent，
因此无需 langchain/neo4j，可在 CI 跑。"尺子"本身也需要一把尺子。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tests.eval.evaluate_outcomes import (
    _case_difficulty,
    _dimension_status,
    _is_degraded,
    _load_cases,
    _percentile,
    _summarize_timings,
    _top_slow_cases,
    _unwrap_songs,
    evaluate_case,
)


def _song(title, artist, genre="", playable=True, **extra):
    song = {"title": title, "artist": artist, "genre": genre,
            "preview_url": "http://x/a.mp3" if playable else None}
    song.update(extra)
    return {"song": {"title": title, "artist": artist, "genre": genre,
                     "preview_url": "http://x/a.mp3" if playable else None, **extra}}


def _result(songs, intent="graph_search", errors=None):
    return {"recommendations": songs, "intent_type": intent, "errors": errors or []}


def test_timing_percentiles_and_summary():
    assert _percentile([10, 20, 30, 40], 0.5) == 25
    summary = _summarize_timings([
        {"timings_ms": {"end_to_end_ms": 100, "intent_ms": 20}},
        {"timings_ms": {"end_to_end_ms": 200, "intent_ms": 40}},
    ])
    assert summary["stages"]["end_to_end_ms"] == {
        "count": 2,
        "mean": 150.0,
        "p50": 150.0,
        "p95": 195.0,
    }


def test_top_slow_cases_orders_by_end_to_end_ms():
    rows = _top_slow_cases([
        {"id": "fast", "case_status": "pass", "timings_ms": {"end_to_end_ms": 10}},
        {"id": "slow", "case_status": "fail", "timings_ms": {"end_to_end_ms": 200}},
        {"id": "missing", "case_status": "pass", "timings_ms": {"intent_ms": 1}},
    ], limit=1)
    assert rows == [{
        "id": "slow",
        "query": None,
        "case_status": "fail",
        "end_to_end_ms": 200.0,
        "intent_ms": None,
        "retrieval_total_ms": None,
        "graphzep_ms": None,
    }]


def test_case_difficulty_views_are_derived_without_duplicate_files():
    assert _case_difficulty({"category": "explicit_artist", "checks": {}}) == "easy"
    assert _case_difficulty({"category": "scenario", "checks": {}}) == "hard"
    assert _case_difficulty({"difficulty": "easy", "category": "scenario", "checks": {}}) == "easy"
    assert _case_difficulty({"category": "explicit_song", "chat_history": [{"role": "user", "content": "x"}]}) == "hard"


def test_dimension_status_separates_intent_and_ranking_layers():
    outcomes = [
        {"name": "artist_match_min_ratio", "status": "pass", "detail": ""},
        {"name": "objective_soft_judge", "status": "fail", "detail": ""},
    ]
    assert _dimension_status(outcomes, "intent") == "pass"
    assert _dimension_status(outcomes, "ranking") == "fail"


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


def test_objective_soft_judge_passes_on_song_attributes_only():
    case = {"id": "t", "query": "睡前平静一点", "checks": {
        "objective_soft_judge": {
            "positive_any": ["peaceful", "sleep", "relaxing"],
            "negative_any": ["energetic", "party"],
            "min_positive_ratio": 0.5,
            "max_negative_ratio": 0.25,
        }}}
    songs = [
        _song("A", "x", moods=["Peaceful"], scenarios=["Sleep"]),
        _song("B", "y", moods=["Dreamy"], scenarios=["Relaxing"]),
        _song("C", "z", moods=["Melancholy"], scenarios=["Late Night"]),
    ]
    rep = evaluate_case(case, _result(songs))
    assert rep["case_status"] == "pass", rep["outcomes"]


def test_objective_soft_judge_fails_on_negative_tags():
    case = {"id": "t", "query": "不要太吵", "checks": {
        "objective_soft_judge": {
            "positive_any": ["peaceful"],
            "negative_any": ["energetic", "party"],
            "min_positive_ratio": 0.2,
            "max_negative_ratio": 0.25,
        }}}
    songs = [
        _song("A", "x", moods=["Energetic"], scenarios=["Party"]),
        _song("B", "y", moods=["Energetic"], scenarios=["Workout"]),
        _song("C", "z", moods=["Peaceful"], scenarios=["Sleep"]),
    ]
    rep = evaluate_case(case, _result(songs))
    assert rep["case_status"] == "fail"
    assert any(o["name"] == "objective_soft_judge" and o["status"] == "fail" for o in rep["outcomes"])


def test_objective_soft_judge_skips_without_objective_fields():
    case = {"id": "t", "query": "q", "checks": {
        "objective_soft_judge": {"positive_any": ["peaceful"], "min_positive_ratio": 0.5}}}
    rep = evaluate_case(case, _result([_song("A", "x"), _song("B", "y")]))
    assert rep["case_status"] == "indeterminate"
    assert rep["outcomes"][0]["status"] == "skip"


def test_expected_clarification_passes_without_songs():
    case = {"id": "t", "query": "有没有类似听感的", "checks": {"expected_clarification": True}}
    result = {
        "recommendations": [],
        "intent_type": "clarification",
        "final_response": "你想按哪一种方向继续？",
        "clarification_options": ["告诉我一首参考歌"],
    }
    rep = evaluate_case(case, result)

    assert rep["case_status"] == "pass"
    assert rep["num_songs"] == 0


def test_expected_clarification_fails_when_agent_guesses():
    case = {"id": "t", "query": "有没有类似听感的", "checks": {"expected_clarification": True}}
    rep = evaluate_case(case, _result([_song("A", "x")], intent="vector_search"))

    assert rep["case_status"] == "fail"


def test_dialog_state_contains_check_passes_on_nested_fields():
    case = {
        "id": "dst",
        "query": "同样氛围换中文",
        "checks": {
            "dialog_state_contains": {
                "hard_constraints.language": "Chinese",
                "soft_intent.vibe": "ethereal",
            }
        },
    }
    result = {
        "recommendations": [],
        "dialog_state": {
            "hard_constraints": {"language": "Chinese"},
            "soft_intent": {"vibe": "ethereal female vocal"},
        },
    }
    rep = evaluate_case(case, result)

    assert rep["case_status"] == "pass"


def test_dialog_delta_contains_check_can_read_last_delta_fallback():
    case = {
        "id": "delta",
        "query": "同样氛围换中文",
        "checks": {"dialog_delta_contains": {"followup": True, "inherited": "soft_intent.vibe"}},
    }
    result = {
        "recommendations": [],
        "dialog_state": {
            "last_delta": {
                "followup": True,
                "inherited": ["soft_intent.vibe"],
            }
        },
    }
    rep = evaluate_case(case, result)

    assert rep["case_status"] == "pass"


def test_dialog_delta_contains_check_can_read_operation_list_index():
    case = {
        "id": "delta-op",
        "query": "再安静一点",
        "checks": {"dialog_delta_contains": {"operations.0.path": "soft_intent.vibe"}},
    }
    result = {
        "recommendations": [],
        "dialog_delta": {
            "followup": True,
            "operations": [
                {"op": "add", "path": "soft_intent.vibe", "value": "quiet, soft"},
            ],
        },
    }
    rep = evaluate_case(case, result)

    assert rep["case_status"] == "pass"


def test_load_cases_split_smoke():
    cases, meta = _load_cases(split="smoke")
    assert meta["split"] == "smoke"
    assert cases
    assert all("query" in c and "checks" in c for c in cases)


def test_load_cases_split_dev_and_holdout():
    dev, dev_meta = _load_cases(split="dev")
    holdout, holdout_meta = _load_cases(split="holdout")
    assert dev_meta["split"] == "dev"
    assert holdout_meta["split"] == "holdout"
    assert len(dev) >= 50
    assert len(holdout) >= 20
    assert {c.get("category") for c in dev}
    assert {c.get("category") for c in holdout}


def test_load_cases_custom_file(tmp_path):
    custom = tmp_path / "cases.json"
    custom.write_text('[{"id":"x","query":"q","checks":{"min_results":1}}]', encoding="utf-8")
    cases, meta = _load_cases(cases_file=str(custom), split="dev")
    assert meta["split"] == "custom"
    assert cases[0]["id"] == "x"


def test_evaluate_case_keeps_category_and_history_flag():
    case = {
        "id": "ctx",
        "category": "multi_turn_context",
        "goal_category": "interaction_refinement",
        "specificity": "LH",
        "query": "换安静点",
        "chat_history": [{"role": "user", "content": "太吵了"}],
        "checks": {"min_results": 1},
    }
    rep = evaluate_case(case, _result([_song("A", "x")]))
    assert rep["category"] == "multi_turn_context"
    assert rep["goal_category"] == "interaction_refinement"
    assert rep["specificity"] == "LH"
    assert rep["has_chat_history"] is True


def test_load_context_case_splits_are_available():
    dev, dev_meta = _load_cases(split="context_dev")
    holdout, holdout_meta = _load_cases(split="context_holdout")
    all_cases, all_meta = _load_cases(split="context_all")

    assert dev_meta["split"] == "context_dev"
    assert holdout_meta["split"] == "context_holdout"
    assert all_meta["split"] == "context_all"
    assert len(dev) >= 52
    assert len(holdout) >= 16
    assert len(all_cases) == len(dev) + len(holdout)
    assert {case.get("goal_category") for case in all_cases}
    assert {case.get("specificity") for case in all_cases} <= {"LL", "HL", "LH", "HH"}
