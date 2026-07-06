from services.llm_feedback_logger import (
    LOG_FILE,
    build_planning_feedback,
    load_planning_feedback,
    log_planning_feedback,
)


def test_planning_feedback_flags_soft_intent_without_tags():
    payload = build_planning_feedback(
        user_input="下雨天，柔软安静一点",
        plan={
            "intent_type": "hybrid_search",
            "reasoning": "雨天柔软",
            "context": "雨天安静歌单",
        },
        retrieval_plan={
            "_intent_type": "hybrid_search",
            "hard_constraints": {},
            "soft_intent": {
                "goal": "下雨天放松",
                "vibe": "soft quiet rainy",
                "avoid": ["Energetic", "Driving"],
            },
            "hints": {},
            "vector_acoustic_query": "",
        },
        provider="dashscope",
        model="qwen3.7-plus",
        user_id="local",
    )

    assert payload["tag_feedback"]["needs_tag_review"] is True
    assert "vector_acoustic_query" in payload["missing_information"]
    assert payload["tag_feedback"]["avoid_terms"] == ["Energetic", "Driving"]
    assert payload["decision_policy"]["auto_apply_tag_changes"] is False
    assert payload["decision_policy"]["hot_path_extra_llm_call"] is False


def test_planning_feedback_logs_jsonl(tmp_path, monkeypatch):
    monkeypatch.setenv("MUSIC_LLM_FEEDBACK_DIR", str(tmp_path))
    payload = build_planning_feedback(
        user_input="我想听陈奕迅的粤语歌",
        plan={"intent_type": "graph_search", "reasoning": "歌手语言"},
        retrieval_plan={
            "hard_constraints": {
                "artist_entities": ["陈奕迅"],
                "language": "Cantonese",
            },
            "soft_intent": {},
            "hints": {},
        },
    )

    path = log_planning_feedback(payload)
    rows = load_planning_feedback(path)

    assert path == tmp_path / LOG_FILE
    assert len(rows) == 1
    assert rows[0]["intent_type"] == "graph_search"
    assert rows[0]["retrieval_plan"]["hard_constraints"]["language"] == "Cantonese"
