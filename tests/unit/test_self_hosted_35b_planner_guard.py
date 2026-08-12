from copy import deepcopy

import pytest

from deploy.self_hosted_35b.planner_guard import (
    build_safe_plan,
    guard_candidate,
    parse_candidate_content,
)


def _candidate(user_text, context=None):
    candidate = deepcopy(build_safe_plan(user_text, context))
    for generated_field in ("version", "source", "execution"):
        candidate.pop(generated_field)
    return candidate


def test_mood_uses_dense_primary_with_optional_graph():
    plan = build_safe_plan("我心情很差，想听温暖治愈的歌")
    assert plan["lane_policy"] == {"graph": "optional", "dense": "required", "web": "off"}
    assert plan["execution"]["profile"] == "dense_primary"


def test_pure_acoustic_request_is_dense_only():
    plan = build_safe_plan("我希望 bass 更重，鼓声更大一些")
    assert plan["lane_policy"] == {"graph": "off", "dense": "required", "web": "off"}
    assert plan["execution"]["profile"] == "dense_only"


def test_genre_request_is_graph_only():
    plan = build_safe_plan("找一些爵士乐")
    assert plan["lane_policy"]["graph"] == "required"
    assert plan["lane_policy"]["dense"] == "off"


def test_reference_request_requires_resolved_context():
    plan = build_safe_plan("给我和刚刚那首歌听感相似的")
    assert plan["response_mode"] == "clarify"
    assert plan["evidence"]["reason_codes"] == ["unresolved_reference"]


def test_resolved_reference_is_dense_anchor_not_hard_filter():
    plan = build_safe_plan(
        "给我和刚刚那首歌听感相似的",
        {"reference_title": "Dreams", "reference_artist": "Fleetwood Mac"},
    )
    assert plan["lane_policy"]["dense"] == "required"
    assert plan["hard"]["song"] == []
    assert plan["evidence"]["reference_songs"][0]["title"] == "Dreams"


def test_library_guidance_never_retrieves():
    plan = build_safe_plan("怎么导入我的网易云歌单？")
    assert plan["dialogue_mode"] == "library_guidance"
    assert all(mode == "off" for mode in plan["lane_policy"].values())


def test_candidate_omitting_required_dense_is_rejected():
    candidate = _candidate("我心情很差，想听治愈的歌")
    candidate["lane_policy"]["dense"] = "off"
    accepted, findings = guard_candidate("我心情很差，想听治愈的歌", candidate)
    assert accepted["source"] == "deterministic_guard"
    assert accepted["lane_policy"]["dense"] == "required"
    assert any("要求的 dense 通道" in finding for finding in findings)


def test_candidate_cannot_upgrade_optional_graph_to_required():
    candidate = _candidate("我心情很差，想听治愈的歌")
    candidate["lane_policy"]["graph"] = "required"
    accepted, findings = guard_candidate("我心情很差，想听治愈的歌", candidate)
    assert accepted["source"] == "deterministic_guard"
    assert accepted["lane_policy"]["graph"] == "optional"
    assert any("缺少实体、类型或年代约束" in finding for finding in findings)


def test_candidate_can_add_graph_for_a_catalog_filter_missed_by_keyword_parser():
    candidate = _candidate("想听低音更重、鼓点更大的中文歌")
    candidate["hard"]["language"] = "中文"
    candidate["lane_policy"]["graph"] = "required"
    candidate["evidence"]["reason_codes"].append("explicit_catalog_filter")
    accepted, findings = guard_candidate("想听低音更重、鼓点更大的中文歌", candidate)
    assert accepted["source"] == "model_candidate_guarded"
    assert accepted["lane_policy"]["graph"] == "required"
    assert findings == ["模型候选通过结构与策略守卫"]


def test_candidate_can_add_dense_and_keep_catalog_graph_as_secondary():
    candidate = _candidate("周末小聚想听轻松明亮的中文流行")
    candidate["lane_policy"] = {"graph": "optional", "dense": "required", "web": "off"}
    candidate["acoustic_queries"] = ["轻松明亮、适合周末小聚的听感"]
    candidate["evidence"]["reason_codes"].append("subjective_affective_goal")
    accepted, findings = guard_candidate("周末小聚想听轻松明亮的中文流行", candidate)
    assert accepted["source"] == "model_candidate_guarded"
    assert accepted["execution"]["profile"] == "dense_primary"
    assert findings == ["模型候选通过结构与策略守卫"]


def test_model_can_refine_generic_dense_fallback_into_explicit_catalog_graph():
    candidate = _candidate("来点 Radiohead 90 年代的作品")
    candidate["lane_policy"] = {"graph": "required", "dense": "off", "web": "off"}
    candidate["hard"]["artist"] = ["Radiohead"]
    candidate["metadata"]["release_year_from"] = 1990
    candidate["metadata"]["release_year_to"] = 1999
    candidate["acoustic_queries"] = []
    candidate["evidence"]["reason_codes"] = [
        "explicit_entity",
        "explicit_catalog_filter",
    ]

    accepted, findings = guard_candidate("来点 Radiohead 90 年代的作品", candidate)

    assert accepted["source"] == "model_candidate_guarded"
    assert accepted["lane_policy"] == {"graph": "required", "dense": "off", "web": "off"}
    assert findings == ["模型候选通过结构与策略守卫"]


def test_candidate_cannot_enable_web_without_deterministic_freshness_evidence():
    candidate = _candidate("想听适合跑步的摇滚")
    candidate["lane_policy"]["web"] = "required"
    candidate["metadata"]["external_knowledge_required"] = True
    candidate["evidence"]["reason_codes"].append("freshness_or_external")
    accepted, findings = guard_candidate("想听适合跑步的摇滚", candidate)
    assert accepted["source"] == "deterministic_guard"
    assert accepted["lane_policy"]["web"] == "off"
    assert any("模型 web 角色" in finding for finding in findings)


def test_candidate_with_thinking_field_is_rejected():
    candidate = _candidate("低音更重、鼓点更大的歌")
    candidate["thinking"] = "private reasoning"
    accepted, findings = guard_candidate("低音更重、鼓点更大的歌", candidate)
    assert accepted["source"] == "deterministic_guard"
    assert "thinking" not in accepted
    assert any("额外字段" in finding for finding in findings)


def test_valid_candidate_is_accepted():
    candidate = _candidate("低音更重、鼓点更大的歌")
    accepted, findings = guard_candidate("低音更重、鼓点更大的歌", candidate)
    assert accepted["source"] == "model_candidate_guarded"
    assert findings == ["模型候选通过结构与策略守卫"]


def test_empty_thinking_wrapper_is_removed_before_json_parse():
    assert parse_candidate_content('<think>\n\n</think>\n{"task_mode":"recommendation"}') == {
        "task_mode": "recommendation"
    }


def test_non_empty_thinking_is_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        parse_candidate_content('<think>private reasoning</think>{"task_mode":"recommendation"}')


def test_json_fence_is_supported():
    assert parse_candidate_content('```json\n{"task_mode":"recommendation"}\n```') == {
        "task_mode": "recommendation"
    }
