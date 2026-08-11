from copy import deepcopy

from deploy.modelscope_space.planner_guard import build_safe_plan, guard_candidate


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
    assert any("安全边界要求 required" in finding for finding in findings)


def test_candidate_cannot_upgrade_optional_graph_to_required():
    candidate = _candidate("我心情很差，想听治愈的歌")
    candidate["lane_policy"]["graph"] = "required"
    accepted, findings = guard_candidate("我心情很差，想听治愈的歌", candidate)
    assert accepted["source"] == "deterministic_guard"
    assert accepted["lane_policy"]["graph"] == "optional"
    assert any("安全边界要求 optional" in finding for finding in findings)


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
