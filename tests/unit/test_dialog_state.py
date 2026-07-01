import pytest

from schemas.dialog_state import (
    DeltaOperation,
    DialogMusicState,
    PlanDelta,
    apply_dialog_state_to_plan,
    apply_plan_delta_operations,
    apply_plan_delta,
    apply_plan_delta_with_report,
    build_deterministic_plan_delta,
    compile_dialog_state_to_plan,
    coerce_followup_general_chat_to_retrieval,
    infer_dialog_state_from_history,
    should_clarify_before_planning,
    update_dialog_result_anchors,
)
from schemas.query_plan import HardConstraints, IntentHints, MusicQueryPlan, RetrievalPlan, SoftIntent


def _plan(
    *,
    intent_type: str = "hybrid_search",
    language: str | None = None,
    artist: list[str] | None = None,
    vibe: str = "",
    genres: list[str] | None = None,
):
    return MusicQueryPlan(
        intent_type=intent_type,
        parameters={"query": "q"},
        retrieval_plan=RetrievalPlan(
            hard_constraints=HardConstraints(
                artist_entities=artist or [],
                language=language,
            ),
            soft_intent=SoftIntent(vibe=vibe),
            hints=IntentHints(genres=genres or []),
            use_vector=True,
            vector_acoustic_query=vibe,
        ),
    )


def test_followup_inherits_previous_vibe_and_replaces_language():
    first = apply_plan_delta(
        None,
        _plan(language="English", vibe="ethereal female vocal, dreamy synth", genres=["dream pop"]),
        "来点空灵英文女声",
    )

    second = apply_plan_delta(first, _plan(language="Chinese"), "同样的氛围，但换成中文")

    assert second.hard_constraints.language == "Chinese"
    assert second.soft_intent.vibe == "ethereal female vocal, dreamy synth"
    assert second.hints.genres == ["dream pop"]
    assert second.turn_count == 2


def test_followup_delta_reports_inherited_and_replaced_fields():
    first = apply_plan_delta(
        None,
        _plan(language="English", vibe="ethereal female vocal, dreamy synth", genres=["dream pop"]),
        "来点空灵英文女声",
    )

    second, delta = apply_plan_delta_with_report(first, _plan(language="Chinese"), "同样的氛围，但换成中文")

    assert second.last_delta == delta
    assert delta.followup is True
    assert delta.topic_shift is False
    assert "soft_intent.vibe" in delta.inherited
    assert delta.replaced["hard_constraints.language"] == "Chinese"


def test_followup_general_chat_is_coerced_to_retrieval_when_state_is_resolved():
    first = apply_plan_delta(
        None,
        _plan(language="English", vibe="ethereal female vocal, dreamy synth", genres=["dream pop"]),
        "来点空灵英文女声",
    )
    state, delta = apply_plan_delta_with_report(
        first,
        _plan(language="Chinese", intent_type="general_chat"),
        "同样的氛围，但换成中文",
    )
    plan = apply_dialog_state_to_plan(_plan(language="Chinese", intent_type="general_chat"), state)

    corrected = coerce_followup_general_chat_to_retrieval(plan, state, "同样的氛围，但换成中文")

    assert delta.followup is True
    assert corrected.intent_type == "hybrid_search"
    assert corrected.retrieval_plan.use_vector is True
    assert corrected.retrieval_plan.hard_constraints.language == "Chinese"
    assert "ethereal female vocal" in (corrected.retrieval_plan.vector_acoustic_query or "")


def test_smalltalk_general_chat_is_not_coerced_without_retrievable_state():
    plan = _plan(intent_type="general_chat")
    corrected = coerce_followup_general_chat_to_retrieval(plan, None, "今天怎么样")

    assert corrected.intent_type == "general_chat"


def test_explicit_new_artist_starts_new_topic():
    first = apply_plan_delta(
        None,
        _plan(language="English", vibe="quiet ambient", genres=["ambient"]),
        "安静英文氛围",
    )

    second = apply_plan_delta(first, _plan(artist=["周杰伦"], language="Chinese"), "周杰伦来几首")

    assert second.hard_constraints.artist_entities == ["周杰伦"]
    assert second.hard_constraints.language == "Chinese"
    assert second.soft_intent.vibe == ""
    assert second.hints.genres == []


def test_unresolved_reference_without_state_triggers_clarification():
    clarification = should_clarify_before_planning("有没有类似听感的", None)

    assert clarification.required is True
    assert clarification.reason == "unresolved_reference_without_state"
    assert clarification.options


def test_private_memory_reference_without_state_triggers_clarification():
    clarification = should_clarify_before_planning("推荐我上个月一直循环但后来又说不喜欢的那首", None)

    assert clarification.required is True
    assert clarification.reason == "private_memory_reference_without_state"
    assert "歌名" in clarification.question


def test_standalone_mood_trajectory_does_not_clarify():
    clarification = should_clarify_before_planning("从难过慢慢到释怀的那种感觉", None)

    assert clarification.required is False


def test_legacy_chat_history_seeds_followup_state():
    state = infer_dialog_state_from_history(
        [
            {"role": "user", "content": "推荐几首空灵的英文女声"},
            {"role": "assistant", "content": "推荐了一些英文空灵女声。"},
        ]
    )
    updated = apply_plan_delta(state, _plan(language="Chinese"), "同样的氛围，但换成中文")

    assert updated.hard_constraints.language == "Chinese"
    assert "ethereal" in updated.soft_intent.vibe
    assert "female vocal" in updated.soft_intent.vibe


def test_legacy_chat_history_extracts_artist_and_work_context():
    state = infer_dialog_state_from_history(
        [
            {"role": "user", "content": "林俊杰的歌"},
            {"role": "assistant", "content": "推荐了一些林俊杰歌曲。"},
            {"role": "user", "content": "给我一些适合写代码的歌"},
        ]
    )

    assert "林俊杰" in state.hard_constraints.artist_entities
    assert state.hints.scenario == "工作"


def test_dialog_state_syncs_back_to_legacy_plan_fields():
    state = apply_plan_delta(
        None,
        _plan(language="Japanese", vibe="city pop at night", genres=["city pop"]),
        "日语 city pop",
    )
    plan = apply_dialog_state_to_plan(_plan(), state)

    assert plan.retrieval_plan.graph_language_filter == "Japanese"
    assert plan.retrieval_plan.graph_genre_filter == "city pop"
    assert plan.retrieval_plan.soft_intent.vibe == "city pop at night"


def test_plan_delta_add_replace_and_remove_are_deterministic():
    initial = DialogMusicState(
        hard_constraints=HardConstraints(language="English"),
        soft_intent=SoftIntent(vibe="dreamy", avoid=["aggressive"]),
        hints=IntentHints(genres=["dream pop"]),
        turn_count=1,
    )
    delta = PlanDelta(
        operations=[
            DeltaOperation(op="replace", path="hard_constraints.language", value="Chinese"),
            DeltaOperation(op="add", path="soft_intent.vibe", value="quieter"),
            DeltaOperation(op="remove", path="soft_intent.avoid", value="aggressive"),
        ],
        confidence=0.96,
    )

    updated, report = apply_plan_delta_operations(initial, delta, "同样氛围，换中文并安静一点")

    assert updated.hard_constraints.language == "Chinese"
    assert updated.soft_intent.vibe == "dreamy; quieter"
    assert updated.soft_intent.avoid == []
    assert report.replaced["hard_constraints.language"] == "Chinese"
    assert report.planner_mode == "delta_llm"


def test_clear_topic_removes_inherited_music_state():
    initial = DialogMusicState(
        hard_constraints=HardConstraints(language="English", artist_entities=["A"]),
        soft_intent=SoftIntent(vibe="dreamy"),
        turn_count=2,
    )
    updated, report = apply_plan_delta_operations(
        initial,
        PlanDelta(operations=[DeltaOperation(op="clear_topic")]),
        "换个话题",
    )

    assert updated.hard_constraints.artist_entities == []
    assert updated.hard_constraints.language is None
    assert updated.soft_intent.vibe == ""
    assert report.topic_shift is True


def test_invalid_delta_path_is_rejected():
    with pytest.raises(ValueError, match="Unsupported dialogue delta path"):
        DeltaOperation(op="replace", path="retrieval_plan.use_web_search", value=True)


def test_common_followup_builds_small_delta_instead_of_full_plan():
    state = DialogMusicState(
        hard_constraints=HardConstraints(language="English"),
        soft_intent=SoftIntent(vibe="ethereal"),
        turn_count=1,
    )
    delta = build_deterministic_plan_delta("同样氛围，换成中文并更安静", state)

    assert delta is not None
    assert delta.planner_mode == "deterministic"
    assert {operation.path for operation in delta.operations} == {
        "hard_constraints.language",
        "soft_intent.vibe",
    }


def test_compile_delta_state_preserves_unmentioned_constraints():
    state = DialogMusicState(
        hard_constraints=HardConstraints(language="Chinese"),
        soft_intent=SoftIntent(vibe="ethereal; quieter"),
        hints=IntentHints(genres=["dream pop"]),
        turn_count=2,
    )
    plan = compile_dialog_state_to_plan(state, "再来一点")

    assert plan.intent_type == "hybrid_search"
    assert plan.retrieval_plan.hard_constraints.language == "Chinese"
    assert plan.retrieval_plan.hints.genres == ["dream pop"]
    assert "ethereal" in (plan.retrieval_plan.vector_acoustic_query or "")


def test_severe_conflict_uses_high_precision_clarification():
    clarification = should_clarify_before_planning("想要非常安静助眠但又炸裂蹦迪的歌", None)

    assert clarification.required is True
    assert clarification.reason == "severe_conflict"
    assert len(clarification.options) == 3


def test_result_anchors_enable_later_song_references():
    state = update_dialog_result_anchors(
        DialogMusicState(turn_count=1),
        [{"song": {"title": "Anchor", "artist": "Singer"}}],
    )

    assert state.last_result_titles == ["Anchor"]
    assert state.last_result_artists == ["Singer"]
    assert should_clarify_before_planning("刚才那首歌类似的", state).required is False
