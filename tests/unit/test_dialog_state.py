from schemas.dialog_state import (
    apply_dialog_state_to_plan,
    apply_plan_delta,
    apply_plan_delta_with_report,
    coerce_followup_general_chat_to_retrieval,
    infer_dialog_state_from_history,
    should_clarify_before_planning,
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
