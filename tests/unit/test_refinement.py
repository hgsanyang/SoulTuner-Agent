from schemas.query_plan import IntentHints, MusicQueryPlan, RetrievalPlan, SoftIntent
from schemas.refinement import build_refinement_suggestions


def _plan(*, intent_type: str = "vector_search", vibe: str = "", genres: list[str] | None = None) -> MusicQueryPlan:
    return MusicQueryPlan(
        intent_type=intent_type,
        retrieval_plan=RetrievalPlan(
            use_vector=True,
            soft_intent=SoftIntent(vibe=vibe),
            hints=IntentHints(genres=genres or []),
            vector_acoustic_query=vibe,
        ),
    )


def test_lofi_rainy_query_gets_non_blocking_refinement_chips():
    suggestion = build_refinement_suggestions(
        user_input="warm lo-fi beats for a rainy sunday afternoon",
        plan=_plan(vibe="warm lo-fi rainy afternoon beats", genres=["lo-fi"]),
        user_profile="偏好流派: 民谣, 独立, 流行；情绪偏向: 热血, 治愈, 怀旧",
    )

    labels = [option.label for option in suggestion.options]

    assert suggestion.confidence < 0.75
    assert labels[:4] == ["更安静", "更有雨天感", "更偏 lo-fi beat", "少人声"]


def test_focus_context_predicts_sparse_vocal_chip():
    suggestion = build_refinement_suggestions(
        user_input="写代码时听的安静音乐",
        plan=_plan(vibe="quiet focus music"),
    )

    labels = [option.label for option in suggestion.options]

    assert "少人声" in labels
    assert suggestion.options


def test_concrete_artist_request_does_not_offer_refinement_chips():
    plan = MusicQueryPlan.model_validate(
        {
            "intent_type": "graph_search",
            "retrieval_plan": {
                "use_graph": True,
                "hard_constraints": {"artist_entities": ["周杰伦"]},
            },
        }
    )

    suggestion = build_refinement_suggestions(user_input="周杰伦的歌", plan=plan)

    assert suggestion.confidence >= 0.85
    assert suggestion.options == []
