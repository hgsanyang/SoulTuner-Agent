"""Refinement 的确定性部分测试。

LLM-first 之后本模块只测两件事：
1. build_refinement_suggestions 只从结构化 plan 字段给出操作置信度，
   不再基于关键词/画像产出任何 chips；
2. schemas.refinement 中不残留关键词语义模板。
"""

from schemas.query_plan import IntentHints, MusicQueryPlan, RetrievalPlan, SoftIntent
from schemas.refinement import build_refinement_suggestions

import schemas.refinement as refinement_module


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


def test_soft_intent_lowers_confidence_and_yields_no_template_chips():
    suggestion = build_refinement_suggestions(
        user_input="warm lo-fi beats for a rainy sunday afternoon",
        plan=_plan(vibe="warm lo-fi rainy afternoon beats", genres=["lo-fi"]),
    )

    assert suggestion.confidence < 0.75
    assert suggestion.reason == "soft_intent_open_ended"
    # chips 由 LLM 在 slate 之后生成，这里必须为空
    assert suggestion.options == []


def test_concrete_constraints_keep_high_confidence():
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
    assert suggestion.reason == "concrete_constraints"
    assert suggestion.options == []


def test_confidence_ignores_free_text_keywords():
    # 同一 plan、不同 user_input 文本，置信度必须一致（不做关键词匹配）
    plan = _plan(vibe="quiet focus music")
    a = build_refinement_suggestions(user_input="写代码时听的安静音乐", plan=plan)
    b = build_refinement_suggestions(user_input="随便来点", plan=plan)

    assert a.confidence == b.confidence
    assert a.reason == b.reason


def test_non_recommend_intents_return_default():
    suggestion = build_refinement_suggestions(user_input="你好", plan=None)
    assert suggestion.confidence == 1.0
    assert suggestion.options == []


def test_no_keyword_semantic_templates_remain():
    forbidden = ("SOFT_AMBIGUITY_CUES", "_profile_options", "_fallback_options")
    for name in forbidden:
        assert not hasattr(refinement_module, name), f"keyword template {name} must stay removed"
