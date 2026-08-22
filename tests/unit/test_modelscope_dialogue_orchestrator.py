from __future__ import annotations

from deploy.modelscope_space import dialogue_orchestrator as dialogue


def test_one_entry_routes_discovery_and_explanations_to_different_roles() -> None:
    assert dialogue.classify_turn("公路旅行") == "recommendation"
    assert dialogue.classify_turn("来点有主唱的英伦摇滚") == "recommendation"
    assert dialogue.classify_turn("为什么下雨天会想听空间感强的音乐？") == "conversation"


def test_short_refinement_inherits_last_recommendation_request() -> None:
    previous = "外面下暴雨，想听安静但不压抑的音乐"

    assert dialogue.classify_turn("再摇滚一点", previous) == "recommendation"
    assert dialogue.contextualize_recommendation("再摇滚一点", previous) == (previous + "\n[本轮调整] 再摇滚一点")


def test_history_is_bounded_and_only_keeps_visible_dialogue_roles() -> None:
    history = [
        {"role": "system", "content": "hidden"},
        *[{"role": "user" if index % 2 == 0 else "assistant", "content": f"turn-{index}"} for index in range(30)],
    ]

    bounded = dialogue.bounded_history(history)

    assert len(bounded) == 20
    assert bounded[0]["content"] == "turn-10"
    assert all(item["role"] in {"user", "assistant"} for item in bounded)
