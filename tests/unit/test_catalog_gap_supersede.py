"""mix_in 被联网补充路线接管的降级语义测试。"""

from agent.catalog_gap import CatalogGapDecision, supersede_mix_in


def test_mix_in_is_superseded_and_traceable():
    decision = CatalogGapDecision(action="mix_in", inventory_count=24, target_web_count=6)
    result = supersede_mix_in(decision, superseded_by="web_supplement_lane")
    assert result.action == "none"
    assert result.details["mix_in_superseded_by"] == "web_supplement_lane"
    assert result.inventory_count == 24


def test_true_fallback_is_never_superseded():
    decision = CatalogGapDecision(action="fallback", inventory_count=0, target_web_count=10)
    result = supersede_mix_in(decision, superseded_by="web_supplement_lane")
    assert result is decision
    assert result.action == "fallback"


def test_none_and_blocked_pass_through():
    for action in ("none", "blocked"):
        decision = CatalogGapDecision(action=action)
        assert supersede_mix_in(decision, superseded_by="x") is decision
