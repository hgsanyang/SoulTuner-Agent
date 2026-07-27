"""Planner input protocol: habit gating, status semantics, anchors, time units."""

from __future__ import annotations

import pytest

from schemas.agent_context import (
    AssembledContext,
    HabitCandidate,
    HabitCard,
    HabitEvidence,
    RecommendedAnchor,
    promote_candidates,
)


def _passing_evidence(**kw) -> HabitEvidence:
    base = {"support_sessions": 8, "distinct_weeks": 4, "lift_vs_baseline": 1.7,
            "context_rate": 0.62, "baseline_rate": 0.36}
    base.update(kw)
    return HabitEvidence(**base)


def test_gate_blocks_weak_evidence():
    weak = HabitCandidate(habit_id="h1", statement="深夜低动态",
                          evidence=_passing_evidence(support_sessions=2))
    assert not weak.passes_gate()
    with pytest.raises(ValueError):
        HabitCard(habit_id="h1", statement="深夜低动态",
                  evidence=_passing_evidence(support_sessions=2))


def test_gate_admits_strong_evidence():
    strong = HabitCandidate(habit_id="h2", statement="工作日深夜偏低动态独立",
                            evidence=_passing_evidence())
    assert strong.passes_gate()
    HabitCard.model_validate(strong.model_dump())


def test_user_correction_is_trusted_not_dropped():
    """A corrected habit is MORE trustworthy than a raw statistic — the earlier
    schema wrongly excluded every user-edited card."""
    corrected = HabitCandidate(habit_id="h3", statement="深夜其实想要更亮一点",
                               status="corrected", source="user_confirmed",
                               evidence=_passing_evidence(support_sessions=1, distinct_weeks=1))
    assert corrected.passes_gate()
    disabled = HabitCandidate(habit_id="h4", statement="不要这条", status="disabled",
                              evidence=_passing_evidence())
    assert not disabled.passes_gate()


def test_promote_puts_user_confirmed_first():
    stat = HabitCandidate(habit_id="s", statement="统计", evidence=_passing_evidence(lift_vs_baseline=2.5))
    user = HabitCandidate(habit_id="u", statement="用户确认", source="user_confirmed",
                          evidence=_passing_evidence(lift_vs_baseline=1.3))
    cards = promote_candidates([stat, user])
    assert [c.habit_id for c in cards] == ["u", "s"]


def test_context_rejects_ungated_habit():
    weak = HabitCandidate(habit_id="w", statement="弱", evidence=_passing_evidence(distinct_weeks=1))
    with pytest.raises(ValueError):
        AssembledContext(temporal_habits=[weak.model_dump()])


def test_session_carries_song_anchors():
    ctx = AssembledContext()
    ctx.session.last_recommendations = [
        RecommendedAnchor(music_id="m1", title="再见", artist="张震岳", rank=1, feedback="like"),
        RecommendedAnchor(music_id="m2", title="小宇", artist="张震岳", rank=2),
    ]
    assert ctx.session.last_recommendations[1].rank == 2
    # "没点开" stays unknown, never a negative label
    assert ctx.session.last_recommendations[1].feedback == ""


def test_timestamps_are_milliseconds():
    card = HabitCandidate(habit_id="t", statement="x", computed_at_ms=1_782_758_548_816,
                          evidence=_passing_evidence())
    assert card.computed_at_ms is not None and card.computed_at_ms > 1_000_000_000_000
