"""Negative feedback must change recommendations without over-generalising.

The rejected design extracted a disliked song's acoustic traits into the query's
avoid list, which turns one data point about one track into "no piano, no slow
tempo, no Chinese". These two mechanisms are deliberately narrower.
"""

from __future__ import annotations

import pytest

from services.negative_feedback import (
    MAX_PENALTY_RATIO,
    negative_anchor_penalty,
    recent_context_rejections,
    similarity_penalty_enabled,
    suppress_rejected,
)

NOW = 1_000_000_000_000


def _fb(music_id="m-1", fit="off", ts=NOW, user="local_admin"):
    return {"music_id": music_id, "context_fit": fit, "ts": ts, "user_id": user}


def _item(music_id):
    return {"song": {"music_id": music_id, "title": music_id}, "similarity_score": 1.0}


def test_only_an_explicit_off_counts_as_a_rejection():
    rows = [_fb("m-1", "off"), _fb("m-2", "partial"), _fb("m-3", ""), _fb("m-4", "fits")]
    assert recent_context_rejections("local_admin", now_ms=NOW, feedback_rows=rows) == {"m-1"}


def test_rejections_expire_with_the_window():
    old = _fb("m-old", ts=NOW - 3 * 60 * 60 * 1000)      # 3h ago, window is 2h
    fresh = _fb("m-new", ts=NOW - 60 * 1000)
    got = recent_context_rejections("local_admin", now_ms=NOW, feedback_rows=[old, fresh])
    assert got == {"m-new"}


def test_another_users_rejection_does_not_affect_you():
    rows = [_fb("m-1", user="someone-else"), _fb("m-2", user="local_admin")]
    assert recent_context_rejections("local_admin", now_ms=NOW, feedback_rows=rows) == {"m-2"}


def test_suppression_removes_only_the_rejected_song():
    items = [_item("m-1"), _item("m-2"), _item("m-3"), _item("m-4"), _item("m-5"), _item("m-6")]
    kept, dropped = suppress_rejected(items, {"m-2"})
    assert dropped == 1
    assert [i["song"]["music_id"] for i in kept] == ["m-1", "m-3", "m-4", "m-5", "m-6"]


def test_suppression_gives_up_rather_than_returning_an_empty_slate():
    """A thin useless result is worse than repeating a rejected song."""
    items = [_item("m-1"), _item("m-2")]
    kept, dropped = suppress_rejected(items, {"m-1", "m-2"}, keep_at_least=5)
    assert dropped == 0 and kept == items


def test_no_rejections_is_a_no_op():
    items = [_item("m-1")]
    assert suppress_rejected(items, set()) == (items, 0)


def test_penalty_is_off_by_default(monkeypatch):
    """It changes ranking, so it stays off until a targeted eval passes."""
    monkeypatch.delenv("MUSIC_NEGATIVE_ANCHOR_PENALTY", raising=False)
    assert similarity_penalty_enabled() is False
    monkeypatch.setenv("MUSIC_NEGATIVE_ANCHOR_PENALTY", "1")
    assert similarity_penalty_enabled() is True


@pytest.mark.parametrize("sim", [0.0, 0.3, 0.7, 1.0, 5.0, -2.0])
def test_penalty_never_exceeds_the_cap(sim):
    assert 0.0 <= negative_anchor_penalty(sim) <= MAX_PENALTY_RATIO


def test_penalty_shrinks_with_distance_from_the_rejected_song():
    near = negative_anchor_penalty(0.9)
    far = negative_anchor_penalty(0.1)
    assert near > far > 0


def test_penalty_shrinks_in_an_unrelated_context():
    """A rejection in one kind of request must not follow you into another."""
    same = negative_anchor_penalty(0.9, context_similarity=1.0)
    other = negative_anchor_penalty(0.9, context_similarity=0.1)
    assert same > other


def test_penalty_decays_over_time():
    half_life = 14 * 24 * 60 * 60 * 1000
    fresh = negative_anchor_penalty(0.9, age_ms=0)
    aged = negative_anchor_penalty(0.9, age_ms=half_life)
    assert aged == pytest.approx(fresh / 2, rel=0.02)


def test_caller_cannot_raise_the_cap():
    """max_ratio is a ceiling, not a suggestion."""
    assert negative_anchor_penalty(1.0, max_ratio=0.95) <= MAX_PENALTY_RATIO
