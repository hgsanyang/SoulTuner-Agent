"""Feedback protocol: two-channel independence, context derivation, unknown≠negative."""

from __future__ import annotations

import pytest

from schemas.feedback_events import (
    ExposureBookkeeping,
    ListeningContext,
    SlateFeedback,
    SongFeedback,
    derive_context,
)


def test_taste_and_context_are_independent_channels():
    """A track can be a long-term favourite AND wrong for tonight."""
    fb = SongFeedback(exposure_id="e1", music_id="m1", taste="like", context_fit="off",
                      off_reasons=["too_loud"], note="平时很爱，但今晚要安静")
    assert fb.taste == "like"
    assert fb.context_fit == "off"
    assert not fb.is_empty()


def test_free_text_alone_is_a_signal():
    fb = SongFeedback(exposure_id="e1", note="前奏太长了")
    assert not fb.is_empty()


def test_no_rating_is_empty_not_negative():
    """Not rating a song carries no signal — it must never become a negative."""
    fb = SongFeedback(exposure_id="e1", music_id="m1")
    assert fb.is_empty()
    assert fb.taste is None and fb.context_fit is None


def test_known_to_user_unknown_is_none_not_false():
    book = ExposureBookkeeping(rank=3)
    assert book.known_to_user is None  # unknown, not "user has not heard it"


def test_unknown_enum_values_rejected():
    with pytest.raises(ValueError):
        SongFeedback(exposure_id="e1", taste="meh")
    with pytest.raises(ValueError):
        SongFeedback(exposure_id="e1", context_fit="sorta")
    with pytest.raises(ValueError):
        SongFeedback(exposure_id="e1", off_reasons=["because"])


def test_slate_best_worst_capped():
    slate = SlateFeedback(exposure_id="e1", overall="partial",
                          best_music_ids=["a", "b", "c"], worst_music_ids=["x"])
    assert len(slate.best_music_ids) == 3
    with pytest.raises(ValueError):
        SlateFeedback(exposure_id="e1", best_music_ids=["a", "b", "c", "d"])


def test_derive_context_fills_local_fields():
    # 2026-07-23T14:30:00Z -> 22:30 Thursday in Shanghai
    ctx = derive_context(1784989800000, "Asia/Shanghai", session_id="s1", scene="加班回家")
    assert ctx.local_hour == 22
    assert ctx.day_type in {"weekday", "weekend"}
    assert ctx.day_part == "night"
    assert ctx.session_id == "s1" and ctx.scene == "加班回家"


def test_derive_context_without_timezone_stays_unknown():
    """Never silently assume the server's timezone — leave it unknown."""
    ctx = derive_context(1784989800000, "")
    assert ctx.local_hour is None
    assert ctx.day_type is None
    assert ctx.ts_ms == 1784989800000


def test_derive_context_bad_timezone_is_soft():
    ctx = derive_context(1784989800000, "Not/AZone")
    assert isinstance(ctx, ListeningContext)
    assert ctx.local_hour is None
