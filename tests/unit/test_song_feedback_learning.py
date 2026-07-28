"""Per-song context_fit finally reaches the offline learner.

Until now this table was written, could be read back for display, and was never
opened by services/ranking_learning.py — so rating a track "doesn't fit right
now" could not influence ranking even in principle.
"""

from __future__ import annotations

from services.ranking_learning import (
    SONG_CONTEXT_FIT_REWARD,
    build_song_feedback_rows,
    learn_ranking_policy,
)


def _exposure(**over):
    payload = {
        "exposure_id": "exp-1",
        "user_id": "local_admin",
        "shown_at_ms": 1_000,
        "ts": 1_500,
        "intent_type": "hybrid_search",
        "items": [
            {"music_id": "m-1", "title": "A", "artist": "X", "rank": 1},
            {"music_id": "m-2", "title": "B", "artist": "Y", "rank": 2},
        ],
    }
    payload.update(over)
    return payload


def _feedback(**over):
    payload = {
        "song_feedback_id": "sf-1",
        "exposure_id": "exp-1",
        "music_id": "m-1",
        "title": "A",
        "artist": "X",
        "user_id": "local_admin",
        "context_fit": "off",
        "off_reasons": ["too_loud"],
        "ts": 2_000,
    }
    payload.update(over)
    return payload


def test_off_becomes_a_negative_row_and_fits_a_positive_one():
    rows, diag = build_song_feedback_rows(
        [_exposure()],
        [_feedback(fid := "sf-1", context_fit="off") if False else _feedback(context_fit="off"),
         _feedback(song_feedback_id="sf-2", music_id="m-2", title="B", artist="Y",
                   context_fit="fits")],
    )
    assert len(rows) == 2
    by_fit = {row["context_fit"]: row for row in rows}
    assert by_fit["off"]["label"] == 0
    assert by_fit["fits"]["label"] == 1
    assert by_fit["off"]["reward"] == SONG_CONTEXT_FIT_REWARD["off"]
    assert diag["matched_song_feedback"] == 2


def test_partial_and_unrated_are_unknown_never_weak_negatives():
    """The standing rule: not rating something is not a negative sample."""
    rows, diag = build_song_feedback_rows(
        [_exposure()],
        [_feedback(context_fit="partial"),
         _feedback(song_feedback_id="sf-2", context_fit=""),
         _feedback(song_feedback_id="sf-3", context_fit=None)],
    )
    assert rows == []
    assert diag["neutral_or_missing_context_fit"] == 3


def test_a_rating_for_a_song_that_was_not_in_the_slate_is_dropped():
    """It cannot be attributed to any ranking decision, so it must not train."""
    rows, diag = build_song_feedback_rows(
        [_exposure()],
        [_feedback(music_id="m-999", title="Not shown", artist="Z")],
    )
    assert rows == []
    assert diag["song_not_in_exposure"] == 1


def test_attribution_anchors_on_shown_at_not_ts():
    """The exposure is written twice; `ts` is the LATER write. Anchoring on it
    would drop feedback given the instant a card appeared."""
    exposure = _exposure(shown_at_ms=1_000, ts=9_000)
    rows, _ = build_song_feedback_rows([exposure], [_feedback(ts=1_200)])
    assert len(rows) == 1, "feedback between shown_at and the final write was dropped"


def test_feedback_far_outside_the_window_is_dropped():
    rows, diag = build_song_feedback_rows(
        [_exposure()], [_feedback(ts=1_000 + 8 * 86_400_000)])
    assert rows == []
    assert diag["outside_attribution_window"] == 1


def test_learn_ranking_policy_reports_song_rows():
    """Wiring test: the pipeline must actually pass song feedback through."""
    report = learn_ranking_policy(
        [_exposure()], [], slate_feedback=[], song_feedback=[_feedback()],
        min_events=1,
    )
    assert report["diagnostics"]["song_rows"] == 1
    assert report["diagnostics"]["training_rows"] >= 1


def test_song_feedback_defaults_to_empty_so_old_callers_still_work():
    report = learn_ranking_policy([_exposure()], [], min_events=1)
    assert report["diagnostics"]["song_rows"] == 0
