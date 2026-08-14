"""Negative feedback must change recommendations without over-generalising.

The rejected design extracted a disliked song's acoustic traits into the query's
avoid list, which turns one data point about one track into "no piano, no slow
tempo, no Chinese". These two mechanisms are deliberately narrower.
"""

from __future__ import annotations

import pytest

from services.negative_feedback import (
    MAX_PENALTY_RATIO,
    apply_negative_anchor_penalty,
    load_candidate_vectors,
    load_rejection_anchors,
    negative_anchor_penalty,
    recent_context_rejection_rows,
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


# ---- review round 2: the four things that were claimed but not wired ----

def test_session_scope_beats_the_time_window():
    """Rejecting under "quiet, before sleep" must not follow you into a
    road-trip request an hour later — the whole point of session scoping."""
    rows = [
        {"music_id": "m-sleep", "context_fit": "off", "ts": NOW,
         "user_id": "local_admin", "context": {"session_id": "s-sleep"}},
        {"music_id": "m-road", "context_fit": "off", "ts": NOW,
         "user_id": "local_admin", "context": {"session_id": "s-road"}},
    ]
    got = recent_context_rejections("local_admin", session_id="s-road",
                                    now_ms=NOW, feedback_rows=rows)
    assert got == {"m-road"}, "a rejection from another session leaked in"


def test_session_id_reaches_streaming_and_both_planner_paths():
    """The UI uses SSE; covering only the non-stream entry point is insufficient."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    agent = (root / "agent" / "music_agent.py").read_text(encoding="utf-8")
    graph = (root / "agent" / "music_graph.py").read_text(encoding="utf-8")
    assert agent.count("resolve_conversation_id(client_context)") >= 2
    assert agent.count("checkpoint_thread_id(user_id, conversation_id)") >= 1
    assert "checkpoint_thread_id(user_id, _conversation_id)" in agent
    assert graph.count('retrieval_plan_dict["_session_id"] = str(state.get("session_id") or "")') >= 2


def test_rejection_snapshot_is_shared_by_suppression_and_ranking():
    rows = [
        {"music_id": "m-sleep", "context_fit": "off", "ts": NOW,
         "user_id": "local_admin", "context": {"session_id": "s-sleep"}},
        {"music_id": "m-road", "context_fit": "off", "ts": NOW,
         "user_id": "local_admin", "context": {"session_id": "s-road"}},
    ]
    got = recent_context_rejection_rows(
        "local_admin",
        session_id="s-road",
        now_ms=NOW,
        feedback_rows=rows,
    )
    assert [row["music_id"] for row in got] == ["m-road"]


def test_latest_rating_wins_so_a_correction_lifts_suppression():
    rows = [
        {"music_id": "m-1", "exposure_id": "e1", "context_fit": "off",
         "ts": NOW - 5000, "user_id": "local_admin"},
        {"music_id": "m-1", "exposure_id": "e1", "context_fit": "fits",
         "ts": NOW, "user_id": "local_admin"},
    ]
    assert recent_context_rejections("local_admin", now_ms=NOW,
                                     feedback_rows=rows) == set()


def test_web_candidate_without_music_id_is_still_suppressed():
    """Online candidates usually have no music_id; id-only matching let a
    rejected song return through the web lane."""
    items = [{"song": {"title": "Summer", "artist": "Calvin Harris"}},
             {"song": {"music_id": "m-2", "title": "B"}},
             {"song": {"music_id": "m-3", "title": "C"}},
             {"song": {"music_id": "m-4", "title": "D"}},
             {"song": {"music_id": "m-5", "title": "E"}},
             {"song": {"music_id": "m-6", "title": "F"}}]
    from services.negative_feedback import song_key
    kept, dropped = suppress_rejected(
        items, set(), rejected_names={song_key("Summer", "Calvin Harris")})
    assert dropped == 1
    assert all(i["song"].get("title") != "Summer" for i in kept)


def test_penalty_has_a_production_call_site():
    """It was a tested function nobody called: the flag changed nothing, which
    made "implemented, off by default" untrue."""
    from pathlib import Path
    src = Path(__file__).resolve().parents[2] / "retrieval" / "hybrid_retrieval.py"
    code = src.read_text(encoding="utf-8")
    assert code.rfind("apply_negative_anchor_penalty(") > code.rfind(
        "apply_post_recall_adjustments("
    ), "penalty must run after the final score is written"


def test_penalty_can_load_anchors_from_the_catalogue():
    """Feedback rows carry no vectors, so anchors must come from the catalogue —
    otherwise the penalty compares nothing and silently scores 0."""
    rows = {"m-1": {"music_id": "m-1", "muq_embedding": [1.0, 0.0]}}
    assert load_rejection_anchors(["m-1"], rows_by_id=rows) == [rows["m-1"]]
    assert load_rejection_anchors([], rows_by_id=rows) == []


def test_anchor_loader_preserves_feedback_age():
    rows = {"m-1": {"music_id": "m-1", "muq_embedding": [1.0, 0.0]}}
    anchors = load_rejection_anchors(
        [{"music_id": "m-1", "ts": NOW - 1234}],
        rows_by_id=rows,
    )
    assert anchors[0]["ts"] == NOW - 1234


def test_candidate_vectors_are_indexed_by_id_and_title_artist():
    items = [{"song": {"title": "Summer", "artist": "Calvin Harris"}}]
    rows = [{
        "music_id": "m-1",
        "title": "Summer",
        "artist": "Calvin Harris",
        "muq_embedding": [1.0, 0.0],
    }]
    indexed = load_candidate_vectors(items, rows=rows)
    assert indexed["id:m-1"]["muq_embedding"] == [1.0, 0.0]
    assert indexed["song:summer|calvinharris"]["music_id"] == "m-1"


def test_real_recall_shape_can_be_penalised_via_catalog_vectors(monkeypatch):
    """Recall intentionally strips heavy vectors; the ranking bridge restores them."""
    from retrieval.recall_sources import _record_to_song

    monkeypatch.setenv("MUSIC_NEGATIVE_ANCHOR_PENALTY", "1")
    song = _record_to_song({
        "title": "Summer",
        "artist": "Calvin Harris",
        "muq_embedding": [1.0, 0.0],
    })
    assert "muq_embedding" not in song
    item = {"song": song, "similarity_score": 1.0}
    candidates = load_candidate_vectors(
        [item],
        rows=[{
            "music_id": "m-1",
            "title": "Summer",
            "artist": "Calvin Harris",
            "muq_embedding": [1.0, 0.0],
        }],
    )
    touched = apply_negative_anchor_penalty(
        [item],
        [{"muq_embedding": [1.0, 0.0], "ts": NOW}],
        now_ms=NOW,
        candidate_vectors=candidates,
    )
    assert touched == 1
    assert item["similarity_score"] < 1.0


def test_penalty_actually_lowers_a_similar_candidate(monkeypatch):
    monkeypatch.setenv("MUSIC_NEGATIVE_ANCHOR_PENALTY", "1")
    near = {"song": {"muq_embedding": [1.0, 0.0]}, "similarity_score": 1.0}
    far = {"song": {"muq_embedding": [0.0, 1.0]}, "similarity_score": 1.0}
    anchors = [{"muq_embedding": [1.0, 0.0], "ts": NOW}]
    apply_negative_anchor_penalty([near, far], anchors, now_ms=NOW)
    assert near["similarity_score"] < 1.0
    assert near["similarity_score"] >= 1.0 - MAX_PENALTY_RATIO
    assert far["similarity_score"] == 1.0        # orthogonal: untouched


def test_penalty_does_nothing_while_disabled(monkeypatch):
    monkeypatch.delenv("MUSIC_NEGATIVE_ANCHOR_PENALTY", raising=False)
    item = {"song": {"muq_embedding": [1.0, 0.0]}, "similarity_score": 1.0}
    assert apply_negative_anchor_penalty(
        [item], [{"muq_embedding": [1.0, 0.0], "ts": NOW}]) == 0
    assert item["similarity_score"] == 1.0
