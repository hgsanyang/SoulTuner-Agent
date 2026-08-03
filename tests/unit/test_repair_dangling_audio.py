"""A status label is not a behaviour change.

The first version of this repair set ``audio_status='missing'`` and stopped
there, and the report claimed those nodes had left playable recall. They had
not: ``retrieval/recall_sources.py::_playable_song_where`` filters on
``unplayable_stub`` and a non-empty ``audio_url`` and never reads
``audio_status``, so all 20 stayed recallable and still failed at play time.

So these tests assert against the real recall predicate rather than against the
label, and the identity tests use full artist sets rather than a lead performer.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.repair_dangling_audio import (
    STATIC_ROOTS,
    Dangling,
    apply,
    find_dangling,
    identity_key,
    plan,
    resolve,
    rollback,
)


class FakeClient:
    """Records writes and answers nothing. Enough to prove what Cypher runs."""

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def execute_query(self, query, params=None):
        self.calls.append((query, dict(params or {})))
        return [{"n": 1}]

    def wrote(self, fragment: str) -> bool:
        return any(fragment in q for q, _ in self.calls)


@pytest.fixture
def data_root(tmp_path):
    for relative in STATIC_ROOTS.values():
        (tmp_path / relative).mkdir(parents=True, exist_ok=True)
    return tmp_path


def _row(eid, title, artists, url, **extra):
    return {"eid": eid, "title": title, "artists": artists, "artist": "",
            "url": url, "music_id": "", **extra}


# ---- identity: the full artist set, not the first one -----------------------

def test_identity_uses_every_artist_not_just_the_lead():
    """A remix credited to "A" and one credited to "A feat. B" are different
    recordings. Keying on the lead performer alone would share one file between
    them and silently serve the wrong track."""
    solo = identity_key("Song", ["A"])
    duo = identity_key("Song", ["A", "B"])
    assert solo != duo


def test_identity_is_order_independent():
    assert identity_key("Song", ["A", "B"]) == identity_key("Song", ["B", "A"])


def test_identity_ignores_punctuation_spacing_and_case():
    assert identity_key(" Yellow ", ["Cold play"]) == identity_key("yellow", ["coldplay"])


def test_identity_separates_different_titles_by_the_same_artist():
    assert identity_key("A", ["X"]) != identity_key("B", ["X"])


def test_a_donor_with_a_different_collaborator_is_not_matched(data_root):
    """The failure this prevents: two tracks sharing a title and a lead artist
    getting one another's audio."""
    donor = data_root / STATIC_ROOTS["/static/online_audio/"] / "Song.mp3"
    donor.write_bytes(b"ID3\x03\x00\x00\x00\x00\x00\x00" + b"\x00" * 32)
    rows = [
        _row("e1", "Song", ["A", "B"], "/static/audio/Song.mp3"),      # missing
        _row("e2", "Song", ["A"], "/static/online_audio/Song.mp3"),    # present
    ]
    planned = plan(rows, data_root)
    assert len(planned) == 1
    assert planned[0].action == "missing", "不同合作者被误配成同一录音"


def test_a_donor_with_the_identical_artist_set_is_matched(data_root):
    donor = data_root / STATIC_ROOTS["/static/online_audio/"] / "Song.mp3"
    donor.write_bytes(b"ID3\x03\x00\x00\x00\x00\x00\x00" + b"\x00" * 32)
    rows = [
        _row("e1", "Song", ["B", "A"], "/static/audio/Song.mp3"),
        _row("e2", "Song", ["A", "B"], "/static/online_audio/Song.mp3"),
    ]
    planned = plan(rows, data_root)
    assert planned[0].action == "repoint"
    assert planned[0].donor_url == "/static/online_audio/Song.mp3"


# ---- one action per node ----------------------------------------------------

def test_a_multi_artist_node_produces_exactly_one_action(data_root):
    """PERFORMED_BY expansion used to emit one row per artist; a node must not
    be repaired three times because it has three performers."""
    rows = [
        _row("e1", "Song", ["A", "B", "C"], "/static/audio/gone.mp3"),
        _row("e1", "Song", ["A", "B", "C"], "/static/audio/gone.mp3"),
        _row("e1", "Song", ["A", "B", "C"], "/static/audio/gone.mp3"),
    ]
    assert len(plan(rows, data_root)) == 1


# ---- what "missing" has to actually do --------------------------------------

def test_missing_preserves_the_url_clears_it_and_sets_the_stub():
    client = FakeClient()
    apply([Dangling(eid="e1", title="T", artists=["A"], url="/static/audio/x.mp3")],
          client, run_id="run-1")
    query = client.calls[0][0]
    assert "s.previous_audio_url = coalesce(s.previous_audio_url, s.audio_url)" in query
    assert "s.audio_url = ''" in query
    assert "s.unplayable_stub = true" in query
    assert "s.audio_status = 'missing'" in query


def test_a_cleared_node_fails_the_real_recall_predicate():
    """Asserted against the predicate recall actually uses, not against the
    label — the label was the thing that turned out to mean nothing."""
    from retrieval.recall_sources import _playable_song_where

    predicate = _playable_song_where()
    assert "unplayable_stub" in predicate
    assert "audio_url" in predicate
    # the label the first version relied on is simply not consulted
    assert "audio_status" not in predicate


def test_apply_stamps_a_run_id_on_everything_it_touches():
    client = FakeClient()
    apply([Dangling(eid="e1", title="T", artists=["A"], url="/static/audio/x.mp3")],
          client, run_id="run-42")
    assert any(params.get("run_id") == "run-42" for _, params in client.calls)


def test_repoint_also_records_the_previous_state():
    client = FakeClient()
    apply([Dangling(eid="e1", title="T", artists=["A"], url="/static/audio/x.mp3",
                    donor_url="/static/online_audio/x.mp3", donor_path="/nope",
                    action="repoint")],
          client, run_id="run-1")
    query = client.calls[0][0]
    assert "s.previous_audio_url" in query
    assert "s.previous_unplayable_stub" in query
    assert "s.possible_duplicate = true" in query


# ---- rollback ---------------------------------------------------------------

def test_rollback_is_scoped_to_one_run():
    """Restoring every node that merely has a saved URL would also undo another
    run's work."""
    client = FakeClient()
    rollback(client, "run-7")
    query, params = client.calls[0]
    assert "s.repair_run_id = $run_id" in query
    assert params["run_id"] == "run-7"


def test_rollback_restores_the_recorded_values_not_fixed_ones():
    """A node that was already unplayable before any repair must not come back
    marked playable."""
    client = FakeClient()
    rollback(client, "run-7")
    query = client.calls[0][0]
    assert "s.audio_status = coalesce(s.previous_audio_status, '')" in query
    assert "s.unplayable_stub = coalesce(s.previous_unplayable_stub, false)" in query
    assert "s.audio_url = coalesce(s.previous_audio_url, s.audio_url)" in query


def test_rollback_clears_its_bookkeeping_so_a_second_run_is_a_no_op():
    client = FakeClient()
    rollback(client, "run-7")
    query = client.calls[0][0]
    for prop in ("s.repair_run_id = null", "s.previous_audio_url = null",
                 "s.previous_audio_status = null", "s.previous_unplayable_stub = null"):
        assert prop in query


# ---- dry run ----------------------------------------------------------------

def test_planning_writes_nothing(data_root):
    """plan() is the dry-run path; it must not reach the database at all."""
    client = FakeClient()
    (data_root / STATIC_ROOTS["/static/audio/"] / "here.mp3").write_bytes(b"ID3" + b"\x00" * 32)
    plan([_row("e1", "T", ["A"], "/static/audio/gone.mp3"),
          _row("e2", "T2", ["B"], "/static/audio/here.mp3")], data_root)
    assert client.calls == []


# ---- path resolution --------------------------------------------------------

def test_every_known_static_prefix_resolves(data_root):
    for prefix, relative in STATIC_ROOTS.items():
        assert resolve(f"{prefix}x.mp3", data_root) == data_root / relative / "x.mp3"


def test_an_unknown_prefix_is_treated_as_present_not_missing(data_root):
    """This script knows three mount points. Calling anything else broken would
    be a guess reported as a finding."""
    rows = [_row("e1", "T", ["A"], "/static/somewhere-else/x.mp3")]
    present, missing = find_dangling(rows, data_root)
    assert len(present) == 1 and missing == []
    assert plan(rows, data_root) == []


def test_an_empty_url_is_skipped_entirely(data_root):
    present, missing = find_dangling([_row("e1", "T", ["A"], "")], data_root)
    assert present == [] and missing == []


def test_a_present_file_is_not_planned(data_root):
    path = data_root / STATIC_ROOTS["/static/audio/"] / "real.mp3"
    path.write_bytes(b"ID3" + b"\x00" * 32)
    assert plan([_row("e1", "T", ["A"], "/static/audio/real.mp3")], data_root) == []


def test_artists_fall_back_to_the_scalar_field(data_root):
    """Older rows carry s.artist and no PERFORMED_BY relationship."""
    row = {"eid": "e1", "title": "T", "artists": [], "artist": "Solo",
           "url": "/static/audio/gone.mp3", "music_id": ""}
    planned = plan([row], data_root)
    assert planned[0].artists == ["Solo"]
    assert planned[0].artist_display == "Solo"
