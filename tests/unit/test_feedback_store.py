"""Canonical SQLite event store: WAL, upsert semantics, guards, export."""

from __future__ import annotations

import pytest

import services.feedback_logger as fl
import services.feedback_store as fs


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("MUSIC_FEEDBACK_DIR", str(tmp_path))
    fs.reset_connection()
    yield tmp_path
    fs.reset_connection()


def test_wal_mode_enabled(store):
    from contextlib import closing

    with closing(fs.get_connection()) as conn:
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert str(mode).lower() == "wal"


def test_store_lives_next_to_the_jsonl_snapshots(store):
    """One env var drives both, so the log and the store can never drift apart."""
    assert fs.store_path().parent == fl._feedback_dir()


def test_exposure_upsert_is_last_write_wins(store):
    fs.upsert_exposure({"exposure_id": "e1", "ts": 1, "provisional": True,
                        "items": [{"music_id": "m1"}], "count": 1})
    fs.upsert_exposure({"exposure_id": "e1", "ts": 2, "provisional": False,
                        "intent_type": "hybrid_search",
                        "items": [{"music_id": "m1"}, {"music_id": "m2"}], "count": 2})
    got = fs.get_exposure("e1")
    assert got["provisional"] is False
    assert got["intent_type"] == "hybrid_search"
    assert len(got["items"]) == 2
    assert fs.counts()["exposures"] == 1  # replaced, not appended


def test_missing_exposure_is_none(store):
    assert fs.get_exposure("nope") is None
    assert fs.get_exposure("") is None


def test_events_insert_and_count(store):
    fs.insert_song_feedback({"song_feedback_id": "s1", "ts": 1, "exposure_id": "e1",
                             "music_id": "m1", "context_fit": "off"})
    fs.insert_slate_feedback({"slate_feedback_id": "sl1", "ts": 1, "exposure_id": "e1",
                              "rating": "partial"})
    fs.insert_user_event({"event_id": "u1", "ts": 1, "exposure_id": "e1", "event_type": "like"})
    c = fs.counts()
    assert c["song_feedback"] == 1 and c["slate_feedback"] == 1 and c["user_events"] == 1


def test_writes_respect_eval_side_effect_guard(store, monkeypatch):
    monkeypatch.setattr(fs, "_writes_disabled", lambda: True)
    fs.upsert_exposure({"exposure_id": "blocked", "ts": 1})
    fs.insert_song_feedback({"song_feedback_id": "blocked", "ts": 1})
    assert fs.get_exposure("blocked") is None
    assert fs.counts()["song_feedback"] == 0


def test_export_jsonl_roundtrip(store, tmp_path):
    fs.upsert_exposure({"exposure_id": "e1", "ts": 10, "items": [], "count": 0})
    fs.upsert_exposure({"exposure_id": "e2", "ts": 20, "items": [], "count": 0})
    out = tmp_path / "snap.jsonl"
    assert fs.export_jsonl("exposures", out) == 2
    lines = [x for x in out.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert len(lines) == 2
    assert fs.export_jsonl("exposures", out, since_ms=15) == 1


def test_logger_mirrors_into_store(store):
    """log_exposure must land in BOTH the snapshot and the canonical store."""
    fl.log_exposure(query="q", user_id="u", request_id="mirror1",
                    recommendations=[{"title": "A", "music_id": "m1"}])
    assert fs.get_exposure("mirror1") is not None
    assert fl.lookup_exposure("mirror1") is not None


def test_every_logged_slate_feedback_is_retained(store):
    """Regression: the logger wrote `feedback_id` while the store keyed on
    `slate_feedback_id`, so every row landed under the primary key "" and
    INSERT OR REPLACE collapsed the whole table to a single event."""
    for i in range(3):
        fl.log_slate_feedback(exposure_id=f"e{i}", rating="partial", note=f"n{i}")
    assert fs.counts()["slate_feedback"] == 3
    assert len(fl.load_jsonl(fl._jsonl_path(fl.SLATE_FEEDBACK_FILE))) == 3


def test_exposure_captures_the_bookkeeping_it_can_know(store):
    """The debias fields were read by the feedback API but never written by the
    exposure snapshot, so production always stored None while the API test
    passed on a hand-written fixture."""
    fl.log_exposure(query="q", user_id="u", request_id="bk1", recommendations=[
        {"title": "A", "music_id": "m1", "recall_sources": ["online_search"],
         "_post_effective_exposure": 2.5},
        {"title": "B", "music_id": "m2", "recall_sources": ["graph"]},
    ])
    items = fs.get_exposure("bk1")["items"]
    # online_search must be recognised, not only the literal "web"
    assert items[0]["catalog_origin"] == "online_new"
    # the decayed float lands in effective_exposure, NOT an int-named count
    assert items[0]["effective_exposure"] == 2.5
    assert items[0]["historical_exposure_count"] is None
    # a local hit is left UNKNOWN rather than guessed as unheard
    assert items[1]["catalog_origin"] is None


def test_event_without_its_id_is_kept_not_collapsed(store):
    """A writer sending the wrong key name must cost us a warning, not the table."""
    for i in range(3):
        fs.insert_song_feedback({"ts": i, "exposure_id": "e1", "music_id": f"m{i}"})
    assert fs.counts()["song_feedback"] == 3
