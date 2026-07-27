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


def test_legacy_slate_feedback_is_not_dropped_by_effective_id(store, tmp_path):
    """Regression (the 175->40 bug): records written before the primary-key fix
    carry only `feedback_id`. Keying dedup/migration on `slate_feedback_id` alone
    silently collapsed all of them. The effective id must fall back to
    feedback_id so every record survives."""
    import json as _json
    rows = [
        {"feedback_id": f"legacy-{i}", "rating": "partial", "ts": i}  # old shape
        for i in range(5)
    ] + [
        {"slate_feedback_id": f"new-{i}", "feedback_id": f"new-{i}", "rating": "off", "ts": 100 + i}
        for i in range(3)
    ]
    (tmp_path / fl.SLATE_FEEDBACK_FILE).write_text(
        "\n".join(_json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    # SQLite still empty -> must read all 8 from JSONL, none dropped
    assert len(fl.load_slate_feedback_canonical()) == 8

    # migrate a subset into SQLite, then the canonical read must still be 8
    # (union of store + log), not just what is in the store
    for r in rows[:4]:
        fs.insert_slate_feedback(r)
    got = fl.load_slate_feedback_canonical()
    assert len(got) == 8, f"union merge lost rows: {len(got)}"
    assert fs.counts()["slate_feedback"] == 4  # legacy rows keyed on feedback_id


def test_user_feedback_fails_loud_when_store_write_fails(store, monkeypatch):
    """SQLite is authoritative for user-submitted feedback: a failed write must
    raise, not silently fall back to JSONL and report success. (Exposure
    telemetry stays fail-soft — covered separately.)"""
    import schemas.feedback_events as fe

    def _boom(_payload):
        raise RuntimeError("disk full")

    monkeypatch.setattr(fs, "insert_song_feedback", _boom)
    fb = fe.SongFeedback(exposure_id="e1", music_id="m1", context_fit="off")
    with pytest.raises(fl.FeedbackStoreError):
        fl.log_song_feedback(fb)
    # the JSONL audit copy must NOT have been written for a failed authoritative write
    assert not (fl._jsonl_path("song_feedback.jsonl")).exists()


def test_exposure_write_stays_fail_soft(store, monkeypatch):
    """Passive telemetry must never fail a request even if the store errors."""
    monkeypatch.setattr(fs, "upsert_exposure", lambda _p: (_ for _ in ()).throw(RuntimeError("x")))
    # must not raise
    fl.log_exposure(query="q", user_id="u", request_id="soft1",
                    recommendations=[{"title": "A", "music_id": "m1"}])


def test_canonical_merges_store_and_log_not_either_or(store, tmp_path):
    """A new event in SQLite must not hide the historical JSONL log."""
    import json as _json
    (tmp_path / "events.jsonl").write_text(
        "\n".join(_json.dumps({"event_id": f"e{i}", "ts": i}) for i in range(10)) + "\n",
        encoding="utf-8")
    fs.insert_user_event({"event_id": "brand-new", "ts": 999})   # only in SQLite
    ids = {e.get("event_id") for e in fl.load_events_canonical()}
    assert "brand-new" in ids and "e0" in ids and len(ids) == 11


def test_event_without_its_id_is_kept_not_collapsed(store):
    """A writer sending the wrong key name must cost us a warning, not the table."""
    for i in range(3):
        fs.insert_song_feedback({"ts": i, "exposure_id": "e1", "music_id": f"m{i}"})
    assert fs.counts()["song_feedback"] == 3
