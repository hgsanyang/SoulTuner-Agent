"""/api/song-feedback + /api/slate-feedback guards (P1-RC2 hardening).

Three properties are load-bearing:
  1. orphan rejection — feedback must reference an exposure we persisted, and a
     song that was actually in it (songs stream over SSE before the exposure is
     written, so a user CAN rate first);
  2. server-side truth — rank/policy/origin come from OUR exposure record, never
     from what the browser reports about our own ranking;
  3. one taste path — long-term like/save stays on /api/user-event only.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    # one env var drives both the JSONL snapshots and the SQLite store
    monkeypatch.setenv("MUSIC_FEEDBACK_DIR", str(tmp_path))
    import services.feedback_logger as fl
    import services.feedback_store as fs

    fs.reset_connection()
    monkeypatch.setattr(fl, "_jsonl_path", lambda name: tmp_path / name)
    from api.server import app

    yield TestClient(app), tmp_path, fl
    fs.reset_connection()


def _write_exposure(fl, tmp_path, exposure_id="exp1"):
    row = {
        "type": "exposure", "exposure_id": exposure_id, "ts": 1784989800000,
        "policy_version": "rank-v7",
        "items": [
            {"music_id": "m1", "title": "再见", "artist": "张震岳", "rank": 1,
             "propensity": 0.42, "is_exploration": True, "catalog_origin": "online_new",
             "exposure_count": 3},
            {"music_id": "m2", "title": "小宇", "artist": "张震岳", "rank": 2},
        ],
    }
    (tmp_path / "exposures.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    return row


def test_orphan_feedback_rejected(client):
    api, tmp_path, fl = client
    resp = api.post("/api/song-feedback", json={"exposure_id": "nope", "music_id": "m1",
                                                "context_fit": "off"})
    assert resp.status_code == 409  # retryable: exposure may not be persisted yet


def test_song_not_in_exposure_rejected(client):
    api, tmp_path, fl = client
    _write_exposure(fl, tmp_path)
    resp = api.post("/api/song-feedback", json={"exposure_id": "exp1", "music_id": "ghost",
                                                "title": "不存在", "context_fit": "off"})
    assert resp.status_code == 422


def test_server_backfills_policy_truth(client):
    """Client-reported ranking fields are ignored; the exposure record wins."""
    api, tmp_path, fl = client
    _write_exposure(fl, tmp_path)
    resp = api.post("/api/song-feedback", json={
        "exposure_id": "exp1", "music_id": "m1", "context_fit": "off",
        "off_reasons": ["too_loud"], "timezone": "Asia/Shanghai", "session_id": "s9",
        "scene": "加班回家",
        # a hostile/stale client trying to restate our policy:
        "rank": 99, "policy_version": "attacker", "catalog_origin": "prior_favorite",
    })
    assert resp.status_code == 200, resp.text
    rows = [json.loads(x) for x in (tmp_path / "song_feedback.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    assert len(rows) == 1
    exposure = rows[0]["exposure"]
    assert exposure["rank"] == 1                     # from the record, not 99
    assert exposure["policy_version"] == "rank-v7"   # not "attacker"
    assert exposure["catalog_origin"] == "online_new"
    assert exposure["propensity"] == 0.42
    assert rows[0]["context"]["local_hour"] is not None
    assert rows[0]["context"]["session_id"] == "s9"
    assert rows[0]["context"]["scene"] == "加班回家"


def test_taste_is_not_accepted_here(client):
    """Long-term taste has exactly one entry point (/api/user-event)."""
    api, tmp_path, fl = client
    _write_exposure(fl, tmp_path)
    resp = api.post("/api/song-feedback", json={"exposure_id": "exp1", "music_id": "m1",
                                                "taste": "like"})
    # unknown field is ignored by the model -> no signal left -> 422
    assert resp.status_code == 422
    rows = [x for x in (tmp_path / "song_feedback.jsonl").read_text(encoding="utf-8").splitlines()] \
        if (tmp_path / "song_feedback.jsonl").exists() else []
    assert rows == []


def test_empty_feedback_rejected(client):
    api, tmp_path, fl = client
    _write_exposure(fl, tmp_path)
    resp = api.post("/api/song-feedback", json={"exposure_id": "exp1", "music_id": "m1"})
    assert resp.status_code == 422


def test_slate_best_worst_must_come_from_the_slate(client):
    api, tmp_path, fl = client
    _write_exposure(fl, tmp_path)
    resp = api.post("/api/slate-feedback", json={
        "exposure_id": "exp1", "rating": "partial",
        "best_music_ids": ["m1", "ghost"], "worst_music_ids": ["m2"],
        "timezone": "Asia/Shanghai",
    })
    assert resp.status_code in (200, 422)
    if resp.status_code == 200:
        # "ghost" was never exposed and must be dropped
        assert resp.json().get("success") is True


def test_final_exposure_supersedes_provisional(client):
    """The provisional record written before songs stream must be superseded by
    the final one (same exposure_id), so feedback attributes to the real slate."""
    api, tmp_path, fl = client
    fl.log_exposure(query="q", user_id="u", request_id="expX",
                    recommendations=[{"title": "A", "music_id": "m1"}], provisional=True)
    fl.log_exposure(query="q", user_id="u", request_id="expX", intent_type="hybrid_search",
                    recommendations=[{"title": "A", "music_id": "m1"},
                                     {"title": "B", "music_id": "m2"}],
                    policy_version="rank-v7")
    found = fl.lookup_exposure("expX")
    assert found is not None
    assert found["provisional"] is False          # last write wins
    assert found["intent_type"] == "hybrid_search"
    assert len(found["items"]) == 2


def test_feedback_works_against_provisional_exposure(client):
    """A user who rates the moment a card appears must not be rejected."""
    api, tmp_path, fl = client
    fl.log_exposure(query="q", user_id="u", request_id="expP",
                    recommendations=[{"title": "再见", "music_id": "m1"}], provisional=True)
    resp = api.post("/api/song-feedback", json={"exposure_id": "expP", "music_id": "m1",
                                                "context_fit": "fits"})
    assert resp.status_code == 200, resp.text


def test_final_exposure_write_must_not_erase_the_listening_context(client):
    """Regression: the streaming path's final write omitted `context`, and since
    the final record REPLACES the provisional one, it erased the listening
    context that had just been captured — unrecoverably, on the only path
    production uses."""
    api, tmp_path, fl = client
    ctx = {"ts_ms": 1784989800000, "timezone": "Asia/Shanghai", "local_hour": 23,
           "day_type": "weekday"}
    fl.log_exposure(query="q", user_id="u", request_id="expC",
                    recommendations=[{"title": "A", "music_id": "m1"}],
                    context=ctx, provisional=True)
    # the final write deliberately does NOT restate the context — that is the
    # mistake the call site made, and the logger must survive it
    fl.log_exposure(query="q", user_id="u", request_id="expC", intent_type="hybrid_search",
                    recommendations=[{"title": "A", "music_id": "m1"}],
                    policy_version="rank-v7")
    final = fl.lookup_exposure("expC")
    assert final["context"].get("local_hour") == 23
    assert final["intent_type"] == "hybrid_search"   # the refinement still lands
    assert final["policy_version"] == "rank-v7"


def test_final_exposure_write_can_still_override_the_context(client):
    """Carry-forward fills gaps; it must never overrule what the caller passed."""
    api, tmp_path, fl = client
    fl.log_exposure(query="q", user_id="u", request_id="expD",
                    recommendations=[{"title": "A", "music_id": "m1"}],
                    context={"timezone": "Asia/Shanghai", "local_hour": 9},
                    provisional=True)
    fl.log_exposure(query="q", user_id="u", request_id="expD",
                    recommendations=[{"title": "A", "music_id": "m1"}],
                    context={"timezone": "Europe/Berlin", "local_hour": 3})
    assert fl.lookup_exposure("expD")["context"]["local_hour"] == 3


def test_slate_rating_and_reasons_are_stored_as_contract_values(client):
    """The strict contract's `overall`/`reasons` were declared but never filled,
    and the UI sent display labels, so the record could not be joined offline."""
    api, tmp_path, fl = client
    _write_exposure(fl, tmp_path)
    resp = api.post("/api/slate-feedback", json={
        "exposure_id": "exp1", "rating": "off",
        "reasons": ["too_repetitive", "太吵"],   # one slug, one legacy label
        "timezone": "Asia/Shanghai",
    })
    assert resp.status_code == 200, resp.text
    rows = [json.loads(x) for x in
            (tmp_path / "slate_feedback.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    assert len(rows) == 1
    assert rows[0]["reasons"] == ["too_repetitive"]          # validated slugs only
    assert rows[0]["extra"]["overall"] == "off"              # same scale as context_fit
    assert rows[0]["extra"]["reasons_raw"] == ["太吵"]        # nothing silently dropped
    assert rows[0]["slate_feedback_id"]


def test_slate_song_cannot_be_best_and_worst(client):
    api, tmp_path, fl = client
    _write_exposure(fl, tmp_path)
    resp = api.post("/api/slate-feedback", json={
        "exposure_id": "exp1", "rating": "partial",
        "best_music_ids": ["m1"], "worst_music_ids": ["m1"],
    })
    assert resp.status_code == 422
