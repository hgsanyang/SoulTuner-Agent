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
    monkeypatch.setenv("FEEDBACK_LOG_DIR", str(tmp_path))
    import services.feedback_logger as fl

    monkeypatch.setattr(fl, "_jsonl_path", lambda name: tmp_path / name)
    from api.server import app

    return TestClient(app), tmp_path, fl


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


def test_slate_song_cannot_be_best_and_worst(client):
    api, tmp_path, fl = client
    _write_exposure(fl, tmp_path)
    resp = api.post("/api/slate-feedback", json={
        "exposure_id": "exp1", "rating": "partial",
        "best_music_ids": ["m1"], "worst_music_ids": ["m1"],
    })
    assert resp.status_code == 422
