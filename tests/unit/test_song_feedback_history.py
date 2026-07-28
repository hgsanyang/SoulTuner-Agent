"""Per-song feedback had a write path and no read path.

You could rate a track and then had no way to see what you said, or even whether
you had already rated it — the data was in SQLite the whole time.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import services.feedback_store as fs


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("MUSIC_FEEDBACK_DIR", str(tmp_path))
    fs.reset_connection()
    from api.server import app

    yield TestClient(app)
    fs.reset_connection()


def _store(**overrides):
    payload = {
        "song_feedback_id": overrides.pop("fid", "sf-1"),
        "exposure_id": "exp-1",
        "music_id": "m-1",
        "title": "Hiraeth",
        "artist": "Bcalm, Banks",
        "user_id": "local_admin",
        "context_fit": "off",
        "off_reasons": ["too_loud"],
        "note": "前奏太长",
        "ts": 1000,
    }
    payload.update(overrides)
    fs.insert_song_feedback(payload)
    return payload


def test_history_returns_what_you_said_newest_first(client):
    _store(fid="sf-1", ts=1000, music_id="m-1", title="A")
    _store(fid="sf-2", ts=3000, music_id="m-2", title="B")
    _store(fid="sf-3", ts=2000, music_id="m-3", title="C")

    body = client.get("/api/song-feedback/history").json()
    assert body["success"] is True
    assert [i["title"] for i in body["items"]] == ["B", "C", "A"]
    first = body["items"][0]
    assert first["context_fit"] == "off"
    assert first["off_reasons"] == ["too_loud"]     # slugs, not Chinese labels
    assert first["note"] == "前奏太长"


def test_rated_music_ids_lets_cards_show_already_rated(client):
    _store(fid="sf-1", music_id="m-1")
    _store(fid="sf-2", music_id="m-2")
    body = client.get("/api/song-feedback/history").json()
    assert body["rated_music_ids"] == ["m-1", "m-2"]


def test_history_is_scoped_to_the_current_user(client):
    """A developer-mode rating must not surface in the personal history."""
    _store(fid="mine", user_id="local_admin", title="Mine")
    _store(fid="theirs", user_id="__dev__:local_admin", title="Theirs")

    titles = [i["title"] for i in client.get("/api/song-feedback/history").json()["items"]]
    assert titles == ["Mine"]


def test_history_is_empty_not_broken_when_nothing_was_rated(client):
    body = client.get("/api/song-feedback/history").json()
    assert body["success"] is True
    assert body["items"] == [] and body["rated_music_ids"] == []


def test_limit_is_clamped(client):
    for i in range(5):
        _store(fid=f"sf-{i}", music_id=f"m-{i}", ts=i)
    assert len(client.get("/api/song-feedback/history?limit=2").json()["items"]) == 2
    # a nonsense limit must not blow up or return everything unbounded
    assert client.get("/api/song-feedback/history?limit=0").json()["success"] is True
