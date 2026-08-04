"""The account integration reads metadata and must never leak the credential.

Two properties matter more than the feature itself:

1. the session cookie stays server-side — it is a long-lived account credential,
   and the browser polling the login endpoint has no use for it;
2. every failure is empty, not raised. Upstream (Binaryify/NeteaseCloudMusicApi)
   was archived read-only in April 2024 and the forks diverged after v4.28.0, so
   this integration will break unattended. A daily-recommendation outage must not
   surface as a broken recommendation flow.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from services.netease_account import (
    QR_CONFIRMED,
    QR_WAITING,
    _normalise_track,
    check_qr_login,
    clear_cookie,
    is_logged_in,
    load_cookie,
    match_against_library,
    save_cookie,
)


@pytest.fixture(autouse=True)
def isolated_session(tmp_path, monkeypatch):
    monkeypatch.setenv("MUSIC_NETEASE_SESSION_PATH", str(tmp_path / "session.json"))
    yield


# ---- credential handling ----------------------------------------------------

def test_a_saved_cookie_round_trips():
    save_cookie("MUSIC_U=abc123")
    assert load_cookie() == "MUSIC_U=abc123"
    assert is_logged_in() is True


def test_no_session_file_means_logged_out():
    assert load_cookie() == ""
    assert is_logged_in() is False


def test_clearing_removes_the_session():
    save_cookie("MUSIC_U=abc123")
    assert clear_cookie() is True
    assert is_logged_in() is False
    assert clear_cookie() is False


def test_a_corrupt_session_file_reads_as_logged_out(tmp_path):
    """A truncated file must not raise on every subsequent call."""
    path = tmp_path / "session.json"
    path.write_text("{not json", encoding="utf-8")
    assert load_cookie() == ""


def test_the_cookie_is_never_in_the_login_response(monkeypatch):
    """A long-lived account credential in a JSON response is how it ends up in
    a log line or a devtools screenshot."""
    async def fake_get(session, endpoint, params=None, *, authed=False):
        return {"code": QR_CONFIRMED, "cookie": "MUSIC_U=secret-value"}

    monkeypatch.setattr("services.netease_account._get", fake_get)
    result = asyncio.run(check_qr_login("unikey-1"))

    assert result["status"] == "confirmed"
    assert "secret-value" not in json.dumps(result)
    # ...but it was stored where it belongs
    assert load_cookie() == "MUSIC_U=secret-value"


def test_a_waiting_scan_stores_nothing(monkeypatch):
    async def fake_get(session, endpoint, params=None, *, authed=False):
        return {"code": QR_WAITING}

    monkeypatch.setattr("services.netease_account._get", fake_get)
    result = asyncio.run(check_qr_login("unikey-1"))
    assert result["status"] == "waiting"
    assert is_logged_in() is False


def test_confirmation_without_a_cookie_is_an_error_not_a_silent_success(monkeypatch):
    async def fake_get(session, endpoint, params=None, *, authed=False):
        return {"code": QR_CONFIRMED, "cookie": ""}

    monkeypatch.setattr("services.netease_account._get", fake_get)
    result = asyncio.run(check_qr_login("unikey-1"))
    assert result["success"] is False
    assert is_logged_in() is False


def test_an_empty_key_is_rejected_before_any_call():
    assert asyncio.run(check_qr_login("  "))["success"] is False


# ---- track normalisation ----------------------------------------------------

def test_a_daily_track_maps_to_the_shape_the_app_speaks():
    song = _normalise_track({
        "id": 1234,
        "name": "海阔天空",
        "ar": [{"name": "Beyond"}],
        "al": {"name": "乐与怒", "picUrl": "https://example/cover.jpg"},
        "dt": 326000,
    })
    assert song["song_id"] == "1234"
    assert song["title"] == "海阔天空"
    assert song["artist"] == "Beyond"
    assert song["album"] == "乐与怒"
    assert song["duration"] == 326000
    assert song["platform"] == "netease"


def test_multiple_artists_join_the_way_the_rest_of_the_app_expects():
    song = _normalise_track({"id": 1, "name": "X", "ar": [{"name": "A"}, {"name": "B"}]})
    assert song["artist"] == "A、B"


def test_the_legacy_artists_album_keys_also_work():
    """Different NetEase endpoints return ar/al or artists/album."""
    song = _normalise_track({
        "id": 7, "name": "Y",
        "artists": [{"name": "C"}],
        "album": {"name": "Z"},
        "duration": 100,
    })
    assert song["artist"] == "C" and song["album"] == "Z" and song["duration"] == 100


# ---- reconciliation with the local shelves ----------------------------------

def _daily(title, artist):
    return {"title": title, "artist": artist, "song_id": "x"}


def test_daily_songs_split_across_library_candidate_and_missing():
    rows = [
        {"title": "海阔天空", "artist": "Beyond", "catalog_tier": "library"},
        {"title": "Summer", "artist": "Calvin Harris", "catalog_tier": "candidate"},
    ]
    result = match_against_library(
        [_daily("海阔天空", "Beyond"), _daily("Summer", "Calvin Harris"), _daily("New", "Nobody")],
        rows=rows,
    )
    assert result["counts"] == {"total": 3, "in_library": 1, "in_candidates": 1, "missing": 1}
    assert result["missing"][0]["title"] == "New"


def test_matching_ignores_punctuation_and_case():
    """The same key as suppression and feedback use — one definition of "the
    same song", or the shelves and the ledger disagree about what you have."""
    rows = [{"title": "Yellow", "artist": "Coldplay", "catalog_tier": "library"}]
    result = match_against_library([_daily("  yellow ", "COLDPLAY")], rows=rows)
    assert result["counts"]["in_library"] == 1


def test_a_library_row_wins_over_a_stale_candidate_row():
    """A saved track must not be reported as merely cached because an old
    candidate row for the same song is still around."""
    rows = [
        {"title": "Yellow", "artist": "Coldplay", "catalog_tier": "candidate"},
        {"title": "Yellow", "artist": "Coldplay", "catalog_tier": "library"},
    ]
    result = match_against_library([_daily("Yellow", "Coldplay")], rows=rows)
    assert result["counts"]["in_library"] == 1
    assert result["counts"]["in_candidates"] == 0


def test_an_empty_library_reports_everything_missing_not_an_error():
    result = match_against_library([_daily("A", "B")], rows=[])
    assert result["counts"]["missing"] == 1


def test_a_row_with_no_tier_counts_as_library():
    """Rows predating catalog_tier are the seeded local catalogue."""
    rows = [{"title": "A", "artist": "B"}]
    assert match_against_library([_daily("A", "B")], rows=rows)["counts"]["in_library"] == 1


# ---- failure is empty, never raised -----------------------------------------

def test_daily_songs_are_empty_when_logged_out():
    from services.netease_account import fetch_daily_songs

    assert asyncio.run(fetch_daily_songs()) == []


def test_a_proxy_outage_yields_an_empty_daily_list(monkeypatch):
    """Upstream is archived and unmaintained; an outage must not raise into the
    recommendation flow."""
    from services.netease_account import fetch_daily_songs

    save_cookie("MUSIC_U=abc")

    async def dead_proxy(session, endpoint, params=None, *, authed=False):
        return {}

    monkeypatch.setattr("services.netease_account._get", dead_proxy)
    assert asyncio.run(fetch_daily_songs()) == []


def test_tracks_missing_a_title_or_artist_are_dropped(monkeypatch):
    from services.netease_account import fetch_daily_songs

    save_cookie("MUSIC_U=abc")

    async def partial(session, endpoint, params=None, *, authed=False):
        return {"data": {"dailySongs": [
            {"id": 1, "name": "Good", "ar": [{"name": "A"}]},
            {"id": 2, "name": "", "ar": [{"name": "B"}]},
            {"id": 3, "name": "No artist", "ar": []},
        ]}}

    monkeypatch.setattr("services.netease_account._get", partial)
    songs = asyncio.run(fetch_daily_songs())
    assert [s["title"] for s in songs] == ["Good"]


def test_a_stale_cookie_reports_logged_out_rather_than_an_outage(monkeypatch):
    """A stored cookie that no longer authenticates makes every later call fail
    in a way that looks like the proxy is down."""
    from services.netease_account import account_status

    save_cookie("MUSIC_U=expired")

    async def no_profile(session, endpoint, params=None, *, authed=False):
        return {}

    monkeypatch.setattr("services.netease_account._get", no_profile)
    status = asyncio.run(account_status())
    assert status["logged_in"] is False
    assert status["stale_session"] is True
