"""Read the user's own NetEase account data: daily recommendations, likes, playlists.

Why this exists: 日推 (daily recommendations) is account data. Every call this
repo makes to the NetEase proxy today is anonymous, which is exactly why the
daily list has never been available — not a technical limit, just a missing
login.

Scope, deliberately narrow:

* **Metadata only.** Titles, artists, albums, ids. Not one byte of audio passes
  through here. Audio acquisition stays where it already is, in
  ``data/pipeline/netease_wishlist_acquire.py``, which reports a track as
  unavailable rather than working around a trial-only URL.
* **The cookie never leaves the backend.** It is the user's account credential:
  not returned by any endpoint, not logged, not written to the frontend bundle.
* **Failure is silent and total.** The upstream API (Binaryify/NeteaseCloudMusicApi)
  was archived read-only in April 2024 and the community forks diverged after
  v4.28.0, so this integration can break without anyone fixing it. Every function
  here returns an empty result instead of raising, and no recommendation path may
  depend on it.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import aiohttp

from config.settings import settings

logger = logging.getLogger(__name__)

# QR login states as returned by /login/qr/check.
QR_EXPIRED = 800
QR_WAITING = 801
QR_SCANNED = 802
QR_CONFIRMED = 803

_TIMEOUT = aiohttp.ClientTimeout(total=15)


def _session_path() -> Path:
    """Where the account cookie lives. Outside the repo, never committed."""
    override = os.getenv("MUSIC_NETEASE_SESSION_PATH")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[1] / "config" / "netease_session.json"


def save_cookie(cookie: str) -> None:
    path = _session_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"cookie": cookie, "saved_at": int(time.time() * 1000)}, ensure_ascii=False),
        encoding="utf-8",
    )
    try:                                    # best effort; Windows ignores the mode
        path.chmod(0o600)
    except OSError:
        pass
    logger.info("[netease] 账号会话已保存")   # never log the value


def load_cookie() -> str:
    path = _session_path()
    if not path.exists():
        return ""
    try:
        return str(json.loads(path.read_text(encoding="utf-8")).get("cookie") or "")
    except Exception as exc:
        logger.warning("[netease] 会话读取失败: %s: %s", type(exc).__name__, exc)
        return ""


def clear_cookie() -> bool:
    path = _session_path()
    if not path.exists():
        return False
    path.unlink()
    logger.info("[netease] 账号会话已清除")
    return True


def is_logged_in() -> bool:
    return bool(load_cookie())


async def _get(
    session: aiohttp.ClientSession,
    endpoint: str,
    params: dict[str, Any] | None = None,
    *,
    authed: bool = False,
) -> dict[str, Any]:
    """One call to the local NetEase proxy. Returns {} on any failure."""
    url = f"{settings.netease_api_base}{endpoint}"
    query = dict(params or {})
    query["timestamp"] = int(time.time() * 1000)   # the proxy caches without it
    if authed:
        cookie = load_cookie()
        if not cookie:
            return {}
        query["cookie"] = cookie
    try:
        async with session.get(url, params=query, timeout=_TIMEOUT) as response:
            if response.status != 200:
                logger.warning("[netease] %s -> HTTP %s", endpoint, response.status)
                return {}
            return await response.json(content_type=None)
    except Exception as exc:
        # type(exc).__name__ matters: asyncio.TimeoutError stringifies to "",
        # which would leave a log line ending in a bare colon.
        logger.warning("[netease] %s 调用失败: %s: %s", endpoint, type(exc).__name__, exc)
        return {}


# ---------------------------------------------------------------- QR login ---

async def start_qr_login() -> dict[str, Any]:
    """Create a login QR code. Returns the image and the key used to poll it."""
    async with aiohttp.ClientSession() as session:
        key_payload = await _get(session, "/login/qr/key")
        unikey = str(((key_payload.get("data") or {}).get("unikey")) or "")
        if not unikey:
            return {"success": False, "error": "无法获取登录 key（网易云代理不可用）"}
        qr = await _get(session, "/login/qr/create", {"key": unikey, "qrimg": "true"})
        image = str(((qr.get("data") or {}).get("qrimg")) or "")
        if not image:
            return {"success": False, "error": "无法生成二维码"}
        return {"success": True, "key": unikey, "qr_image": image}


async def check_qr_login(unikey: str) -> dict[str, Any]:
    """Poll one QR login attempt. On confirmation the cookie is stored server-side.

    The cookie is deliberately absent from the return value — the browser polling
    this endpoint has no use for it, and putting a long-lived account credential
    into a JSON response is how it ends up in a log or a devtools screenshot.
    """
    if not unikey.strip():
        return {"success": False, "error": "缺少登录 key"}
    async with aiohttp.ClientSession() as session:
        payload = await _get(session, "/login/qr/check", {"key": unikey})
    code = int(payload.get("code") or 0)
    if code == QR_CONFIRMED:
        cookie = str(payload.get("cookie") or "")
        if not cookie:
            return {"success": False, "status": "error", "error": "确认成功但未返回会话"}
        save_cookie(cookie)
        return {"success": True, "status": "confirmed", "message": "登录成功"}
    return {
        "success": True,
        "status": {QR_EXPIRED: "expired", QR_WAITING: "waiting", QR_SCANNED: "scanned"}
        .get(code, "unknown"),
        "code": code,
    }


async def account_status() -> dict[str, Any]:
    """Who is logged in. Returns nickname/id only — never the cookie."""
    if not is_logged_in():
        return {"logged_in": False}
    async with aiohttp.ClientSession() as session:
        payload = await _get(session, "/user/account", authed=True)
    profile = payload.get("profile") or {}
    if not profile:
        # A stored cookie that no longer authenticates is worse than none: it
        # makes every later call fail in a way that looks like an outage.
        return {"logged_in": False, "stale_session": True}
    return {
        "logged_in": True,
        "nickname": profile.get("nickname") or "",
        "user_id": profile.get("userId") or "",
    }


# ------------------------------------------------------------- daily songs ---

def _normalise_track(raw: dict[str, Any]) -> dict[str, Any]:
    """One NetEase track payload -> the shape the rest of the app speaks."""
    artists = raw.get("ar") or raw.get("artists") or []
    names = [str(a.get("name") or "").strip() for a in artists if isinstance(a, dict)]
    album = raw.get("al") or raw.get("album") or {}
    return {
        "song_id": str(raw.get("id") or ""),
        "title": str(raw.get("name") or "").strip(),
        "artist": "、".join(n for n in names if n),
        "album": str(album.get("name") or "").strip() if isinstance(album, dict) else "",
        "cover_url": str(album.get("picUrl") or "") if isinstance(album, dict) else "",
        "duration": int(raw.get("dt") or raw.get("duration") or 0),
        "source": "online_search",
        "platform": "netease",
    }


async def fetch_daily_songs(limit: int = 30) -> list[dict[str, Any]]:
    """The account's daily recommendations, as metadata.

    Empty list when not logged in or when the proxy fails — callers must treat
    an empty result as "no daily list today", never as an error worth surfacing
    in the recommendation flow.
    """
    if not is_logged_in():
        return []
    async with aiohttp.ClientSession() as session:
        payload = await _get(session, "/recommend/songs", authed=True)
    tracks = ((payload.get("data") or {}).get("dailySongs")) or []
    songs = [_normalise_track(t) for t in tracks if isinstance(t, dict)]
    return [s for s in songs if s["title"] and s["artist"]][:limit]


async def fetch_liked_song_ids() -> list[str]:
    """The account's liked-track ids. Useful as a taste signal, not as audio."""
    if not is_logged_in():
        return []
    async with aiohttp.ClientSession() as session:
        payload = await _get(session, "/likelist", authed=True)
    return [str(i) for i in (payload.get("ids") or [])]


# --------------------------------------------------- reconcile with library ---

def match_against_library(
    songs: list[dict[str, Any]],
    *,
    rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Split a fetched list into "already in the library" and "not yet".

    Uses the same normalised title+artist key as the feedback and suppression
    paths. One definition of "the same song" across the app, or the shelves and
    the ledger quietly disagree about what the user already has.
    """
    from services.catalog_tier import CANDIDATE, LIBRARY
    from services.negative_feedback import song_key

    if rows is None:
        rows = _load_library_rows()

    known: dict[str, str] = {}
    for row in rows or []:
        key = song_key(row.get("title"), row.get("artist"))
        # library wins over candidate: a track saved on the shelf must not be
        # reported as "only a cache entry" because a stale candidate row exists.
        if known.get(key) == LIBRARY:
            continue
        known[key] = str(row.get("catalog_tier") or LIBRARY)

    in_library, in_candidates, missing = [], [], []
    for song in songs:
        tier = known.get(song_key(song.get("title"), song.get("artist")))
        if tier == LIBRARY:
            in_library.append(song)
        elif tier == CANDIDATE:
            in_candidates.append(song)
        else:
            missing.append(song)
    return {
        "in_library": in_library,
        "in_candidates": in_candidates,
        "missing": missing,
        "counts": {
            "total": len(songs),
            "in_library": len(in_library),
            "in_candidates": len(in_candidates),
            "missing": len(missing),
        },
    }


def _load_library_rows() -> list[dict[str, Any]]:
    try:
        from retrieval.neo4j_client import get_neo4j_client

        return [
            dict(row)
            for row in get_neo4j_client().execute_query(
                """
                MATCH (s:Song)
                OPTIONAL MATCH (s)-[:PERFORMED_BY]->(a:Artist)
                RETURN s.title AS title,
                       coalesce(a.name, s.artist, '') AS artist,
                       coalesce(s.catalog_tier, 'library') AS catalog_tier
                """,
                {},
            )
        ]
    except Exception as exc:
        logger.warning("[netease] 曲库对账读取失败: %s: %s", type(exc).__name__, exc)
        return []
