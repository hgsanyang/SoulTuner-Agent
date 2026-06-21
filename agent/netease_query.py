"""Helpers for turning natural language music requests into Netease API calls."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import re
from typing import Any, Callable


@dataclass(frozen=True)
class NeteaseQueryPlan:
    query: str
    mode: str = "search"
    artist_terms: tuple[str, ...] = ()
    song_terms: tuple[str, ...] = ()


def _dedupe(items: list[str]) -> tuple[str, ...]:
    seen = set()
    result = []
    for item in items:
        cleaned = str(item or "").strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key not in seen:
            seen.add(key)
            result.append(cleaned)
    return tuple(result)


def has_chinese(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text or ""))


def is_recent_new_song_query(text: str) -> bool:
    text = text or ""
    return bool(re.search(r"(最近|最新|新近|今年|本周|本月|刚出|刚发|新歌|新曲|新专)", text)) and bool(
        re.search(r"(新歌|新曲|歌曲|歌|音乐|推荐|有什么|榜)", text)
    )


def extract_terms(retrieval_plan: dict[str, Any] | None) -> tuple[tuple[str, ...], tuple[str, ...]]:
    plan = retrieval_plan or {}
    hard = plan.get("hard_constraints") or {}
    artist_terms = list(plan.get("graph_artist_entities") or hard.get("artist_entities") or [])
    song_terms = list(plan.get("graph_song_entities") or hard.get("song_entities") or [])
    return _dedupe(artist_terms), _dedupe(song_terms)


def normalize_artist_name(name: str) -> str:
    return re.sub(r"[\s,，、.\-_/|·・&＋+]+", "", (name or "").lower())


def artist_matches(artist_text: str, artist_terms: tuple[str, ...]) -> bool:
    if not artist_terms:
        return True
    haystack = normalize_artist_name(artist_text)
    if not haystack:
        return False
    for term in artist_terms:
        needle = normalize_artist_name(term)
        if needle and (needle in haystack or haystack in needle):
            return True
    return False


def parse_play_url_payload(payload: dict[str, Any] | None) -> tuple[dict[str, str], dict[str, bool]]:
    """Extract playable URLs and trial flags from a Netease song/url response."""
    play_urls: dict[str, str] = {}
    trial_flags: dict[str, bool] = {}
    for item in (payload or {}).get("data", []):
        song_id = str(item.get("id", ""))
        play_url = item.get("url")
        if not song_id or not play_url:
            continue
        play_urls[song_id] = play_url
        trial_flags[song_id] = item.get("freeTrialInfo") is not None
    return play_urls, trial_flags


async def fetch_json_with_retry(
    session: Any,
    url: str,
    *,
    timeout: Any,
    attempts: int = 2,
    retry_delay: float = 0.25,
    on_retry: Callable[[int, Exception], None] | None = None,
) -> dict[str, Any]:
    """Fetch JSON with a bounded retry for transient proxy/network failures."""
    attempts = max(1, int(attempts))
    for attempt in range(1, attempts + 1):
        try:
            async with session.get(url, timeout=timeout) as response:
                return await response.json()
        except Exception as exc:
            if attempt >= attempts:
                raise
            if on_retry is not None:
                on_retry(attempt, exc)
            if retry_delay > 0:
                await asyncio.sleep(retry_delay)
    return {}


def clean_natural_query(text: str) -> str:
    query = re.sub(r"[《》\[\]【】]", " ", text or "")
    query = re.sub(r"^(搜索|查找|找|听|播放|我想听|帮我找|推荐|来几首|给我来几首)\s*", "", query)
    query = re.sub(r"(的歌|的歌曲|的音乐|的作品|唱的歌|唱的歌曲)\s*$", "", query)
    query = re.sub(r"\s+", " ", query).strip()
    return query


def build_netease_query_plan(
    user_input: str,
    fallback_query: str = "",
    retrieval_plan: dict[str, Any] | None = None,
    intent_parameters: dict[str, Any] | None = None,
) -> NeteaseQueryPlan:
    """Choose a stable Netease query or endpoint mode from graph/web planner context."""
    artist_terms, song_terms = extract_terms(retrieval_plan)
    params = intent_parameters or {}

    if is_recent_new_song_query(user_input) or is_recent_new_song_query(fallback_query):
        return NeteaseQueryPlan(query="华语新歌榜", mode="new_songs", artist_terms=artist_terms, song_terms=song_terms)

    chinese_artists = [a for a in artist_terms if has_chinese(a)]
    chinese_songs = [s for s in song_terms if has_chinese(s)]
    if chinese_artists and chinese_songs:
        return NeteaseQueryPlan(
            query=f"{chinese_artists[0]} {chinese_songs[0]}",
            artist_terms=artist_terms,
            song_terms=song_terms,
        )
    if chinese_artists:
        return NeteaseQueryPlan(query=chinese_artists[0], artist_terms=artist_terms, song_terms=song_terms)

    candidates = [
        fallback_query,
        params.get("query", ""),
        (retrieval_plan or {}).get("web_search_keywords", ""),
        user_input,
    ]
    for candidate in candidates:
        cleaned = clean_natural_query(candidate)
        if cleaned:
            return NeteaseQueryPlan(query=cleaned, artist_terms=artist_terms, song_terms=song_terms)

    return NeteaseQueryPlan(query=clean_natural_query(user_input), artist_terms=artist_terms, song_terms=song_terms)
