"""Catalog enrichment helpers for the offline data flywheel.

The flywheel should enrich songs without inventing facts.  This module keeps
metadata, tags, and optional web knowledge cards provenance-aware so downstream
retrieval can decide what is safe to trust.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any, Mapping

from services.tag_policy import clean_tag_payload


DEFAULT_CONFIDENCE_BY_SOURCE = {
    "manual": 1.0,
    "netease": 0.92,
    "local_metadata": 0.82,
    "llm_lyrics": 0.72,
    "web": 0.68,
    "audio_model": 0.65,
    "unknown": 0.5,
}

MAX_FACTS = 8
MAX_SUMMARY_CHARS = 900
MAX_FACT_CHARS = 220


def clamp_confidence(value: Any, default: float = 0.5) -> float:
    """Clamp a confidence value into [0, 1]."""

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = float(default)
    return max(0.0, min(1.0, round(numeric, 3)))


def source_confidence(source: str | None, default: float = 0.5) -> float:
    """Return the default confidence for a metadata/tag source."""

    key = str(source or "unknown").strip().casefold() or "unknown"
    return DEFAULT_CONFIDENCE_BY_SOURCE.get(key, default)


def _clean_text(value: Any, *, max_length: int = 500) -> str:
    text = " ".join(str(value or "").strip().split())
    return text[:max_length]


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _parse_epoch_year(value: Any) -> int | None:
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return None
    if numeric <= 0:
        return None
    # Netease publishTime is usually milliseconds since epoch.
    seconds = numeric / 1000 if numeric > 10_000_000_000 else numeric
    try:
        year = datetime.fromtimestamp(seconds, tz=timezone.utc).year
    except (OSError, OverflowError, ValueError):
        return None
    return year if 1900 <= year <= 2100 else None


def extract_release_year(metadata: Mapping[str, Any]) -> int | None:
    """Extract a conservative release year from known metadata fields."""

    for key in ("release_year", "year"):
        try:
            year = int(metadata.get(key) or 0)
        except (TypeError, ValueError):
            year = 0
        if 1900 <= year <= 2100:
            return year

    for key in ("publishTime", "publish_time", "release_timestamp"):
        year = _parse_epoch_year(metadata.get(key))
        if year:
            return year

    for key in ("release_date", "publish_date", "date"):
        text = str(metadata.get(key) or "")
        match = re.search(r"\b(19\d{2}|20\d{2}|2100)\b", text)
        if match:
            return int(match.group(1))
    return None


def normalize_artist_list(raw_artists: Any) -> list[str]:
    """Normalize Netease-style or plain artist lists."""

    artists: list[str] = []
    for item in raw_artists or []:
        name = ""
        if isinstance(item, Mapping):
            name = str(item.get("name") or "")
        elif isinstance(item, (list, tuple)) and item:
            name = str(item[0] or "")
        else:
            name = str(item or "")
        name = _clean_text(name, max_length=120)
        if name and name.casefold() not in {a.casefold() for a in artists}:
            artists.append(name)
    return artists


def normalize_acquisition_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize downloaded/local metadata without adding uncertain fields."""

    artists = normalize_artist_list(metadata.get("artist") or metadata.get("artists"))
    title = _clean_text(metadata.get("musicName") or metadata.get("title"), max_length=220)
    album = _clean_text(metadata.get("album"), max_length=220) or "Unknown"
    source_platform = _clean_text(
        metadata.get("source_platform") or metadata.get("platform") or metadata.get("source"),
        max_length=80,
    )
    source_platform = source_platform if source_platform not in {"online"} else "netease"
    release_year = extract_release_year(metadata)

    normalized = {
        "music_id": str(metadata.get("musicId") or metadata.get("song_id") or metadata.get("id") or ""),
        "title": title,
        "artists": artists,
        "artist": "、".join(artists),
        "album": album,
        "album_id": str(metadata.get("album_id") or metadata.get("albumId") or ""),
        "duration": int(metadata.get("duration") or 0),
        "format": _clean_text(metadata.get("format") or metadata.get("ext"), max_length=20),
        "source": _clean_text(metadata.get("source"), max_length=80) or "online",
        "source_platform": source_platform or "unknown",
        "source_id": str(metadata.get("source_id") or metadata.get("musicId") or metadata.get("song_id") or ""),
        "release_year": release_year,
        "cover_url": _clean_text(metadata.get("cover_url") or metadata.get("picUrl"), max_length=500),
        "lyrics_available": bool(metadata.get("lyrics_available") or metadata.get("lrc_url")),
        "metadata_source": _clean_text(metadata.get("metadata_source"), max_length=80) or source_platform or "unknown",
    }
    if metadata.get("artist_ids"):
        normalized["artist_ids"] = [
            str(item)
            for item in metadata.get("artist_ids") or []
            if str(item or "").strip()
        ]
    if metadata.get("aliases"):
        normalized["aliases"] = [
            _clean_text(item, max_length=180)
            for item in metadata.get("aliases") or []
            if _clean_text(item, max_length=180)
        ]
    return {key: value for key, value in normalized.items() if value not in ("", None)}


def prepare_tag_enrichment(
    payload: Mapping[str, Any],
    *,
    source: str,
    confidence: float | None = None,
) -> dict[str, Any]:
    """Clean tags and attach JSON provenance/score fields for Neo4j properties."""

    tags = clean_tag_payload(dict(payload))
    score = clamp_confidence(confidence, source_confidence(source))
    tag_sources = {
        field: {tag: source for tag in values}
        for field, values in tags.items()
        if values
    }
    tag_confidence = {
        field: {tag: score for tag in values}
        for field, values in tags.items()
        if values
    }
    return {
        **tags,
        "tag_source": source,
        "tag_confidence_json": _json_dumps(tag_confidence),
        "tag_sources_json": _json_dumps(tag_sources),
    }


def normalize_knowledge_card(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize optional web/RAG knowledge about a song or artist.

    These cards are for search expansion and explanation.  They are not treated
    as ground-truth catalog fields unless a later verifier promotes them.
    """

    kind = _clean_text(payload.get("kind"), max_length=40).casefold()
    if kind not in {"song", "artist"}:
        kind = "song"
    summary = _clean_text(payload.get("summary"), max_length=MAX_SUMMARY_CHARS)
    facts = [
        _clean_text(fact, max_length=MAX_FACT_CHARS)
        for fact in payload.get("facts") or []
        if _clean_text(fact, max_length=MAX_FACT_CHARS)
    ][:MAX_FACTS]
    source_url = _clean_text(payload.get("source_url") or payload.get("url"), max_length=800)
    source = _clean_text(payload.get("source"), max_length=80) or "web"
    confidence = clamp_confidence(payload.get("confidence"), source_confidence(source))
    return {
        "kind": kind,
        "title": _clean_text(payload.get("title"), max_length=220),
        "artist": _clean_text(payload.get("artist"), max_length=220),
        "summary": summary,
        "facts": facts,
        "source": source,
        "source_url": source_url,
        "confidence": confidence,
    }


def build_song_knowledge_query(title: str, artist: str = "") -> str:
    """Build a web-search query for an offline song knowledge card."""

    title = _clean_text(title, max_length=160)
    artist = _clean_text(artist, max_length=160)
    if artist:
        return f"{title} {artist} 歌曲 发行 背景 风格"
    return f"{title} 歌曲 发行 背景 风格"


def build_artist_knowledge_query(artist: str) -> str:
    """Build a web-search query for an offline artist knowledge card."""

    artist = _clean_text(artist, max_length=160)
    return f"{artist} 音乐人 简介 风格 代表作"
