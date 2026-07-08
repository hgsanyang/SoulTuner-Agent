"""On-demand knowledge enrichment for songs that actually get recommended.

Bulk release-year backfills are expensive and often unnecessary.  This module
only schedules background Qwen web-search enrichment for songs that appear in a
real recommendation slate and still lack a sourced knowledge card.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Iterable, Mapping

from config.settings import settings
from services.knowledge_vector_index import upsert_cards_to_qdrant
from services.music_knowledge_enrichment import enrich_song_cards_batch
from services.music_knowledge_graph import upsert_knowledge_card_to_neo4j
from services.music_knowledge_store import MusicKnowledgeStore

logger = logging.getLogger(__name__)
_IN_FLIGHT: set[str] = set()


def _unwrap_song(item: Any) -> Mapping[str, Any]:
    if isinstance(item, Mapping):
        song = item.get("song", item)
        return song if isinstance(song, Mapping) else {}
    return {}


def _song_key(title: str, artist: str) -> str:
    return f"{title.strip().casefold()}::{artist.strip().casefold()}"


def _has_release_year(value: Any) -> bool:
    try:
        year = int(value or 0)
        return 1900 <= year <= 2100
    except (TypeError, ValueError):
        return False


def _card_is_complete(card: Mapping[str, Any] | None, song: Mapping[str, Any]) -> bool:
    if not card:
        return False
    has_summary = bool(str(card.get("summary") or "").strip())
    has_source = bool(str(card.get("source_url") or "").strip())
    has_year = _has_release_year(card.get("release_year")) or _has_release_year(song.get("release_year"))
    return has_summary and has_source and has_year


def select_missing_knowledge_songs(
    recommendations: Iterable[Any],
    *,
    store: MusicKnowledgeStore | None = None,
    limit: int | None = None,
) -> list[dict[str, str]]:
    """Return recommended songs that still need sourced knowledge cards."""

    store = store or MusicKnowledgeStore()
    max_items = max(0, int(limit if limit is not None else settings.knowledge_background_enrichment_limit))
    if max_items <= 0:
        return []
    selected: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in recommendations or []:
        song = _unwrap_song(item)
        title = str(song.get("title") or "").strip()
        artist = str(song.get("artist") or "").strip()
        if not title or not artist:
            continue
        key = _song_key(title, artist)
        if key in seen or key in _IN_FLIGHT:
            continue
        seen.add(key)
        try:
            card = store.get_song_card(title, artist)
        except Exception:
            card = None
        if _card_is_complete(card, song):
            continue
        selected.append({"title": title, "artist": artist})
        if len(selected) >= max_items:
            break
    return selected


async def _run_backfill(songs: list[dict[str, str]]) -> None:
    keys = {_song_key(song["title"], song["artist"]) for song in songs}
    _IN_FLIGHT.update(keys)
    try:
        cards = await enrich_song_cards_batch(songs, use_llm_summary=True)
        if not cards:
            logger.info("[KnowledgeBackfill] no cards returned for %d recommended songs", len(songs))
            return
        try:
            upsert_cards_to_qdrant(cards)
        except Exception as exc:
            logger.debug("[KnowledgeBackfill] qdrant sync skipped: %s", exc)
        try:
            from retrieval.neo4j_client import get_neo4j_client

            client = get_neo4j_client()
            for card in cards:
                upsert_knowledge_card_to_neo4j(client, card)
        except Exception as exc:
            logger.debug("[KnowledgeBackfill] neo4j sync skipped: %s", exc)
        logger.info("[KnowledgeBackfill] enriched %d/%d recommended songs", len(cards), len(songs))
    finally:
        _IN_FLIGHT.difference_update(keys)


def schedule_recommendation_knowledge_backfill(recommendations: Iterable[Any]) -> dict[str, Any]:
    """Schedule non-blocking knowledge enrichment for the current slate."""

    if settings.eval_disable_side_effects or not settings.knowledge_background_enrichment_enabled:
        return {"scheduled": 0, "reason": "disabled"}
    songs = select_missing_knowledge_songs(recommendations)
    if not songs:
        return {"scheduled": 0, "reason": "complete"}
    try:
        asyncio.create_task(_run_backfill(songs))
        return {"scheduled": len(songs), "reason": "queued"}
    except RuntimeError:
        return {"scheduled": 0, "reason": "no_running_loop"}
