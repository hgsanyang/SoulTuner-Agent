"""Background flywheel for online recommendation candidates.

Online search results are recommendation candidates first.  This module turns
accepted online candidates into catalog assets without blocking the response:
temporary audio is acquired, metadata is written to Neo4j, and enrichment is
queued.  Explicit user positive feedback upgrades the audio retention to saved.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import aiohttp

from config.settings import settings

logger = logging.getLogger(__name__)

ONLINE_SOURCES = {"online_search", "web"}
POSITIVE_SAVE_EVENTS = {"like", "save"}
_ACQUIRE_LOCKS: dict[str, asyncio.Lock] = {}


def _song_from_item(item: dict[str, Any]) -> dict[str, Any]:
    song = item.get("song") if isinstance(item.get("song"), dict) else item
    return song if isinstance(song, dict) else {}


def is_online_candidate(song: dict[str, Any]) -> bool:
    """Return True when a song came from live online search, not local catalog."""
    source = str(song.get("source") or "").strip().lower()
    recall_sources = {str(value).strip().lower() for value in (song.get("recall_sources") or [])}
    audio_url = str(song.get("audio_url") or song.get("preview_url") or "")
    if audio_url.startswith("/static/online_audio/"):
        return False
    return source in ONLINE_SOURCES or "web" in recall_sources


def collect_online_candidates(items: list[dict[str, Any]] | Any, *, limit: int | None = None) -> list[dict[str, Any]]:
    """Collect deduped online candidates from recommendation-shaped rows."""
    max_items = int(limit if limit is not None else getattr(settings, "online_auto_flywheel_limit", 10))
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items or []:
        song = _song_from_item(item) if isinstance(item, dict) else {}
        if not song or not is_online_candidate(song):
            continue
        title = str(song.get("title") or "").strip()
        artist = str(song.get("artist") or "").strip()
        source_id = str(song.get("song_id") or song.get("source_id") or song.get("music_id") or "").strip()
        if not title or not artist or not source_id:
            continue
        key = source_id or f"{title.casefold()}::{artist.casefold()}"
        if key in seen:
            continue
        seen.add(key)
        candidates.append(song)
        if len(candidates) >= max_items:
            break
    return candidates


def should_auto_acquire_feedback(event_type: str, extra: dict[str, Any] | None) -> bool:
    if event_type not in POSITIVE_SAVE_EVENTS:
        return False
    payload = extra or {}
    source = str(payload.get("source") or "").strip().lower()
    source_id = payload.get("song_id") or payload.get("source_id")
    if source:
        return source in ONLINE_SOURCES and bool(source_id)
    platform = str(payload.get("platform") or "").strip().lower()
    return bool(source_id) and platform in {"netease", "online"}


def _payload_for_acquirer(song: dict[str, Any]) -> dict[str, Any]:
    artist = str(song.get("artist") or "").strip()
    artists = [{"name": part.strip()} for part in artist.replace("/", "、").split("、") if part.strip()]
    album = song.get("album")
    return {
        "id": song.get("song_id") or song.get("source_id") or song.get("music_id"),
        "name": song.get("title"),
        "artists": artists,
        "album": {"name": album or "Unknown"},
        "duration": song.get("duration") or 0,
    }


def _candidate_key(song: dict[str, Any]) -> str:
    source_id = str(song.get("song_id") or song.get("source_id") or song.get("music_id") or "").strip()
    if source_id:
        return source_id
    return f"{str(song.get('title') or '').casefold()}::{str(song.get('artist') or '').casefold()}"


async def _enqueue_enrichment(acquired: list[dict[str, Any]]) -> str | None:
    if not acquired:
        return None
    from services.ingest_queue import IngestQueueValidationError, enqueue_songs
    from tools.acquire_music import _background_flywheel

    inline_ingest = os.getenv("MUSIC_INLINE_INGEST_ENABLED", "0").lower() in {"1", "true", "yes"}
    if inline_ingest:
        asyncio.create_task(_background_flywheel(acquired))
        return "inline"
    try:
        return enqueue_songs(acquired)
    except IngestQueueValidationError as exc:
        logger.warning("[online-flywheel] enrichment queue rejected online songs: %s", exc)
        return None


async def acquire_and_ingest_online_candidates(
    candidates: list[dict[str, Any]],
    *,
    retention: str = "temporary",
    requested_by: str = "auto_recommendation",
) -> dict[str, Any]:
    """Acquire resolved online candidates, write metadata to Neo4j, and queue enrichment."""
    if not candidates:
        return {"requested": 0, "acquired": 0, "job_id": None}

    from tools.acquire_music import OnlineMusicAcquirer, _quick_ingest_to_neo4j

    acquirer = OnlineMusicAcquirer()
    acquired: list[dict[str, Any]] = []
    async with aiohttp.ClientSession() as session:
        for song in candidates:
            try:
                key = _candidate_key(song)
                lock = _ACQUIRE_LOCKS.setdefault(key, asyncio.Lock())
                async with lock:
                    result = await acquirer.acquire_resolved_song(
                        _payload_for_acquirer(song),
                        session,
                        retention=retention,
                        requested_by=requested_by,
                    )
                if result:
                    acquired.append(result)
            except Exception as exc:
                logger.warning("[online-flywheel] acquire failed for %s - %s: %s", song.get("title"), song.get("artist"), exc)

    if acquired:
        await _quick_ingest_to_neo4j(acquired)
    job_id = await _enqueue_enrichment(acquired)
    logger.info(
        "[online-flywheel] requested=%d acquired=%d retention=%s job=%s",
        len(candidates),
        len(acquired),
        retention,
        job_id or "-",
    )
    return {"requested": len(candidates), "acquired": len(acquired), "job_id": job_id}


def schedule_online_recommendation_flywheel(items: list[dict[str, Any]] | Any) -> int:
    """Schedule background ingestion for online recommendations and return candidate count."""
    if settings.eval_disable_side_effects:
        return 0
    if not getattr(settings, "online_auto_flywheel_enabled", True):
        return 0
    candidates = collect_online_candidates(items)
    if not candidates:
        return 0
    asyncio.create_task(
        acquire_and_ingest_online_candidates(
            candidates,
            retention="temporary",
            requested_by="auto_recommendation",
        )
    )
    return len(candidates)


def schedule_online_feedback_flywheel(
    *,
    event_type: str,
    title: str,
    artist: str,
    extra: dict[str, Any] | None,
) -> bool:
    """Schedule acquisition/retention upgrade for explicit positive feedback."""
    if settings.eval_disable_side_effects or not getattr(settings, "online_auto_flywheel_enabled", True):
        return False
    payload = extra or {}
    if not should_auto_acquire_feedback(event_type, payload):
        return False
    song = {
        "title": title,
        "artist": artist,
        "song_id": payload.get("song_id") or payload.get("source_id") or payload.get("music_id"),
        "source": payload.get("source") or "online_search",
        "platform": payload.get("platform") or "netease",
        "album": payload.get("album") or "Unknown",
        "duration": payload.get("duration") or 0,
    }
    asyncio.create_task(
        acquire_and_ingest_online_candidates(
            [song],
            retention="saved",
            requested_by=f"user_{event_type}",
        )
    )
    return True


def cleanup_expired_temporary_audio(
    *,
    ttl_hours: int | None = None,
    now: float | None = None,
    metadata_dir: Path | None = None,
    audio_dir: Path | None = None,
    update_neo4j: bool = True,
) -> dict[str, Any]:
    """Release stale temporary MP3 files while keeping metadata and embeddings."""
    from services.ingest_queue import list_jobs
    from tools.acquire_music import ONLINE_AUDIO_DIR, ONLINE_META_DIR

    ttl = int(ttl_hours if ttl_hours is not None else getattr(settings, "online_temp_audio_ttl_hours", 24))
    if ttl <= 0:
        return {"released": 0, "skipped_active": 0, "disabled": True}

    meta_root = metadata_dir or Path(ONLINE_META_DIR)
    audio_root = audio_dir or Path(ONLINE_AUDIO_DIR)
    if not meta_root.exists():
        return {"released": 0, "skipped_active": 0, "disabled": False}

    active_keys: set[str] = set()
    for job in list_jobs(limit=5000):
        if job.get("status") not in {"pending", "processing", "failed"}:
            continue
        for song in job.get("songs") or []:
            source_id = str(song.get("song_id") or song.get("source_id") or "").strip()
            title = str(song.get("title") or "").strip().casefold()
            artist = str(song.get("artist") or "").strip().casefold()
            if source_id:
                active_keys.add(f"id:{source_id}")
            if title and artist:
                active_keys.add(f"name:{title}::{artist}")

    released = 0
    skipped_active = 0
    current_time = float(now if now is not None else time.time())
    neo4j_client = None
    if update_neo4j:
        try:
            from retrieval.neo4j_client import get_neo4j_client

            neo4j_client = get_neo4j_client()
        except Exception:
            neo4j_client = None

    for meta_path in meta_root.glob("*_meta.json"):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if str(meta.get("audio_retention") or "temporary").lower() == "saved":
                continue
            if str(meta.get("acquire_status") or "ready").lower() != "ready":
                continue
            acquired_at = str(meta.get("acquired_at") or "").strip()
            try:
                from datetime import datetime

                acquired_ts = datetime.fromisoformat(acquired_at).timestamp() if acquired_at else meta_path.stat().st_mtime
            except (TypeError, ValueError):
                acquired_ts = meta_path.stat().st_mtime
            if current_time - acquired_ts < ttl * 3600:
                continue

            source_id = str(meta.get("source_id") or meta.get("musicId") or "").strip()
            title = str(meta.get("musicName") or "").strip()
            artists = meta.get("artist") or []
            artist = "、".join([a[0] if isinstance(a, list) else str(a) for a in artists]) if artists else ""
            keys = {f"name:{title.casefold()}::{artist.casefold()}"}
            if source_id:
                keys.add(f"id:{source_id}")
            if keys & active_keys:
                skipped_active += 1
                continue

            file_basename = meta_path.name[:-len("_meta.json")]
            audio_path = audio_root / f"{file_basename}.{meta.get('format') or 'mp3'}"
            if not audio_path.exists():
                continue
            audio_path.unlink()
            meta["audio_status"] = "released"
            meta["audio_released_at"] = int(current_time * 1000)
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            released += 1

            if neo4j_client is not None and source_id:
                neo4j_client.execute_query(
                    """
                    MATCH (s:Song)
                    WHERE toString(s.music_id) = $source_id OR toString(s.source_id) = $source_id
                    SET s.audio_url = '', s.audio_status = 'released', s.updated_at = timestamp()
                    """,
                    {"source_id": source_id},
                )
        except Exception as exc:
            logger.warning("[online-flywheel] temporary audio cleanup skipped %s: %s", meta_path.name, exc)

    if released:
        logger.info("[online-flywheel] released %d stale temporary audio files", released)
    return {"released": released, "skipped_active": skipped_active, "disabled": False}
