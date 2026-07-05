"""Prepare online-acquired songs for quick ingest and offline enrichment.

Default mode is a dry-run.  Use --quick-ingest to write verified metadata into
Neo4j, and --enqueue to schedule lyrics/audio-vector enrichment for the worker.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.catalog_enrichment import normalize_acquisition_metadata  # noqa: E402
from services.ingest_queue import enqueue_songs, list_jobs  # noqa: E402
from tools.acquire_music import _quick_ingest_to_neo4j  # noqa: E402

DEFAULT_ONLINE_ROOT = PROJECT_ROOT.parent / "data" / "online_acquired"


def safe_filename(text: str) -> str:
    return "".join(c for c in str(text or "") if c not in r'\/:*?"<>|').strip()


def artist_string(raw_artists: Any) -> str:
    names: list[str] = []
    for item in raw_artists or []:
        if isinstance(item, Mapping):
            name = str(item.get("name") or "").strip()
        elif isinstance(item, (list, tuple)) and item:
            name = str(item[0] or "").strip()
        else:
            name = str(item or "").strip()
        if name:
            names.append(name)
    return "、".join(names) if names else "Unknown"


def expected_basename(meta: Mapping[str, Any]) -> str:
    title = safe_filename(str(meta.get("musicName") or meta.get("title") or "Unknown"))
    artist = safe_filename(artist_string(meta.get("artist") or meta.get("artists")))
    return f"{title} - {artist}"


def build_song_from_meta(meta: Mapping[str, Any], root: Path) -> dict[str, Any]:
    normalized = normalize_acquisition_metadata(meta)
    basename = expected_basename(meta)
    fmt = str(meta.get("format") or normalized.get("format") or "mp3").lower().lstrip(".")
    audio_dir = root / "audio"
    found_ext = ""
    for ext in [fmt, "mp3", "flac", "m4a", "wav", "aac", "ogg"]:
        if (audio_dir / f"{basename}.{ext}").exists():
            found_ext = ext
            break
    if found_ext:
        fmt = found_ext

    return {
        "song_id": normalized.get("music_id") or normalized.get("source_id") or f"online_{basename}",
        "title": normalized.get("title") or str(meta.get("musicName") or ""),
        "artist": normalized.get("artist") or artist_string(meta.get("artist") or meta.get("artists")),
        "album": normalized.get("album", "Unknown"),
        "duration": normalized.get("duration", 0),
        "audio_url": f"/static/online_audio/{basename}.{fmt}",
        "cover_url": f"/static/online_covers/{basename}_cover.jpg",
        "lrc_url": f"/static/online_lyrics/{basename}.lrc",
        "file_basename": basename,
        "ext": fmt,
        "release_year": normalized.get("release_year"),
        "source_platform": normalized.get("source_platform", "netease"),
        "source_id": normalized.get("source_id") or normalized.get("music_id", ""),
        "metadata_source": normalized.get("metadata_source", "netease"),
        "album_id": normalized.get("album_id", ""),
        "has_audio": bool(found_ext),
        "has_cover": (root / "covers" / f"{basename}_cover.jpg").exists(),
        "has_lyrics": (root / "lyrics" / f"{basename}.lrc").exists(),
    }


def load_online_songs(root: Path = DEFAULT_ONLINE_ROOT) -> list[dict[str, Any]]:
    songs: list[dict[str, Any]] = []
    meta_dir = root / "metadata"
    for path in sorted(meta_dir.glob("*_meta.json")) if meta_dir.exists() else []:
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        song = build_song_from_meta(meta, root)
        song["meta_file"] = path.name
        songs.append(song)
    return songs


def _queued_song_keys() -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for job in list_jobs(limit=500):
        if job.get("status") not in {"pending", "processing"}:
            continue
        for song in job.get("songs") or []:
            keys.add((str(song.get("title", "")).casefold(), str(song.get("artist", "")).casefold()))
    return keys


def filter_ingestable(songs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ok: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for song in songs:
        reason = ""
        if not song.get("title") or not song.get("artist"):
            reason = "missing title/artist"
        elif not song.get("has_audio"):
            reason = "missing audio file"
        if reason:
            rejected.append({**song, "reject_reason": reason})
        else:
            ok.append(song)
    return ok, rejected


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.online_root)
    songs = load_online_songs(root)
    ingestable, rejected = filter_ingestable(songs)
    queued_keys = _queued_song_keys()
    enqueue_candidates = [
        song
        for song in ingestable
        if (song["title"].casefold(), song["artist"].casefold()) not in queued_keys
    ]

    result: dict[str, Any] = {
        "online_root": str(root),
        "metadata_files": len(songs),
        "ingestable": len(ingestable),
        "rejected": [{"title": row.get("title"), "artist": row.get("artist"), "reason": row["reject_reason"]} for row in rejected],
        "already_queued": len(ingestable) - len(enqueue_candidates),
        "quick_ingested": 0,
        "queued_job_id": "",
        "dry_run": not args.quick_ingest and not args.enqueue,
    }

    if args.quick_ingest and ingestable:
        await _quick_ingest_to_neo4j(ingestable)
        result["quick_ingested"] = len(ingestable)

    if args.enqueue and enqueue_candidates:
        job_id = enqueue_songs(enqueue_candidates)
        result["queued_job_id"] = job_id
        result["queued"] = len(enqueue_candidates)
    else:
        result["queued"] = 0

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare online-acquired songs for P11 ingestion.")
    parser.add_argument("--online-root", default=str(DEFAULT_ONLINE_ROOT))
    parser.add_argument("--quick-ingest", action="store_true", help="Write valid metadata into Neo4j")
    parser.add_argument("--enqueue", action="store_true", help="Queue valid songs for offline enrichment")
    args = parser.parse_args()
    print(json.dumps(asyncio.run(_run(args)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
