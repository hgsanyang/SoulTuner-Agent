"""Audit local catalog-enrichment readiness without calling LLM/GPU/network.

This is a diagnostic companion for P11.  It scans downloaded/pending assets and
reports which pieces are present before the expensive background flywheel runs.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping
from urllib.error import URLError
from urllib.request import urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_ONLINE_ROOT = PROJECT_ROOT.parent / "data" / "online_acquired"
DEFAULT_KNOWLEDGE_STORE = PROJECT_ROOT.parent / "data" / "knowledge_cache" / "music_knowledge.sqlite"


def safe_filename(text: str) -> str:
    """Mirror the downloader's conservative Windows-safe filename policy."""

    return "".join(c for c in str(text or "") if c not in r'\/:*?"<>|').strip()


def artist_string(raw_artists: Any) -> str:
    names: list[str] = []
    for item in raw_artists or []:
        if isinstance(item, list) and item:
            name = str(item[0] or "").strip()
        elif isinstance(item, dict):
            name = str(item.get("name") or "").strip()
        else:
            name = str(item or "").strip()
        if name:
            names.append(name)
    return "、".join(names) if names else "Unknown"


def normalize_catalog_key(title: Any, artist: Any = "") -> str:
    """Build a stable duplicate-diagnosis key without mutating data."""
    text = f"{title or ''}::{artist or ''}".casefold()
    text = re.sub(r"\([^)]*(?:live|remaster|伴奏|cover|翻唱|版)[^)]*\)", "", text)
    text = re.sub(r"（[^）]*(?:live|remaster|伴奏|cover|翻唱|版)[^）]*）", "", text)
    text = re.sub(r"[\s\-_/|｜:：,，.。]+", "", text)
    return text


def expected_basename(meta: Mapping[str, Any]) -> str:
    title = safe_filename(str(meta.get("musicName") or meta.get("title") or "Unknown"))
    artist = safe_filename(artist_string(meta.get("artist") or meta.get("artists")))
    return f"{title} - {artist}"


def summarize_online_acquired(root: Path = DEFAULT_ONLINE_ROOT) -> dict[str, Any]:
    """Summarize downloaded online-acquired songs and missing enrichment inputs."""

    meta_dir = root / "metadata"
    audio_dir = root / "audio"
    cover_dir = root / "covers"
    lyrics_dir = root / "lyrics"

    rows: list[dict[str, Any]] = []
    totals = Counter()
    duplicate_counter: Counter[str] = Counter()
    duplicate_examples: dict[str, list[dict[str, Any]]] = {}
    for meta_file in sorted(meta_dir.glob("*_meta.json")) if meta_dir.exists() else []:
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
        except Exception as exc:
            rows.append({"file": meta_file.name, "valid_json": False, "error": str(exc)[:200]})
            totals["invalid_metadata_json"] += 1
            continue

        basename = expected_basename(meta)
        fmt = str(meta.get("format") or "mp3").lower().lstrip(".")
        audio_candidates = [audio_dir / f"{basename}.{fmt}"] + [
            audio_dir / f"{basename}.{ext}" for ext in ("mp3", "flac", "m4a", "wav", "aac", "ogg")
        ]
        has_audio = any(path.exists() for path in audio_candidates)
        has_cover = (cover_dir / f"{basename}_cover.jpg").exists()
        has_lyrics = (lyrics_dir / f"{basename}.lrc").exists()
        has_release_year = bool(meta.get("release_year"))
        title = meta.get("musicName") or meta.get("title") or ""
        artist = artist_string(meta.get("artist") or meta.get("artists"))
        duplicate_key = normalize_catalog_key(title, artist)
        if duplicate_key:
            duplicate_counter[duplicate_key] += 1
            duplicate_examples.setdefault(duplicate_key, []).append(
                {
                    "file": meta_file.name,
                    "music_id": str(meta.get("musicId") or ""),
                    "title": title,
                    "artist": artist,
                }
            )

        for key, present in (
            ("audio", has_audio),
            ("cover", has_cover),
            ("lyrics", has_lyrics),
            ("release_year", has_release_year),
        ):
            totals[f"has_{key}" if present else f"missing_{key}"] += 1

        rows.append(
            {
                "file": meta_file.name,
                "music_id": str(meta.get("musicId") or ""),
                "title": title,
                "artist": artist,
                "basename": basename,
                "has_audio": has_audio,
                "has_cover": has_cover,
                "has_lyrics": has_lyrics,
                "has_release_year": has_release_year,
                "source_platform": meta.get("source_platform") or meta.get("source") or "",
            }
        )

    totals["metadata_files"] = len(rows)
    duplicate_groups = [
        {
            "key": key,
            "count": count,
            "examples": duplicate_examples.get(key, [])[:5],
        }
        for key, count in duplicate_counter.most_common()
        if count > 1
    ]
    ready_rows = [
        row
        for row in rows
        if row.get("has_audio") and row.get("has_cover") and row.get("has_lyrics")
    ]
    return {
        "root": str(root),
        "totals": dict(totals),
        "readiness": {
            "metadata_files": len(rows),
            "ready_for_ingest_minimal": len([row for row in rows if row.get("has_audio")]),
            "ready_for_enrichment": len(ready_rows),
            "duplicate_groups": len(duplicate_groups),
        },
        "duplicate_groups": duplicate_groups[:20],
        "problem_rows": [
            row
            for row in rows
            if not row.get("has_audio", True)
            or not row.get("has_cover", True)
            or not row.get("has_lyrics", True)
            or not row.get("has_release_year", True)
            or row.get("valid_json") is False
        ],
    }


def summarize_ingest_queue(limit: int = 200) -> dict[str, Any]:
    """Summarize queued enrichment jobs, if the queue module is available."""

    try:
        from services.ingest_queue import list_jobs
    except Exception as exc:
        return {"available": False, "error": str(exc)}

    jobs = list_jobs(limit=limit)
    by_status = Counter(str(job.get("status") or "unknown") for job in jobs)
    invalid = [job for job in jobs if job.get("valid") is False]
    return {
        "available": True,
        "jobs": len(jobs),
        "by_status": dict(by_status),
        "invalid_jobs": [
            {
                "job_id": job.get("job_id"),
                "status": job.get("status"),
                "validation_error": job.get("validation_error"),
            }
            for job in invalid[:20]
        ],
    }


def summarize_knowledge_store(path: Path = DEFAULT_KNOWLEDGE_STORE) -> dict[str, Any]:
    """Summarize the offline SQLite knowledge cards without requiring Neo4j."""

    if not path.exists():
        return {"available": False, "path": str(path), "artist_cards": 0, "song_cards": 0}

    try:
        import sqlite3

        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        with conn:
            artist = conn.execute(
                """
                SELECT count(*) AS total,
                       sum(CASE WHEN coalesce(summary, '') <> '' THEN 1 ELSE 0 END) AS with_summary,
                       sum(CASE WHEN confidence >= 0.55 THEN 1 ELSE 0 END) AS trusted
                FROM artist_cards
                """
            ).fetchone()
            song = conn.execute(
                """
                SELECT count(*) AS total,
                       sum(CASE WHEN coalesce(summary, '') <> '' THEN 1 ELSE 0 END) AS with_summary,
                       sum(CASE WHEN release_year IS NOT NULL THEN 1 ELSE 0 END) AS with_release_year,
                       sum(CASE WHEN confidence >= 0.55 THEN 1 ELSE 0 END) AS trusted
                FROM song_cards
                """
            ).fetchone()
            sources = conn.execute(
                """
                SELECT provider, count(*) AS n
                FROM sources
                GROUP BY provider
                ORDER BY n DESC
                """
            ).fetchall()
    except Exception as exc:
        return {"available": False, "path": str(path), "error": str(exc)[:200]}

    return {
        "available": True,
        "path": str(path),
        "artist_cards": int(artist["total"] or 0),
        "artist_cards_with_summary": int(artist["with_summary"] or 0),
        "trusted_artist_cards": int(artist["trusted"] or 0),
        "song_cards": int(song["total"] or 0),
        "song_cards_with_summary": int(song["with_summary"] or 0),
        "song_cards_with_release_year": int(song["with_release_year"] or 0),
        "trusted_song_cards": int(song["trusted"] or 0),
        "sources_by_provider": {str(row["provider"] or "unknown"): int(row["n"] or 0) for row in sources},
    }


def summarize_qdrant_health(
    *,
    base_url: str = "http://localhost:6333",
    collection: str = "soultuner_music_knowledge",
    timeout: float = 0.8,
) -> dict[str, Any]:
    """Return best-effort Qdrant status for the knowledge-card vector mirror."""

    base = str(base_url or "").rstrip("/")
    try:
        with urlopen(f"{base}/collections/{collection}", timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8") or "{}")
        result = body.get("result") or {}
        points_count = result.get("points_count")
        vectors_count = result.get("vectors_count")
        status = result.get("status") or body.get("status") or "available"
        return {
            "available": True,
            "url": base,
            "collection": collection,
            "status": status,
            "points_count": int(points_count or vectors_count or 0),
        }
    except (OSError, URLError, ValueError, json.JSONDecodeError) as exc:
        return {
            "available": False,
            "url": base,
            "collection": collection,
            "error": str(exc)[:200],
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit P11 data flywheel readiness.")
    parser.add_argument("--online-root", default=str(DEFAULT_ONLINE_ROOT))
    parser.add_argument("--queue-limit", type=int, default=200)
    parser.add_argument("--knowledge-store", default=str(DEFAULT_KNOWLEDGE_STORE))
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--qdrant-collection", default="soultuner_music_knowledge")
    args = parser.parse_args()

    report = {
        "online_acquired": summarize_online_acquired(Path(args.online_root)),
        "ingest_queue": summarize_ingest_queue(limit=args.queue_limit),
        "knowledge_store": summarize_knowledge_store(Path(args.knowledge_store)),
        "qdrant": summarize_qdrant_health(
            base_url=args.qdrant_url,
            collection=args.qdrant_collection,
        ),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
