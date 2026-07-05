"""Audit local catalog-enrichment readiness without calling LLM/GPU/network.

This is a diagnostic companion for P11.  It scans downloaded/pending assets and
reports which pieces are present before the expensive background flywheel runs.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_ONLINE_ROOT = PROJECT_ROOT.parent / "data" / "online_acquired"


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
                "title": meta.get("musicName") or meta.get("title") or "",
                "artist": artist_string(meta.get("artist") or meta.get("artists")),
                "basename": basename,
                "has_audio": has_audio,
                "has_cover": has_cover,
                "has_lyrics": has_lyrics,
                "has_release_year": has_release_year,
                "source_platform": meta.get("source_platform") or meta.get("source") or "",
            }
        )

    totals["metadata_files"] = len(rows)
    return {
        "root": str(root),
        "totals": dict(totals),
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit P11 data flywheel readiness.")
    parser.add_argument("--online-root", default=str(DEFAULT_ONLINE_ROOT))
    parser.add_argument("--queue-limit", type=int, default=200)
    args = parser.parse_args()

    report = {
        "online_acquired": summarize_online_acquired(Path(args.online_root)),
        "ingest_queue": summarize_ingest_queue(limit=args.queue_limit),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
