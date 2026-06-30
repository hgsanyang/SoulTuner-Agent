"""Acquire a curated NetEase wishlist into the local ingestion staging area.

The script is intentionally conservative:

* parse a Markdown wishlist of ``- **01** Title —— Artist`` rows;
* search the configured local NeteaseCloudMusicApi service;
* download only one high-confidence, full-playable match per row;
* write processed_audio/audio|lyrics|covers|metadata files plus a manifest
  compatible with ``local_download_flywheel.py`` and ``ingest_to_neo4j.py``;
* report ambiguous, unavailable, and missing songs instead of guessing.

It does not bypass DRM or paid-access restrictions. If the NetEase proxy only
returns a trial URL or no URL, the row is reported as unavailable.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from dataclasses import asdict
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any
from urllib.parse import quote

import aiohttp

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings
from data.pipeline.local_download_flywheel import (
    PROCESSED_ROOT,
    TAG_DIR,
    StagedSong,
    _safe_filename,
    tag_staged,
    write_manifest,
)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


DEFAULT_WISHLIST = Path(r"D:\Users\sanyang\Desktop\新增曲目.md")
VERSION_WORDS = {
    "live",
    "remix",
    "acoustic",
    "slowed",
    "reverb",
    "demo",
    "cover",
    "karaoke",
    "instrumental",
    "伴奏",
    "翻唱",
    "现场",
    "版",
}


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _norm(text: str) -> str:
    text = str(text or "").lower()
    text = re.sub(r"\([^)]*\)|（[^）]*）", "", text)
    return re.sub(r"[\W_]+", "", text, flags=re.UNICODE)


def _norm_keep_version(text: str) -> str:
    return re.sub(r"[\W_]+", "", str(text or "").lower(), flags=re.UNICODE)


def _split_artists(text: str) -> list[str]:
    parts = re.split(r"\s*(?:/|、|,|，|&| x | X |×| feat\.? | ft\.? )\s*", str(text or ""), flags=re.I)
    return [_clean_text(p) for p in parts if _clean_text(p)]


def _has_version_marker(text: str) -> bool:
    lowered = str(text or "").lower()
    return any(word in lowered for word in VERSION_WORDS)


def _duration_ms(song: dict[str, Any]) -> int:
    return int(song.get("duration") or song.get("dt") or 0)


def parse_wishlist(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(r"^\s*-\s*\*\*(\d+)\*\*\s*(.*?)\s*——\s*(.*?)\s*$", re.M)
    rows: list[dict[str, Any]] = []
    for raw_index, title, artist_text in pattern.findall(text):
        rows.append(
            {
                "index": int(raw_index),
                "title": _clean_text(title),
                "artist_text": _clean_text(artist_text),
                "artists": _split_artists(artist_text),
            }
        )
    return rows


def _parse_indexes(spec: str) -> set[int]:
    indexes: set[int] = set()
    for part in str(spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            indexes.update(range(int(start), int(end) + 1))
        else:
            indexes.add(int(part))
    return indexes


def _artist_similarity(expected_artists: list[str], candidate_artists: list[str]) -> float:
    if not expected_artists:
        return 0.0
    scores: list[float] = []
    for expected in expected_artists:
        en = _norm_keep_version(expected)
        if not en:
            continue
        best = 0.0
        for candidate in candidate_artists:
            cn = _norm_keep_version(candidate)
            if not cn:
                continue
            if en in cn or cn in en:
                best = max(best, 1.0)
            else:
                best = max(best, SequenceMatcher(None, en, cn).ratio())
        scores.append(best)
    return sum(scores) / len(scores) if scores else 0.0


def score_candidate(row: dict[str, Any], song: dict[str, Any]) -> dict[str, Any]:
    title = song.get("name", "")
    candidate_artists = [
        a.get("name", "")
        for a in (song.get("artists") or song.get("ar") or [])
        if a.get("name")
    ]
    title_score = SequenceMatcher(None, _norm(row["title"]), _norm(title)).ratio()
    artist_score = _artist_similarity(row["artists"], candidate_artists)

    requested_version = _has_version_marker(row["title"])
    candidate_version = _has_version_marker(title)
    version_penalty = 0.0
    if candidate_version and not requested_version:
        version_penalty = -0.18
    elif requested_version and not candidate_version:
        version_penalty = -0.06

    score = title_score * 0.72 + artist_score * 0.28 + version_penalty
    return {
        "score": round(score, 4),
        "title_score": round(title_score, 4),
        "artist_score": round(artist_score, 4),
        "version_penalty": version_penalty,
        "title": title,
        "artists": candidate_artists,
        "duration": _duration_ms(song),
    }


def _is_duplicate_choice(best: dict[str, Any], second: dict[str, Any] | None) -> bool:
    if not second:
        return False
    if abs(best["match"]["score"] - second["match"]["score"]) > 0.035:
        return False
    same_title = _norm(best["song"].get("name", "")) == _norm(second["song"].get("name", ""))
    same_artists = " ".join(best["match"]["artists"]) == " ".join(second["match"]["artists"])
    duration_close = abs(best["match"]["duration"] - second["match"]["duration"]) <= 2500
    return same_title and same_artists and duration_close


async def _fetch_json(session: aiohttp.ClientSession, url: str) -> dict[str, Any]:
    async with session.get(url, timeout=settings.netease_api_timeout) as resp:
        if resp.status != 200:
            return {}
        try:
            return await resp.json()
        except Exception:
            return {}


async def _download_file(session: aiohttp.ClientSession, url: str, path: Path) -> tuple[bool, int]:
    try:
        async with session.get(url, timeout=settings.audio_download_timeout) as resp:
            if resp.status != 200:
                return False, 0
            data = await resp.read()
        if len(data) < 512 * 1024:
            return False, len(data)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return True, len(data)
    except Exception:
        return False, 0


def _load_existing_music_ids(metadata_dir: Path) -> set[str]:
    ids: set[str] = set()
    if not metadata_dir.exists():
        return ids
    for path in metadata_dir.glob("*_meta.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        music_id = str(data.get("musicId") or "").strip()
        if music_id:
            ids.add(music_id)
    return ids


def _write_metadata(path: Path, meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_lrc(path: Path, title: str, artist: str, album: str, lyric: str) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if lyric and lyric.strip():
        path.write_text(lyric, encoding="utf-8")
        return True
    path.write_text(
        f"[ti:{title}]\n[ar:{artist}]\n[al:{album}]\n"
        "[by:netease_wishlist_acquire]\n"
        "Lyrics unavailable. Infer tags from title, artist, album, and metadata.\n",
        encoding="utf-8",
    )
    return False


async def choose_candidate(
    session: aiohttp.ClientSession,
    row: dict[str, Any],
    *,
    min_score: float,
) -> tuple[dict[str, Any] | None, str, list[dict[str, Any]]]:
    query = f"{row['title']} {row['artist_text']}"
    search = await _fetch_json(
        session,
        f"{settings.netease_api_base}/search?keywords={quote(query)}&limit=10",
    )
    songs = (search.get("result") or {}).get("songs") or []
    if not songs:
        return None, "not_found", []

    scored = [{"song": song, "match": score_candidate(row, song)} for song in songs]
    scored.sort(key=lambda item: item["match"]["score"], reverse=True)
    best = scored[0]
    second = scored[1] if len(scored) > 1 else None
    match = best["match"]
    if (
        match["score"] < min_score
        or match["title_score"] < 0.82
        or (row["artists"] and match["artist_score"] < 0.42)
    ):
        return None, "low_confidence", scored[:5]
    if second and abs(match["score"] - second["match"]["score"]) <= 0.035 and not _is_duplicate_choice(best, second):
        return None, "ambiguous", scored[:5]
    return best["song"], "matched", scored[:5]


async def stage_one(
    session: aiohttp.ClientSession,
    row: dict[str, Any],
    *,
    processed_root: Path,
    existing_ids: set[str],
    min_score: float,
    dry_run: bool,
) -> tuple[StagedSong | None, dict[str, Any]]:
    song, status, candidates = await choose_candidate(session, row, min_score=min_score)
    report: dict[str, Any] = {
        "index": row["index"],
        "requested_title": row["title"],
        "requested_artist": row["artist_text"],
        "status": status,
        "candidates": [
            {
                "id": c["song"].get("id"),
                "title": c["song"].get("name"),
                "artists": c["match"]["artists"],
                "album": (c["song"].get("album") or {}).get("name", ""),
                "score": c["match"]["score"],
                "title_score": c["match"]["title_score"],
                "artist_score": c["match"]["artist_score"],
            }
            for c in candidates
        ],
    }
    if not song:
        return None, report

    song_id = str(song.get("id") or "")
    if song_id and song_id in existing_ids:
        report["status"] = "skipped_existing"
        report["matched_id"] = song_id
        return None, report

    detail_task = _fetch_json(session, f"{settings.netease_api_base}/song/detail?ids={song_id}")
    lyric_task = _fetch_json(session, f"{settings.netease_api_base}/lyric?id={song_id}")
    url_task = _fetch_json(session, f"{settings.netease_api_base}/song/url?id={song_id}&level=exhigh")
    detail_data, lyric_data, url_data = await asyncio.gather(detail_task, lyric_task, url_task)

    url_items = url_data.get("data") or []
    url_item = url_items[0] if url_items else {}
    play_url = url_item.get("url")
    if not play_url:
        report["status"] = "no_play_url"
        return None, report
    if url_item.get("freeTrialInfo") is not None:
        report["status"] = "trial_only"
        return None, report

    detail_song = ((detail_data.get("songs") or [None])[0]) or {}
    title = song.get("name") or row["title"]
    artists = [a.get("name", "") for a in (song.get("artists") or []) if a.get("name")]
    artist = "、".join(artists) or row["artist_text"] or "Unknown"
    album = (song.get("album") or {}).get("name") or (detail_song.get("al") or {}).get("name") or "Unknown"
    duration = int(song.get("duration") or detail_song.get("dt") or 0)
    ext = str(url_item.get("type") or "mp3").lower()
    if ext not in {"mp3", "flac", "m4a"}:
        ext = "mp3"

    basename = f"{_safe_filename(title)} - {_safe_filename(artist)}"
    audio_path = processed_root / "audio" / f"{basename}.{ext}"
    lrc_path = processed_root / "lyrics" / f"{basename}.lrc"
    cover_path = processed_root / "covers" / f"{basename}_cover.jpg"
    metadata_path = processed_root / "metadata" / f"{basename}_meta.json"

    if dry_run:
        report.update(
            {
                "status": "dry_run_matched",
                "matched_id": song_id,
                "matched_title": title,
                "matched_artist": artist,
                "album": album,
            }
        )
        return None, report

    if not audio_path.exists():
        downloaded, size = await _download_file(session, play_url, audio_path)
        if not downloaded:
            report["status"] = "download_failed"
            report["downloaded_bytes"] = size
            return None, report

    lyrics = (lyric_data.get("lrc") or {}).get("lyric") or ""
    has_lyrics = _write_lrc(lrc_path, title, artist, album, lyrics)
    cover_url = (detail_song.get("al") or {}).get("picUrl") or (song.get("album") or {}).get("picUrl") or ""
    if cover_url and not cover_path.exists():
        await _download_file(session, cover_url, cover_path)

    _write_metadata(
        metadata_path,
        {
            "musicId": int(song_id) if song_id.isdigit() else song_id,
            "musicName": title,
            "artist": [[name, 0] for name in artists],
            "album": album,
            "duration": duration,
            "format": ext,
            "source": "netease_wishlist",
            "wishlist_index": row["index"],
            "wishlist_requested_title": row["title"],
            "wishlist_requested_artist": row["artist_text"],
            "enriched_at": datetime.now().isoformat(),
        },
    )

    staged = StagedSong(
        filename=lrc_path.name,
        title=title,
        artist=artist,
        album=album,
        duration=duration,
        music_id=song_id,
        audio_path=str(audio_path),
        metadata_path=str(metadata_path),
        lrc_path=str(lrc_path),
        source_path=str(DEFAULT_WISHLIST),
        matched_by="netease_wishlist",
        has_lyrics=has_lyrics,
    )
    existing_ids.add(song_id)
    report.update(
        {
            "status": "downloaded",
            "matched_id": song_id,
            "matched_title": title,
            "matched_artist": artist,
            "album": album,
            "audio_path": str(audio_path),
            "has_lyrics": has_lyrics,
        }
    )
    return staged, report


async def run(args: argparse.Namespace) -> tuple[list[StagedSong], list[dict[str, Any]]]:
    rows = parse_wishlist(Path(args.wishlist))
    if args.indexes:
        selected = _parse_indexes(args.indexes)
        rows = [row for row in rows if row["index"] in selected]
    if args.limit:
        rows = rows[: args.limit]
    processed_root = Path(args.processed_root)
    existing_ids = _load_existing_music_ids(processed_root / "metadata")
    timeout = aiohttp.ClientTimeout(total=max(settings.netease_api_timeout * 6, 30))
    staged: list[StagedSong] = []
    report: list[dict[str, Any]] = []

    async with aiohttp.ClientSession(timeout=timeout) as session:
        for idx, row in enumerate(rows, 1):
            try:
                item, row_report = await stage_one(
                    session,
                    row,
                    processed_root=processed_root,
                    existing_ids=existing_ids,
                    min_score=args.min_score,
                    dry_run=args.dry_run,
                )
            except Exception as exc:
                item = None
                row_report = {
                    "index": row["index"],
                    "requested_title": row["title"],
                    "requested_artist": row["artist_text"],
                    "status": "error",
                    "error": str(exc),
                }
            report.append(row_report)
            if item:
                staged.append(item)
            print(
                f"[{idx}/{len(rows)}] {row['title']} - {row['artist_text']} => "
                f"{row_report['status']}"
            )
    return staged, report


def _write_report(report: list[dict[str, Any]]) -> Path:
    TAG_DIR.mkdir(parents=True, exist_ok=True)
    path = TAG_DIR / f"netease_wishlist_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Acquire a curated NetEase wishlist into processed_audio.")
    parser.add_argument("--wishlist", default=str(DEFAULT_WISHLIST))
    parser.add_argument("--processed-root", default=str(PROCESSED_ROOT))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--indexes", default="", help="Comma-separated wishlist indexes or ranges, e.g. 1,9,15-18.")
    parser.add_argument("--min-score", type=float, default=0.78)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--tag", action="store_true", help="Run local_download_flywheel tag generation for downloaded rows.")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--provider", default="dashscope")
    parser.add_argument("--model", default="qwen3.7-plus")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--llm-timeout", type=int, default=180, help="LLM timeout for batch tagging.")
    parser.add_argument("--replace-existing", action="store_true")
    args = parser.parse_args()

    staged, report = asyncio.run(run(args))
    report_path = _write_report(report)
    print(f"Report: {report_path}")

    manifest_path: Path | None = None
    if staged and not args.dry_run:
        manifest_path = write_manifest(staged)
        print(f"Manifest: {manifest_path}")

    if args.tag and staged and not args.dry_run:
        tagged = tag_staged(staged, args)
        print(f"Tagged: {tagged}/{len(staged)}")

    counts: dict[str, int] = {}
    for item in report:
        counts[item["status"]] = counts.get(item["status"], 0) + 1
    print("Summary:", json.dumps(counts, ensure_ascii=False, sort_keys=True))
    if manifest_path:
        print(f"Next ingest target: {manifest_path}")


if __name__ == "__main__":
    main()
