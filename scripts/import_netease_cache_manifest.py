#!/usr/bin/env python
"""Build a manifest of the local NetEase cache: what is there, what we already have.

Reads **file names and sizes only**. It never opens a ``.uc`` file, never decodes
one, and never writes to the cache directory. The cache is obfuscated on purpose
(XOR 0xA3), and stripping that is a circumvention this repo does not do — the
file name alone carries everything the manifest needs:

    {song_id}-{quality}-{md5}.uc      e.g. 1054603-320-1964b0….uc

So the manifest answers "which tracks does this cache correspond to, and which of
them are already in my library" without touching the protected bytes. Turning any
of the remainder into audio is a separate, account-entitlement-gated step that
this script does not perform.

    python scripts/import_netease_cache_manifest.py                      # dry run
    python scripts/import_netease_cache_manifest.py --out manifest.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:                     # runtime import stays lazy, inside the
    import aiohttp                    # functions that actually make requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Import states, in the order a row can move through them.
DISCOVERED = "discovered"
METADATA_READY = "metadata_ready"
METADATA_MISSING = "metadata_missing"
DUPLICATE_EXACT = "duplicate_exact"
DUPLICATE_SUSPECTED = "duplicate_suspected"

CACHE_NAME = re.compile(r"^(?P<song_id>\d+)-(?P<quality>\d+)-(?P<digest>[0-9a-f]+)\.uc!?$", re.I)
DEFAULT_CACHE_DIR = Path(os.environ.get("LOCALAPPDATA", "")) / "NetEase" / "CloudMusic" / "Cache" / "Cache"

# Two tracks with the same title+artist are only the *same recording* if their
# durations agree. Live takes, remasters and covers routinely share both fields,
# so this stays a "suspected" verdict a human resolves — never an auto-merge.
DURATION_TOLERANCE_MS = 2000


@dataclass
class CacheEntry:
    song_id: str
    quality: str
    bytes: int
    path: str
    state: str = DISCOVERED
    title: str = ""
    artist: str = ""
    album: str = ""
    duration_ms: int = 0
    lyrics: str = ""
    matched_music_id: str = ""
    matched_title: str = ""
    reasons: list[str] = field(default_factory=list)


def scan_cache(cache_dir: Path) -> list[CacheEntry]:
    """Parse file names. Nothing here opens a cache file."""
    entries: list[CacheEntry] = []
    if not cache_dir.exists():
        return entries
    for path in cache_dir.iterdir():
        match = CACHE_NAME.match(path.name)
        if not match:
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        entries.append(
            CacheEntry(
                song_id=match.group("song_id"),
                quality=match.group("quality"),
                bytes=size,
                path=str(path),
            )
        )
    return entries


async def fetch_metadata(song_ids: list[str], *, batch: int = 50) -> dict[str, dict[str, Any]]:
    """Public track metadata for a list of ids, via the local NetEase proxy.

    Anonymous — ``/song/detail`` needs no login. Anything the proxy cannot answer
    is simply absent from the result; the caller marks those rows
    ``metadata_missing`` rather than inventing values.
    """
    import aiohttp

    from config.settings import settings

    found: dict[str, dict[str, Any]] = {}
    async with aiohttp.ClientSession() as session:
        for start in range(0, len(song_ids), batch):
            chunk = song_ids[start : start + batch]
            url = f"{settings.netease_api_base}/song/detail"
            try:
                async with session.get(url, params={"ids": ",".join(chunk)}, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status != 200:
                        continue
                    payload = await response.json(content_type=None)
            except Exception:
                continue
            for track in payload.get("songs") or []:
                if not isinstance(track, dict):
                    continue
                artists = track.get("ar") or track.get("artists") or []
                names = [str(a.get("name") or "").strip() for a in artists if isinstance(a, dict)]
                album = track.get("al") or track.get("album") or {}
                found[str(track.get("id"))] = {
                    "title": str(track.get("name") or "").strip(),
                    "artist": "、".join(n for n in names if n),
                    "album": str(album.get("name") or "").strip() if isinstance(album, dict) else "",
                    "duration_ms": int(track.get("dt") or track.get("duration") or 0),
                }
    return found


async def fetch_lyrics(song_ids: list[str], *, batch: int = 20) -> dict[str, str]:
    """Fetch LRC lyrics for a list of song ids, via the local NetEase proxy.

    Anonymous — ``/lyric`` needs no login. Returns a mapping from song_id to
    the raw LRC string. Songs without lyrics (instrumental, proxy failure) are
    simply absent from the result.
    """
    import aiohttp

    from config.settings import settings

    found: dict[str, str] = {}
    async with aiohttp.ClientSession() as session:
        for start in range(0, len(song_ids), batch):
            chunk = song_ids[start : start + batch]
            tasks = []
            for sid in chunk:
                url = f"{settings.netease_api_base}/lyric"
                tasks.append(_fetch_lyric_one(session, url, sid))
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for sid, result in zip(chunk, results):
                if isinstance(result, str) and result.strip():
                    found[sid] = result
    return found


async def _fetch_lyric_one(
    session: aiohttp.ClientSession, url: str, song_id: str,
) -> str:
    """Fetch lyrics for a single song id. Returns empty string on failure."""
    import aiohttp

    try:
        async with session.get(
            url, params={"id": song_id}, timeout=aiohttp.ClientTimeout(total=15),
        ) as response:
            if response.status != 200:
                return ""
            payload = await response.json(content_type=None)
    except Exception:
        return ""
    return str((payload.get("lrc") or {}).get("lyric") or "")


def load_library_index(rows: list[dict[str, Any]] | None = None) -> tuple[dict[str, dict], dict[str, list[dict]]]:
    """Two lookups over the local catalogue: by NetEase id, and by title+artist."""
    from services.negative_feedback import song_key

    if rows is None:
        rows = _query_library()

    by_id: dict[str, dict] = {}
    by_key: dict[str, list[dict]] = {}
    for row in rows or []:
        for candidate in (row.get("music_id"), row.get("source_id")):
            value = str(candidate or "").strip()
            if value.isdigit():
                by_id[value] = row
        by_key.setdefault(song_key(row.get("title"), row.get("artist")), []).append(row)
    return by_id, by_key


def _query_library() -> list[dict[str, Any]]:
    from retrieval.neo4j_client import get_neo4j_client

    return [
        dict(row)
        for row in get_neo4j_client().execute_query(
            """
            MATCH (s:Song)
            OPTIONAL MATCH (s)-[:PERFORMED_BY]->(a:Artist)
            RETURN coalesce(toString(s.music_id), '') AS music_id,
                   coalesce(toString(s.source_id), '') AS source_id,
                   s.title AS title,
                   coalesce(a.name, s.artist, '') AS artist,
                   coalesce(s.duration, 0) AS duration
            """,
            {},
        )
    ]


def classify(
    entries: Iterable[CacheEntry],
    metadata: dict[str, dict[str, Any]],
    by_id: dict[str, dict],
    by_key: dict[str, list[dict]],
    lyrics: dict[str, str] | None = None,
) -> list[CacheEntry]:
    """Attach metadata, lyrics and decide duplicate status. Never merges anything."""
    from services.negative_feedback import song_key

    lyrics = lyrics or {}
    result: list[CacheEntry] = []
    for entry in entries:
        meta = metadata.get(entry.song_id)
        if meta:
            entry.title = meta["title"]
            entry.artist = meta["artist"]
            entry.album = meta["album"]
            entry.duration_ms = meta["duration_ms"]
            entry.state = METADATA_READY
        else:
            entry.state = METADATA_MISSING
            entry.reasons.append("proxy 未返回该 id 的公开元数据")
            result.append(entry)
            continue

        # Attach lyrics if available.
        entry.lyrics = lyrics.get(entry.song_id, "")

        # L1 — same NetEase id. Unambiguous.
        hit = by_id.get(entry.song_id)
        if hit:
            entry.state = DUPLICATE_EXACT
            entry.matched_music_id = str(hit.get("music_id") or "")
            entry.matched_title = str(hit.get("title") or "")
            entry.reasons.append("song_id 与本地曲库精确一致")
            result.append(entry)
            continue

        # L2 — same normalised title+artist. Only "suspected": a live take, a
        # remaster and a cover all share these two fields.
        for row in by_key.get(song_key(entry.title, entry.artist), []):
            local_ms = int(row.get("duration") or 0)
            if local_ms and local_ms < 10000:      # some rows store seconds
                local_ms *= 1000
            close = (
                abs(local_ms - entry.duration_ms) <= DURATION_TOLERANCE_MS
                if local_ms and entry.duration_ms
                else False
            )
            entry.state = DUPLICATE_SUSPECTED
            entry.matched_music_id = str(row.get("music_id") or "")
            entry.matched_title = str(row.get("title") or "")
            entry.reasons.append(
                "歌名+歌手一致且时长接近（±2s），需人工确认是否同一录音"
                if close
                else "歌名+歌手一致但时长不符，很可能是 Live/重制/翻唱，不可自动合并"
            )
            break
        result.append(entry)
    return result


def summarise(entries: list[CacheEntry]) -> dict[str, Any]:
    states = Counter(e.state for e in entries)
    by_state_bytes = Counter()
    for entry in entries:
        by_state_bytes[entry.state] += entry.bytes
    total = sum(e.bytes for e in entries)
    dup = [e for e in entries if e.state in (DUPLICATE_EXACT, DUPLICATE_SUSPECTED)]
    with_lyrics = sum(1 for e in entries if e.lyrics.strip())
    return {
        "cache_files": len(entries),
        "cache_bytes": total,
        "cache_gib": round(total / (1024 ** 3), 3),
        "by_state": dict(states.most_common()),
        "by_state_gib": {k: round(v / (1024 ** 3), 3) for k, v in by_state_bytes.most_common()},
        "already_have": len(dup),
        "already_have_gib": round(sum(e.bytes for e in dup) / (1024 ** 3), 3),
        "quality_mix": dict(Counter(e.quality for e in entries).most_common()),
        "lyrics_available": with_lyrics,
        "lyrics_missing": len(entries) - with_lyrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--out", help="write the manifest as JSONL")
    parser.add_argument("--summary-out", help="write the summary as JSON")
    parser.add_argument("--limit", type=int, default=0, help="only scan N entries (for a smoke run)")
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="skip the proxy lookup; scan file names only",
    )
    parser.add_argument(
        "--no-lyrics",
        action="store_true",
        help="skip fetching lyrics (only effective when metadata is fetched)",
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    entries = scan_cache(cache_dir)
    if args.limit:
        entries = entries[: args.limit]
    if not entries:
        print(f"没有在 {cache_dir} 找到 .uc 缓存文件")
        return 0
    print(f"扫描到 {len(entries)} 个缓存文件，"
          f"{round(sum(e.bytes for e in entries) / (1024 ** 3), 2)} GiB（只读文件名与大小）")

    metadata: dict[str, dict[str, Any]] = {}
    lyrics: dict[str, str] = {}
    if not args.no_metadata:
        ids = sorted({e.song_id for e in entries})
        print(f"向本地网易云代理查询 {len(ids)} 个 id 的公开元数据…")
        metadata = asyncio.run(fetch_metadata(ids))
        print(f"取到 {len(metadata)} 条")

        if not args.no_lyrics:
            # Only fetch lyrics for ids that have metadata.
            lyric_ids = sorted(metadata.keys())
            print(f"拉取 {len(lyric_ids)} 首歌的歌词…")
            lyrics = asyncio.run(fetch_lyrics(lyric_ids))
            print(f"取到 {len(lyrics)} 首有歌词")

    by_id, by_key = load_library_index()
    print(f"本地曲库索引：{len(by_id)} 个带网易云 id，{len(by_key)} 个歌名+歌手键")

    entries = classify(entries, metadata, by_id, by_key, lyrics)
    summary = summarise(entries)

    print("\n== 汇总 ==")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    if args.out:
        Path(args.out).write_text(
            "\n".join(json.dumps(asdict(e), ensure_ascii=False) for e in entries) + "\n",
            encoding="utf-8",
        )
        print(f"\nmanifest -> {args.out}")
    if args.summary_out:
        Path(args.summary_out).write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"summary  -> {args.summary_out}")

    print("\n本脚本不下载、不删除、不解码任何缓存文件。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
