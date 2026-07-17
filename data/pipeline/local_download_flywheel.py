"""Stage locally downloaded music and generate graph tags with online context.

This is the controlled ingestion pre-step for files downloaded outside the app:

1. Scan raw NetEase downloads under ``data/raw_ncm``.
2. Decrypt today's ``.ncm`` files or copy normal audio files into
   ``data/processed_audio``.
3. Enrich metadata/lyrics/cover through the local NetEase API.
4. Batch-call the configured LLM to produce the tag JSON consumed by
   ``ingest_to_neo4j.py``.

The script intentionally does not write Neo4j directly. Keep graph writes and
embedding extraction in the existing ingestion script so the data flywheel has
one canonical persistence path.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import shutil
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import aiohttp
from mutagen import File as MutagenFile

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings
from data.pipeline.ncm_pipeline import decrypt_ncm


RAW_DIR = WORKSPACE_ROOT / "data" / "raw_ncm"
PROCESSED_ROOT = WORKSPACE_ROOT / "data" / "processed_audio"
TAG_DIR = PROJECT_ROOT / "data" / "pipeline" / "gemini_prompts"
TAG_RESULT_PATH = TAG_DIR / "gemini_result.json"
SUPPORTED_AUDIO = {".mp3", ".flac", ".wav", ".m4a", ".ogg"}

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def _safe_filename(text: str) -> str:
    return "".join(c for c in text if c not in r'\/:*?"<>|').strip() or "Unknown"


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _norm(text: str) -> str:
    return re.sub(r"[\W_]+", "", str(text or "").lower(), flags=re.UNICODE)


def _split_artist_title(stem: str) -> tuple[str, str]:
    """Raw NetEase exports are usually ``Artist - Title``."""
    if " - " in stem:
        artist, title = stem.split(" - ", 1)
        return _clean_text(artist), _clean_text(title)
    return "Unknown", _clean_text(stem)


def _mutagen_value(tags: Any, names: Iterable[str]) -> str:
    if not tags:
        return ""
    for name in names:
        value = tags.get(name)
        if isinstance(value, list) and value:
            return _clean_text(value[0])
        if value:
            return _clean_text(value)
    return ""


def _read_audio_tags(path: Path) -> Dict[str, str]:
    try:
        audio = MutagenFile(str(path), easy=True)
    except Exception:
        audio = None
    tags = getattr(audio, "tags", None)
    title = _mutagen_value(tags, ["title"])
    artist = _mutagen_value(tags, ["artist", "albumartist"])
    album = _mutagen_value(tags, ["album"])
    if not title or not artist:
        inferred_artist, inferred_title = _split_artist_title(path.stem)
        title = title or inferred_title
        artist = artist or inferred_artist
    return {"title": title or path.stem, "artist": artist or "Unknown", "album": album or "Unknown"}


def _duration_ms(path: Path) -> int:
    try:
        audio = MutagenFile(str(path))
        length = float(getattr(getattr(audio, "info", None), "length", 0) or 0)
        return int(length * 1000)
    except Exception:
        return 0


def _clean_lrc(text: str) -> str:
    text = re.sub(r"\[\d{2}:\d{2}\.\d{2,3}\]", "", text or "")
    text = re.sub(
        r"^\[(ar|ti|al|by|offset|hash|total|sign):.*\]$",
        "",
        text,
        flags=re.MULTILINE | re.IGNORECASE,
    )
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def _load_tag_results(path: Path = TAG_RESULT_PATH) -> List[Dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    raw = path.read_text(encoding="utf-8").strip()
    while raw and raw[-1] not in "]":
        raw = raw[:-1]
    try:
        data = json.loads(raw)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _save_tag_results(results: List[Dict[str, Any]], path: Path = TAG_RESULT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


def _parse_json_array(text: str) -> List[Dict[str, Any]]:
    text = (text or "").strip()
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0].strip()
    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end >= start:
        text = text[start : end + 1]
    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError("LLM response is not a JSON array")
    return [item for item in data if isinstance(item, dict)]


@dataclass
class StagedSong:
    filename: str
    title: str
    artist: str
    album: str
    duration: int
    music_id: str
    audio_path: str
    metadata_path: str
    lrc_path: str
    source_path: str
    matched_by: str
    has_lyrics: bool


async def _fetch_json(session: aiohttp.ClientSession, url: str) -> Dict[str, Any]:
    async with session.get(url, timeout=settings.netease_api_timeout) as resp:
        if resp.status != 200:
            return {}
        try:
            return await resp.json()
        except Exception:
            return {}


def _score_match(query_title: str, query_artist: str, song: Dict[str, Any]) -> float:
    title = song.get("name", "")
    artists = " ".join(a.get("name", "") for a in song.get("artists", []) or song.get("ar", []))
    title_score = SequenceMatcher(None, _norm(query_title), _norm(title)).ratio()
    artist_score = SequenceMatcher(None, _norm(query_artist), _norm(artists)).ratio() if query_artist else 0.0
    return title_score * 0.75 + artist_score * 0.25


async def _lookup_netease(
    session: aiohttp.ClientSession,
    title: str,
    artist: str,
) -> Dict[str, Any]:
    query = _clean_text(f"{title} {artist}")
    if not query:
        return {}
    url = f"{settings.netease_api_base}/search?keywords={query}&limit=5"
    data = await _fetch_json(session, url)
    songs = data.get("result", {}).get("songs", []) if data else []
    if not songs:
        return {}
    best = max(songs, key=lambda song: _score_match(title, artist, song))
    if _score_match(title, artist, best) < 0.45:
        return {}

    song_id = str(best.get("id") or "")
    detail = {}
    lyrics = ""
    if song_id:
        detail_data, lyric_data = await asyncio.gather(
            _fetch_json(session, f"{settings.netease_api_base}/song/detail?ids={song_id}"),
            _fetch_json(session, f"{settings.netease_api_base}/lyric?id={song_id}"),
        )
        detail_songs = detail_data.get("songs", []) if detail_data else []
        detail = detail_songs[0] if detail_songs else {}
        lyrics = (lyric_data.get("lrc") or {}).get("lyric") or ""

    artists = [a.get("name", "") for a in best.get("artists", []) if a.get("name")]
    return {
        "music_id": song_id,
        "title": best.get("name") or title,
        "artists": artists or ([artist] if artist else ["Unknown"]),
        "album": (best.get("album") or {}).get("name") or title,
        "duration": int(best.get("duration") or 0),
        "cover_url": (detail.get("al") or {}).get("picUrl", ""),
        "lyrics": lyrics,
    }


async def _download_file(session: aiohttp.ClientSession, url: str, path: Path) -> bool:
    if not url:
        return False
    try:
        async with session.get(url, timeout=settings.audio_download_timeout) as resp:
            if resp.status != 200:
                return False
            path.write_bytes(await resp.read())
            return True
    except Exception:
        return False


def _write_metadata(path: Path, meta: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_lrc(path: Path, title: str, artist: str, album: str, lyrics: str) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if lyrics and _clean_lrc(lyrics):
        path.write_text(lyrics, encoding="utf-8")
        return True
    placeholder = (
        f"[ti:{title}]\n[ar:{artist}]\n[al:{album}]\n"
        "[by:local_download_flywheel]\n"
        "Lyrics unavailable. Infer tags from title, artist, album, and online metadata.\n"
    )
    path.write_text(placeholder, encoding="utf-8")
    return False


async def _stage_audio_file(
    path: Path,
    processed_root: Path,
    session: aiohttp.ClientSession,
) -> Optional[StagedSong]:
    raw_info = _read_audio_tags(path)
    lookup = await _lookup_netease(session, raw_info["title"], raw_info["artist"])
    title = lookup.get("title") or raw_info["title"]
    artists = lookup.get("artists") or [raw_info["artist"]]
    artist = "、".join(a for a in artists if a) or "Unknown"
    album = lookup.get("album") or raw_info["album"] or "Unknown"
    duration = int(lookup.get("duration") or _duration_ms(path))
    music_id = str(lookup.get("music_id") or f"local_{_safe_filename(title)}_{_safe_filename(artist)}")

    basename = f"{_safe_filename(title)} - {_safe_filename(artist)}"
    ext = path.suffix.lower().lstrip(".")
    audio_path = processed_root / "audio" / f"{basename}.{ext}"
    metadata_path = processed_root / "metadata" / f"{basename}_meta.json"
    cover_path = processed_root / "covers" / f"{basename}_cover.jpg"
    lrc_path = processed_root / "lyrics" / f"{basename}.lrc"

    audio_path.parent.mkdir(parents=True, exist_ok=True)
    if not audio_path.exists():
        shutil.copy2(path, audio_path)

    if lookup.get("cover_url") and not cover_path.exists():
        cover_path.parent.mkdir(parents=True, exist_ok=True)
        await _download_file(session, lookup["cover_url"], cover_path)

    has_lyrics = _write_lrc(lrc_path, title, artist, album, lookup.get("lyrics", ""))
    _write_metadata(
        metadata_path,
        {
            "musicId": int(music_id) if music_id.isdigit() else music_id,
            "musicName": title,
            "artist": [[name, 0] for name in artists],
            "album": album,
            "duration": duration,
            "format": ext,
            "source": "local_download",
            "raw_source_path": str(path),
            "enriched_at": datetime.now().isoformat(),
        },
    )
    return StagedSong(
        filename=lrc_path.name,
        title=title,
        artist=artist,
        album=album,
        duration=duration,
        music_id=music_id,
        audio_path=str(audio_path),
        metadata_path=str(metadata_path),
        lrc_path=str(lrc_path),
        source_path=str(path),
        matched_by="netease" if lookup else "filename",
        has_lyrics=has_lyrics,
    )


async def _stage_ncm_file(
    path: Path,
    processed_root: Path,
    session: aiohttp.ClientSession,
) -> Optional[StagedSong]:
    output_path, meta, _skipped = decrypt_ncm(str(path), str(processed_root))
    audio_path = Path(output_path)
    title = meta.get("musicName") or audio_path.stem.split(" - ")[0]
    artist_names = [a[0] for a in meta.get("artist", []) if a] or ["Unknown"]
    artist = "、".join(artist_names)
    album = meta.get("album", "Unknown")
    music_id = str(meta.get("musicId") or f"local_{audio_path.stem}")

    lookup: Dict[str, Any] = {}
    if not meta.get("musicId"):
        lookup = await _lookup_netease(session, title, artist)
        if lookup:
            music_id = str(lookup.get("music_id") or music_id)
            title = lookup.get("title") or title
            artist_names = lookup.get("artists") or artist_names
            artist = "、".join(artist_names)
            album = lookup.get("album") or album

    basename = audio_path.stem
    metadata_path = processed_root / "metadata" / f"{basename}_meta.json"
    lrc_path = processed_root / "lyrics" / f"{basename}.lrc"
    cover_path = processed_root / "covers" / f"{basename}_cover.jpg"
    lyrics = lookup.get("lyrics", "")
    if not lyrics and meta.get("musicId"):
        lyric_data = await _fetch_json(session, f"{settings.netease_api_base}/lyric?id={meta['musicId']}")
        lyrics = (lyric_data.get("lrc") or {}).get("lyric") or ""
    has_lyrics = _write_lrc(lrc_path, title, artist, album, lyrics)
    if lookup.get("cover_url") and not cover_path.exists():
        await _download_file(session, lookup["cover_url"], cover_path)
    if metadata_path.exists():
        try:
            current = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            current = {}
    else:
        current = {}
    current.update(
        {
            "musicId": int(music_id) if music_id.isdigit() else music_id,
            "musicName": title,
            "artist": [[name, 0] for name in artist_names],
            "album": album,
            "duration": int(current.get("duration") or meta.get("duration") or lookup.get("duration") or 0),
            "format": audio_path.suffix.lstrip("."),
            "source": "local_download",
            "raw_source_path": str(path),
            "enriched_at": datetime.now().isoformat(),
        }
    )
    _write_metadata(metadata_path, current)
    return StagedSong(
        filename=lrc_path.name,
        title=title,
        artist=artist,
        album=album,
        duration=int(current.get("duration") or 0),
        music_id=music_id,
        audio_path=str(audio_path),
        metadata_path=str(metadata_path),
        lrc_path=str(lrc_path),
        source_path=str(path),
        matched_by="ncm_meta" if meta else ("netease" if lookup else "filename"),
        has_lyrics=has_lyrics,
    )


def _iter_raw_files(raw_dir: Path, since_date: Optional[str], limit: Optional[int]) -> List[Path]:
    since = datetime.fromisoformat(since_date) if since_date else None
    files = [
        p
        for p in raw_dir.rglob("*")
        if p.is_file() and (p.suffix.lower() == ".ncm" or p.suffix.lower() in SUPPORTED_AUDIO)
    ]
    if since:
        files = [p for p in files if datetime.fromtimestamp(p.stat().st_mtime) >= since]
    files.sort(key=lambda p: p.stat().st_mtime)
    return files[:limit] if limit else files


def _build_tag_prompt(songs: List[StagedSong]) -> str:
    blocks = []
    for song in songs:
        lrc = Path(song.lrc_path)
        lyrics = _clean_lrc(lrc.read_text(encoding="utf-8", errors="ignore") if lrc.exists() else "")
        blocks.append(
            "\n".join(
                [
                    f"【filename】: {song.filename}",
                    f"【title】: {song.title}",
                    f"【artist】: {song.artist}",
                    f"【album】: {song.album}",
                    f"【matched_by】: {song.matched_by}",
                    f"【lyrics_or_metadata】: {lyrics[:2200]}",
                ]
            )
        )
    return f"""你是 Neo4j 音乐知识图谱的构建专家。请根据歌曲元数据、联网补到的歌词/专辑信息，为以下 {len(songs)} 首歌生成稳定标签。

请严格返回纯 JSON 数组，不要 markdown，不要解释。每个对象必须包含：
- filename: 必须与输入 filename 完全一致
- moods: 1-5 个，按实际内容选择，限常见英文标签，如 Happy/Melancholy/Healing/Energetic/Relaxing/Romantic/Nostalgic/Angry/Hopeful/Dreamy/Lonely/Peaceful/Tense/Bittersweet
- themes: 0-5 个，按实际内容选择，如 Love/Heartbreak/Growth/Nature/Youth/Life/Friendship/Freedom/Loss/Self-discovery/Night/Journey/Society/Urban/Memory
- scenarios: 1-5 个，按实际适配场景选择，如 Study/Driving/Workout/Sleep/Commute/Rainy Day/Party/Cooking/Travel/Late Night/Morning/Work/Relaxing/Walking/After Breakup
- vibe: 1 个，如 Lo-fi/Cinematic/Indie/Acoustic/Dreampop/Urban/Retro/Ethereal/Raw/Warm/Dark/Bright/Rebellious
- genres: 1-5 个，按实际风格选择，如 Rock/Indie/Pop/Alternative/Folk/Electronic/J-Pop/K-Pop/Cantopop/Hip-Hop/R&B/Soul/Funk/Jazz/Classical/Ambient/Metal/Dance/Ballad/Lo-fi/Post-Rock/Punk
- language: 只能是 English/Chinese/Japanese/Korean/Cantonese/Instrumental/Mixed/Other/Unknown
- region: 只能是 Western/Mainland China/Taiwan/Hong Kong/Japan/Korea/Other/Unknown

不要为了凑数量硬填标签；不确定就少填。若歌词缺失，请根据歌名、歌手、专辑和常识推断；如果明显纯音乐，language 用 Instrumental。

输入：
{chr(10).join(blocks)}
"""


def _call_llm_for_tags(
    songs: List[StagedSong],
    provider: str,
    model: str,
    temperature: float,
    timeout: int,
) -> List[Dict[str, Any]]:
    """Use the same LangChain/OpenAI-compatible path as the main Agent.

    The native LiteLLM wrapper is useful for small scripts, but the main Agent
    already validates DashScope through ChatOpenAI-compatible requests. Reusing
    that path keeps ingestion tagging aligned with production traffic.
    """
    from llms.multi_llm import get_chat_model

    llm = get_chat_model(
        provider=provider,
        model_name=model,
        temperature=temperature,
        max_tokens=10000,
        timeout=timeout,
    )
    response = llm.invoke(
        [
            ("system", "你是专业音乐标签提取器，只返回纯 JSON 数组。"),
            ("human", _build_tag_prompt(songs)),
        ]
    )
    content = response.content if hasattr(response, "content") else str(response)
    return _parse_json_array(content)


async def stage_files(args: argparse.Namespace) -> List[StagedSong]:
    raw_files = _iter_raw_files(Path(args.raw_dir), args.since_date, args.limit)
    if args.dry_run:
        for path in raw_files:
            print(path)
        return []
    print(f"Found raw files: {len(raw_files)}")
    staged: List[StagedSong] = []
    timeout = aiohttp.ClientTimeout(total=max(settings.netease_api_timeout * 4, 20))
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for idx, path in enumerate(raw_files, 1):
            try:
                if path.suffix.lower() == ".ncm":
                    song = await _stage_ncm_file(path, Path(args.processed_root), session)
                else:
                    song = await _stage_audio_file(path, Path(args.processed_root), session)
                if song:
                    staged.append(song)
                    print(f"[{idx}/{len(raw_files)}] staged: {song.title} - {song.artist}")
            except Exception as exc:
                print(f"[{idx}/{len(raw_files)}] failed: {path.name}: {exc}")
    return staged


def write_manifest(staged: List[StagedSong]) -> Path:
    TAG_DIR.mkdir(parents=True, exist_ok=True)
    path = TAG_DIR / f"local_download_manifest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps([asdict(song) for song in staged], ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_manifest(path: Path) -> List[StagedSong]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [StagedSong(**item) for item in data]


def tag_staged(staged: List[StagedSong], args: argparse.Namespace) -> int:
    existing = _load_tag_results()
    existing_filenames = {item.get("filename") for item in existing if item.get("filename")}
    if args.replace_existing:
        staged_filenames = {song.filename for song in staged}
        existing = [item for item in existing if item.get("filename") not in staged_filenames]
        targets = staged
    else:
        targets = [song for song in staged if song.filename not in existing_filenames]
    print(f"Tag targets: {len(targets)}/{len(staged)} (replace_existing={args.replace_existing})")
    new_results: List[Dict[str, Any]] = []
    for start in range(0, len(targets), args.batch_size):
        batch = targets[start : start + args.batch_size]
        print(f"Tag batch {start // args.batch_size + 1}: {len(batch)} songs")
        try:
            batch_results = _call_llm_for_tags(batch, args.provider, args.model, args.temperature, args.llm_timeout)
        except Exception as exc:
            print(f"  LLM tag failed: {exc}")
            continue
        allowed = {song.filename for song in batch}
        filtered = [item for item in batch_results if item.get("filename") in allowed]
        new_results.extend(filtered)
        _save_tag_results(existing + new_results)
        print(f"  saved {len(filtered)} tags")
    return len(new_results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Local download data flywheel: stage, enrich, and tag.")
    parser.add_argument("--raw-dir", default=str(RAW_DIR))
    parser.add_argument("--processed-root", default=str(PROCESSED_ROOT))
    parser.add_argument("--since-date", default=None, help="Only process files modified after YYYY-MM-DD.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--provider", default="dashscope")
    parser.add_argument("--model", default="qwen3.7-plus")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--llm-timeout", type=int, default=180, help="LLM timeout for batch tagging.")
    parser.add_argument("--manifest", default=None, help="Use an existing staged manifest and only run tagging.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-llm-tags", action="store_true")
    parser.add_argument("--replace-existing", action="store_true", help="Regenerate tags for songs already in gemini_result.json.")
    args = parser.parse_args()

    if args.manifest:
        staged = load_manifest(Path(args.manifest))
        if args.limit:
            staged = staged[: args.limit]
        print(f"Loaded manifest: {args.manifest} ({len(staged)} songs)")
    else:
        staged = asyncio.run(stage_files(args))
    if args.dry_run:
        print("Dry run complete.")
        return
    if not args.manifest:
        manifest = write_manifest(staged)
        print(f"Manifest: {manifest}")
    if args.skip_llm_tags:
        print("Skipped LLM tags.")
        return
    tagged = tag_staged(staged, args)
    print(f"Tagged: {tagged}/{len(staged)}")
    print(f"Tag result: {TAG_RESULT_PATH}")


if __name__ == "__main__":
    main()
