"""Stage yt-dlp manual downloads into the canonical processed_audio layout.

The manual downloader writes files under ``data/yt_dlp_manual/downloads`` as
triplets like ``0001 - Title - Artist.mp3``, ``.info.json``, and cover images.
This script turns those files into the same ``processed_audio`` structure used
by ``ingest_to_neo4j.py``:

* ``processed_audio/audio/<Title - Artist>.<ext>``
* ``processed_audio/metadata/<Title - Artist>_meta.json``
* ``processed_audio/covers/<Title - Artist>_cover.jpg``
* ``processed_audio/lyrics/<Title - Artist>.lrc``

It intentionally does not write Neo4j directly.  The GPU-heavy vector work stays
in ``ingest_to_neo4j.py --manifest`` so there is one canonical graph write path.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


RAW_DIR = PROJECT_ROOT / "data" / "yt_dlp_manual" / "downloads"
PROCESSED_ROOT = WORKSPACE_ROOT / "data" / "processed_audio"
REPORT_DIR = PROJECT_ROOT / "data" / "pipeline" / "gemini_prompts"

SUPPORTED_AUDIO = {".mp3", ".flac", ".wav", ".m4a", ".ogg", ".opus", ".webm"}
COVER_EXTS = (".jpg", ".jpeg", ".png", ".webp")
INVALID_FILENAME_CHARS = r'\/:*?"<>|'
AUDIO_PREFERENCE = {
    ".mp3": 0,
    ".flac": 1,
    ".wav": 2,
    ".m4a": 3,
    ".ogg": 4,
    ".opus": 5,
    ".webm": 6,
}


@dataclass
class ManualSongRecord:
    raw_audio_path: str
    info_path: str
    cover_path: str
    title: str
    artist: str
    album: str
    duration: int
    music_id: str
    source_url: str
    extractor: str
    target_basename: str
    target_audio_path: str
    target_metadata_path: str
    target_cover_path: str
    target_lrc_path: str
    status: str
    reasons: list[str]


@dataclass
class StagedManualSong:
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


def _clean_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _safe_filename(text: Any) -> str:
    cleaned = _clean_text(text)
    safe = "".join(c for c in cleaned if c not in INVALID_FILENAME_CHARS).strip()
    safe = re.sub(r"\s+", " ", safe)
    return safe or "Unknown"


def _strip_index_prefix(stem: str) -> str:
    return re.sub(r"^\s*\d{1,5}\s*-\s*", "", stem).strip()


def _split_title_artist(stem: str) -> tuple[str, str]:
    stem = _strip_index_prefix(stem)
    if " - " in stem:
        title, artist = stem.rsplit(" - ", 1)
        return _clean_text(title), _clean_text(artist)
    return _clean_text(stem), "Unknown"


def _normalise_artist_text(text: Any) -> str:
    artist = _clean_text(text)
    artist = re.sub(r"\s*-\s*Topic$", "", artist, flags=re.I).strip()
    return artist or "Unknown"


def _coerce_artists(value: Any, fallback: str) -> list[str]:
    if isinstance(value, list):
        artists = [_normalise_artist_text(v) for v in value if _clean_text(v)]
    elif value:
        artists = [_normalise_artist_text(value)]
    else:
        artists = []
    if not artists and fallback:
        artists = [_normalise_artist_text(fallback)]
    return artists or ["Unknown"]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _metadata_for_audio(audio_path: Path) -> tuple[dict[str, Any], Path | None]:
    info_path = audio_path.with_suffix(".info.json")
    return _read_json(info_path), info_path if info_path.exists() else None


def _cover_for_audio(audio_path: Path) -> Path | None:
    stem = audio_path.with_suffix("")
    return _first_existing(stem.with_suffix(ext) for ext in COVER_EXTS)


def _music_id(info: dict[str, Any], fallback_basename: str) -> str:
    yt_id = _clean_text(info.get("id") or info.get("display_id"))
    if yt_id:
        return f"yt_{yt_id}"
    token = re.sub(r"[^0-9A-Za-z_\-]+", "_", fallback_basename).strip("_")
    return f"yt_manual_{token or 'unknown'}"


def _duration_ms(info: dict[str, Any]) -> int:
    duration = info.get("duration")
    try:
        return int(float(duration or 0) * 1000)
    except Exception:
        return 0


def _description_excerpt(info: dict[str, Any], limit: int = 1800) -> str:
    text = _clean_text(info.get("description", ""))
    return text[:limit]


def _language_hint(title: str, artist: str, tags: list[str]) -> str:
    haystack = " ".join([title, artist, *tags])
    if re.search(r"[\u3040-\u30ff]", haystack):
        return "Japanese"
    if re.search(r"[\uac00-\ud7af]", haystack):
        return "Korean"
    if re.search(r"[\u4e00-\u9fff]", haystack):
        return "Chinese"
    return "Unknown"


def _build_record(audio_path: Path, processed_root: Path) -> ManualSongRecord:
    info, info_path = _metadata_for_audio(audio_path)
    fallback_title, fallback_artist = _split_title_artist(audio_path.stem)

    title = _clean_text(info.get("track") or info.get("alt_title") or info.get("title") or fallback_title)
    artists = _coerce_artists(info.get("artists") or info.get("artist") or info.get("creator"), fallback_artist)
    artist = "、".join(artists)
    album = _clean_text(info.get("album") or info.get("playlist_title") or "Unknown")
    duration = _duration_ms(info)
    music_id = _music_id(info, _strip_index_prefix(audio_path.stem))
    source_url = _clean_text(info.get("webpage_url") or info.get("original_url") or info.get("url"))
    extractor = _clean_text(info.get("extractor_key") or info.get("extractor") or "yt-dlp")

    basename = f"{_safe_filename(title)} - {_safe_filename(artist)}"
    ext = audio_path.suffix.lower().lstrip(".")
    audio_target = processed_root / "audio" / f"{basename}.{ext}"
    meta_target = processed_root / "metadata" / f"{basename}_meta.json"
    cover_target = processed_root / "covers" / f"{basename}_cover.jpg"
    lrc_target = processed_root / "lyrics" / f"{basename}.lrc"

    reasons: list[str] = []
    if not info_path:
        reasons.append("missing_info_json")
    if not _cover_for_audio(audio_path):
        reasons.append("missing_cover")
    if _processed_audio_exists(processed_root, basename):
        status = "already_processed"
    else:
        status = "stage_needed"

    return ManualSongRecord(
        raw_audio_path=str(audio_path),
        info_path=str(info_path or ""),
        cover_path=str(_cover_for_audio(audio_path) or ""),
        title=title or fallback_title,
        artist=artist,
        album=album,
        duration=duration,
        music_id=music_id,
        source_url=source_url,
        extractor=extractor,
        target_basename=basename,
        target_audio_path=str(audio_target),
        target_metadata_path=str(meta_target),
        target_cover_path=str(cover_target),
        target_lrc_path=str(lrc_target),
        status=status,
        reasons=reasons,
    )


def _processed_audio_exists(processed_root: Path, basename: str) -> bool:
    audio_dir = processed_root / "audio"
    if not audio_dir.exists():
        return False
    for ext in SUPPORTED_AUDIO:
        if (audio_dir / f"{basename}{ext}").exists():
            return True
    return False


def iter_audio_files(raw_dir: Path, since_date: str | None = None, limit: int | None = None) -> list[Path]:
    since = datetime.fromisoformat(since_date) if since_date else None
    files = [path for path in raw_dir.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO]
    if since:
        files = [path for path in files if datetime.fromtimestamp(path.stat().st_mtime) >= since]
    files.sort(key=lambda path: (_strip_index_prefix(path.stem).lower(), path.suffix.lower()))
    return files[:limit] if limit else files


def build_records(raw_dir: Path, processed_root: Path, since_date: str | None = None, limit: int | None = None) -> list[ManualSongRecord]:
    return [_build_record(path, processed_root) for path in iter_audio_files(raw_dir, since_date, limit)]


def summarise(records: list[ManualSongRecord]) -> dict[str, Any]:
    by_status: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    for record in records:
        by_status[record.status] = by_status.get(record.status, 0) + 1
        for reason in record.reasons:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    return {
        "total_audio": len(records),
        "by_status": by_status,
        "reason_counts": reason_counts,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_audit_report(records: list[ManualSongRecord]) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / f"yt_dlp_manual_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    _write_json(path, {"summary": summarise(records), "records": [asdict(r) for r in records]})
    return path


def _write_lrc_placeholder(path: Path, record: ManualSongRecord, info: dict[str, Any]) -> bool:
    tags = info.get("tags") if isinstance(info.get("tags"), list) else []
    language = _language_hint(record.title, record.artist, [str(t) for t in tags])
    lines = [
        f"[ti:{record.title}]",
        f"[ar:{record.artist}]",
        f"[al:{record.album}]",
        "[by:yt_dlp_manual_flywheel]",
        "Lyrics unavailable. Infer tags from title, artist, album, and source metadata.",
        f"Language hint: {language}",
        f"Source platform: {record.extractor}",
    ]
    if tags:
        lines.append("Tags: " + ", ".join(str(t) for t in tags[:12]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return False


def _write_metadata(record: ManualSongRecord, info: dict[str, Any], audio_ext: str) -> None:
    artists = [part.strip() for part in re.split(r"\s*(?:、|,|，|/|&| x | X |×)\s*", record.artist) if part.strip()]
    tags = info.get("tags") if isinstance(info.get("tags"), list) else []
    payload = {
        "musicId": record.music_id,
        "musicName": record.title,
        "artist": [[name, 0] for name in (artists or [record.artist])],
        "album": record.album,
        "duration": record.duration,
        "format": audio_ext,
        "source": "yt_dlp_manual",
        "dataset": "yt_dlp_manual",
        "raw_source_path": record.raw_audio_path,
        "source_url": record.source_url,
        "extractor": record.extractor,
        "yt_dlp_id": record.music_id.replace("yt_", "", 1),
        "yt_dlp_title": info.get("title", ""),
        "yt_dlp_channel": info.get("channel") or info.get("uploader") or "",
        "yt_dlp_tags": tags[:50],
        "release_year": info.get("release_year") or "",
        "language": _language_hint(record.title, record.artist, [str(t) for t in tags]),
        "enriched_at": datetime.now().isoformat(),
    }
    _write_json(Path(record.target_metadata_path), payload)


def dedupe_records(records: list[ManualSongRecord]) -> list[ManualSongRecord]:
    """Keep one canonical staged file per music_id, preferring stable audio formats."""
    best_by_id: dict[str, tuple[tuple[int, int, str], int, ManualSongRecord]] = {}
    for index, record in enumerate(records):
        key = record.music_id or record.target_basename
        suffix = Path(record.raw_audio_path).suffix.lower()
        rank = (
            AUDIO_PREFERENCE.get(suffix, 99),
            0 if record.status == "stage_needed" else 1,
            record.raw_audio_path.lower(),
        )
        previous = best_by_id.get(key)
        if previous is None or rank < previous[0]:
            best_by_id[key] = (rank, index, record)

    chosen_indexes = {entry[1] for entry in best_by_id.values()}
    return [record for index, record in enumerate(records) if index in chosen_indexes]


def stage_records(records: list[ManualSongRecord], include_existing: bool = False, overwrite: bool = False) -> list[StagedManualSong]:
    staged: list[StagedManualSong] = []
    for record in dedupe_records(records):
        if record.status == "already_processed" and not include_existing:
            continue

        raw_audio = Path(record.raw_audio_path)
        target_audio = Path(record.target_audio_path)
        info = _read_json(Path(record.info_path)) if record.info_path else {}
        audio_ext = raw_audio.suffix.lower().lstrip(".")

        target_audio.parent.mkdir(parents=True, exist_ok=True)
        if overwrite or not target_audio.exists():
            shutil.copy2(raw_audio, target_audio)

        if record.cover_path:
            cover_target = Path(record.target_cover_path)
            cover_target.parent.mkdir(parents=True, exist_ok=True)
            if overwrite or not cover_target.exists():
                shutil.copy2(record.cover_path, cover_target)

        _write_lrc_placeholder(Path(record.target_lrc_path), record, info)
        _write_metadata(record, info, audio_ext)

        staged.append(
            StagedManualSong(
                filename=Path(record.target_lrc_path).name,
                title=record.title,
                artist=record.artist,
                album=record.album,
                duration=record.duration,
                music_id=record.music_id,
                audio_path=record.target_audio_path,
                metadata_path=record.target_metadata_path,
                lrc_path=record.target_lrc_path,
                source_path=record.raw_audio_path,
                matched_by="yt_dlp_info" if record.info_path else "filename",
                has_lyrics=False,
            )
        )
    return staged


def write_manifest(staged: list[StagedManualSong]) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / f"yt_dlp_manual_manifest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    _write_json(path, [asdict(song) for song in staged])
    return path


def tag_staged_manifest(staged: list[StagedManualSong], args: argparse.Namespace) -> int:
    """Reuse the existing local-download LLM tagging path for staged files."""
    from data.pipeline.local_download_flywheel import StagedSong, tag_staged

    compatible = [
        StagedSong(
            filename=song.filename,
            title=song.title,
            artist=song.artist,
            album=song.album,
            duration=song.duration,
            music_id=song.music_id,
            audio_path=song.audio_path,
            metadata_path=song.metadata_path,
            lrc_path=song.lrc_path,
            source_path=song.source_path,
            matched_by=song.matched_by,
            has_lyrics=song.has_lyrics,
        )
        for song in staged
    ]
    return tag_staged(compatible, args)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage yt-dlp manual downloads for SoulTuner ingestion.")
    parser.add_argument("--raw-dir", default=str(RAW_DIR))
    parser.add_argument("--processed-root", default=str(PROCESSED_ROOT))
    parser.add_argument("--since-date", default=None, help="Only process files modified after YYYY-MM-DD.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Only audit files and write an audit report.")
    parser.add_argument("--stage", action="store_true", help="Copy missing files into processed_audio and write a manifest.")
    parser.add_argument("--include-existing", action="store_true", help="Include already processed files in the manifest.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite staged audio/cover/metadata files.")
    parser.add_argument("--skip-llm-tags", action="store_true", help="Do not call the LLM tagger after staging.")
    parser.add_argument("--replace-existing", action="store_true", help="Regenerate tags for songs already in gemini_result.json.")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--provider", default="dashscope")
    parser.add_argument("--model", default="qwen3.7-plus")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--llm-timeout", type=int, default=180)
    args = parser.parse_args()

    records = build_records(Path(args.raw_dir), Path(args.processed_root), args.since_date, args.limit)
    report = write_audit_report(records)
    summary = summarise(records)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Audit report: {report}")

    if args.dry_run or not args.stage:
        print("Dry run complete. Use --stage to copy files into processed_audio.")
        return

    staged = stage_records(records, include_existing=args.include_existing, overwrite=args.overwrite)
    manifest = write_manifest(staged)
    print(f"Staged: {len(staged)}")
    print(f"Manifest: {manifest}")
    if not staged:
        return

    if args.skip_llm_tags:
        print("Skipped LLM tags.")
        return

    tagged = tag_staged_manifest(staged, args)
    print(f"Tagged: {tagged}/{len(staged)}")


if __name__ == "__main__":
    main()
