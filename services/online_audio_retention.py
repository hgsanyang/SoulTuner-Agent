"""Retention helpers for online-acquired audio files."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any


SUPPORTED_AUDIO_EXTS = {"mp3", "flac", "wav", "m4a", "aac", "ogg"}


def _safe_basename(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if Path(text).name != text or any(sep in text for sep in ("/", "\\")):
        raise ValueError("Invalid online audio basename")
    return text


def _artist_string(meta: dict[str, Any]) -> str:
    artists = meta.get("artist") or []
    if isinstance(artists, str):
        return artists.strip()
    if isinstance(artists, list):
        names = []
        for item in artists:
            if isinstance(item, list) and item:
                names.append(str(item[0] or "").strip())
            elif isinstance(item, dict):
                names.append(str(item.get("name") or "").strip())
            else:
                names.append(str(item or "").strip())
        return "、".join(name for name in names if name)
    return ""


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _candidate_meta_paths(
    root: Path,
    *,
    file_basename: str = "",
    title: str = "",
    artist: str = "",
    song_id: str = "",
) -> list[Path]:
    safe_file_basename = _safe_basename(file_basename) if file_basename else ""
    meta_dir = root / "metadata"
    if not meta_dir.exists():
        return []
    if safe_file_basename:
        return [meta_dir / f"{safe_file_basename}_meta.json"]
    title_key = str(title or "").strip().casefold()
    artist_key = str(artist or "").strip().casefold()
    song_id_key = str(song_id or "").strip()
    matches: list[Path] = []
    for path in sorted(meta_dir.glob("*_meta.json"), key=lambda item: item.stat().st_mtime, reverse=True):
        meta = _load_json(path)
        if not meta:
            continue
        meta_id = str(meta.get("musicId") or meta.get("source_id") or "").strip()
        if song_id_key and meta_id == song_id_key:
            matches.append(path)
            continue
        if title_key and artist_key:
            meta_title = str(meta.get("musicName") or meta.get("title") or "").strip().casefold()
            meta_artist = _artist_string(meta).casefold()
            if meta_title == title_key and meta_artist == artist_key:
                matches.append(path)
    return matches


def retain_online_audio(
    root: Path,
    *,
    file_basename: str = "",
    ext: str = "mp3",
    title: str = "",
    artist: str = "",
    song_id: str = "",
    retention_reason: str = "user_saved",
) -> dict[str, Any]:
    """Mark an online-acquired audio file as long-term retained.

    This does not download or re-run ingestion. It only promotes a temporary
    cached audio file after the user explicitly saves/likes/collects it.
    """

    requested_ext = str(ext or "mp3").lower().lstrip(".")
    if requested_ext not in SUPPORTED_AUDIO_EXTS:
        raise ValueError("Unsupported audio extension")

    for meta_path in _candidate_meta_paths(
        root,
        file_basename=file_basename,
        title=title,
        artist=artist,
        song_id=song_id,
    ):
        if not meta_path.exists():
            continue
        meta = _load_json(meta_path)
        if not meta:
            continue
        basename = meta_path.name[: -len("_meta.json")]
        audio_path = root / "audio" / f"{basename}.{requested_ext}"
        actual_ext = requested_ext
        if not audio_path.exists():
            for candidate_ext in SUPPORTED_AUDIO_EXTS:
                candidate = root / "audio" / f"{basename}.{candidate_ext}"
                if candidate.exists():
                    audio_path = candidate
                    actual_ext = candidate_ext
                    break
        if not audio_path.exists():
            return {
                "success": False,
                "reason": "audio_missing",
                "file_basename": basename,
                "metadata_path": str(meta_path),
            }

        now = datetime.now().isoformat()
        meta["audio_retention"] = "saved"
        meta["retention_reason"] = retention_reason
        meta["retained_at"] = now
        meta["acquire_status"] = meta.get("acquire_status") or "ready"
        meta["audio_status"] = "cached"
        meta["format"] = actual_ext
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return {
            "success": True,
            "file_basename": basename,
            "ext": actual_ext,
            "title": meta.get("musicName") or title,
            "artist": _artist_string(meta) or artist,
            "song_id": str(meta.get("musicId") or meta.get("source_id") or song_id or ""),
            "audio_url": f"/static/online_audio/{basename}.{actual_ext}",
            "retained_at": now,
        }

    return {"success": False, "reason": "metadata_not_found"}
