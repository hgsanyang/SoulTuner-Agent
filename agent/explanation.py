"""Deterministic recommendation text for low-latency operation."""

from __future__ import annotations

from typing import Any, Iterable, Mapping


def build_fast_explanation(recommendations: Iterable[Any], limit: int = 5) -> str:
    songs: list[Mapping[str, Any]] = []
    for item in recommendations or []:
        if not isinstance(item, Mapping):
            continue
        song = item.get("song", item)
        if isinstance(song, Mapping) and song.get("title"):
            songs.append(song)
        if len(songs) >= limit:
            break

    if not songs:
        return "抱歉，没有找到合适的音乐推荐。"

    lines = [f"为你找到了 {len(songs)} 首歌："]
    for index, song in enumerate(songs, 1):
        title = str(song.get("title") or "未知歌曲")
        artist = str(song.get("artist") or "未知歌手")
        lines.append(f"{index}. 《{title}》 - {artist}")
    return "\n".join(lines)
