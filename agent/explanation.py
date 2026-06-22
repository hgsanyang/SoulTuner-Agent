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


async def emit_fast_explanation(
    recommendations: Iterable[Any],
    explanation_queue: Any = None,
) -> str:
    """Build the fast response and emit the same stream events as the graph node."""
    recommendations = list(recommendations or [])
    response = build_fast_explanation(recommendations)

    if explanation_queue:
        songs_payload = []
        for index, item in enumerate(recommendations):
            song = item.get("song", item) if isinstance(item, Mapping) else item
            if isinstance(song, Mapping) and song.get("title"):
                songs_payload.append({"song": song, "index": index})
        if songs_payload:
            await explanation_queue.put({"__songs__": songs_payload})
        await explanation_queue.put(response)
        await explanation_queue.put(None)

    return response
