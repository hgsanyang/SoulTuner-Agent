"""Deterministic retrieval fixture for infrastructure-free end-to-end checks."""

from __future__ import annotations

from schemas.music_state import ToolOutput

MOCK_SONGS = [
    {
        "title": "First Light",
        "artist": "SoulTuner Demo",
        "audio_url": "/static/mock/first-light.mp3",
        "language": "Instrumental",
        "genres": ["ambient", "electronic"],
        "moods": ["平静", "治愈"],
        "scenarios": ["学习", "冥想"],
    },
    {
        "title": "City After Rain",
        "artist": "SoulTuner Demo",
        "audio_url": "/static/mock/city-after-rain.mp3",
        "language": "Japanese",
        "genres": ["pop"],
        "moods": ["温柔", "怀旧"],
        "scenarios": ["通勤", "下雨天"],
    },
    {
        "title": "Open Highway",
        "artist": "SoulTuner Demo",
        "audio_url": "/static/mock/open-highway.mp3",
        "language": "English",
        "genres": ["rock"],
        "moods": ["热血"],
        "scenarios": ["开车", "旅行"],
    },
]


def mock_retrieve(query: str, limit: int) -> ToolOutput:
    items = [
        {
            "song": song,
            "reason": f"Mock retrieval fixture for: {query}",
            "similarity_score": round(0.95 - index * 0.05, 2),
        }
        for index, song in enumerate(MOCK_SONGS[:limit])
    ]
    markdown = "\n".join(
        f"{index}. **{item['song']['title']}** - {item['song']['artist']}"
        for index, item in enumerate(items, 1)
    )
    return ToolOutput(success=True, data=items, raw_markdown=markdown)
