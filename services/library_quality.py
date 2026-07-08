"""Library quality helpers shared by API and tests."""

from __future__ import annotations

import re
from typing import Any, Mapping


UNKNOWN_VALUES = {None, "", "Unknown", "unknown", "未知", "未标注", "null", "none"}


def has_value(value: Any) -> bool:
    if isinstance(value, list):
        return any(has_value(item) for item in value)
    return value not in UNKNOWN_VALUES and str(value).strip() not in UNKNOWN_VALUES


def is_playable_song(song: Mapping[str, Any]) -> bool:
    if song.get("unplayable_stub") is True:
        return False
    return has_value(song.get("audio_url") or song.get("preview_url") or song.get("play_url"))


def vector_coverage_from_dims(*, muq_dim: Any = 0, m2d_dim: Any = 0, omar_dim: Any = 0) -> dict[str, bool]:
    def _int(value: Any) -> int:
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    return {
        "muq": _int(muq_dim) == 512,
        "m2d": _int(m2d_dim) == 768,
        "omar": _int(omar_dim) == 1024,
    }


def missing_fields_for_song(song: Mapping[str, Any], vector_coverage: Mapping[str, bool] | None = None) -> list[str]:
    missing: list[str] = []
    for field_name, key in (
        ("audio", "audio_url"),
        ("cover", "cover_url"),
        ("lyrics", "lrc_url"),
        ("artist", "artist"),
        ("language", "language"),
        ("release_year", "release_year"),
    ):
        if not has_value(song.get(key)):
            missing.append(field_name)
    vectors = vector_coverage or {}
    for field_name in ("muq", "m2d", "omar"):
        if not vectors.get(field_name):
            missing.append(f"{field_name}_embedding")
    return missing


def quality_score(missing_fields: list[str]) -> float:
    weights = {
        "audio": 0.24,
        "artist": 0.12,
        "language": 0.10,
        "cover": 0.07,
        "lyrics": 0.06,
        "release_year": 0.08,
        "muq_embedding": 0.12,
        "m2d_embedding": 0.10,
        "omar_embedding": 0.11,
    }
    penalty = sum(weights.get(field, 0.04) for field in missing_fields)
    return round(max(0.0, min(1.0, 1.0 - penalty)), 4)


def duplicate_key(title: Any, artist: Any = "") -> str:
    text = f"{title or ''}::{artist or ''}".casefold()
    text = re.sub(r"\([^)]*(?:live|remaster|伴奏|cover|翻唱|版)[^)]*\)", "", text)
    text = re.sub(r"（[^）]*(?:live|remaster|伴奏|cover|翻唱|版)[^）]*）", "", text)
    text = re.sub(r"[\s\-_/|｜:：,，.。]+", "", text)
    return text


def pending_asset_status(*, has_audio: bool, has_cover: bool, has_lyrics: bool) -> dict[str, Any]:
    missing = []
    if not has_audio:
        missing.append("audio")
    if not has_cover:
        missing.append("cover")
    if not has_lyrics:
        missing.append("lyrics")
    return {
        "valid": has_audio,
        "missing_assets": missing,
        "status": "ready" if has_audio else "invalid",
    }
