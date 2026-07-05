"""Shared tag hygiene for catalog ingestion and manual library editing.

The policy is intentionally conservative: it never invents tags and never
forces a fixed count.  It only cleans, deduplicates, and caps user/LLM supplied
values so downstream retrieval sees stable categorical signals.
"""

from __future__ import annotations

from typing import Any, Iterable


TAG_FIELDS = ("genres", "moods", "themes", "scenarios")
MAX_TAGS_PER_FIELD = 5
MAX_TAG_LENGTH = 80
UNKNOWN_TAGS = {"", "unknown", "none", "null", "n/a", "na", "未知", "未标注", "无"}


def clean_tag_values(
    values: Iterable[Any] | None,
    *,
    max_tags: int = MAX_TAGS_PER_FIELD,
    max_length: int = MAX_TAG_LENGTH,
) -> list[str]:
    """Return clean tag values without forcing the list to be non-empty."""

    out: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        text = str(value or "").strip().strip(",，/|;；")
        text = " ".join(text.split())
        key = text.casefold()
        if key in UNKNOWN_TAGS or key in seen:
            continue
        seen.add(key)
        out.append(text[:max_length])
        if len(out) >= max(0, int(max_tags)):
            break
    return out


def clean_tag_payload(payload: dict[str, Any]) -> dict[str, list[str]]:
    """Clean the four user-facing categorical tag fields."""

    return {field: clean_tag_values(payload.get(field)) for field in TAG_FIELDS}

