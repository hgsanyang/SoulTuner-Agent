"""Small local knowledge-card cache for optional music RAG enrichment.

This module intentionally has no network dependency.  Fetchers such as Tavily,
Netease details, or a manual curator can write normalized cards here; online
recommendation can later read the cache without waiting for web search.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from services.catalog_enrichment import normalize_knowledge_card

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_PATH = PROJECT_ROOT.parent / "data" / "knowledge_cache" / "music_knowledge.jsonl"


def knowledge_key(kind: str, title: str = "", artist: str = "") -> str:
    """Build a stable, privacy-neutral key for a song/artist knowledge card."""

    clean_kind = str(kind or "song").strip().casefold()
    clean_title = " ".join(str(title or "").strip().casefold().split())
    clean_artist = " ".join(str(artist or "").strip().casefold().split())
    if clean_kind == "artist":
        return f"artist::{clean_artist or clean_title}"
    return f"song::{clean_title}::{clean_artist}"


class MusicKnowledgeCache:
    """Append-friendly JSONL store with last-write-wins reads."""

    def __init__(self, path: Path | str = DEFAULT_CACHE_PATH):
        self.path = Path(path)

    def load_all(self) -> dict[str, dict[str, Any]]:
        if not self.path.exists():
            return {}
        rows: dict[str, dict[str, Any]] = {}
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                card = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = str(card.get("key") or "")
            if key:
                rows[key] = card
        return rows

    def upsert(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        card = normalize_knowledge_card(payload)
        key = knowledge_key(card["kind"], card.get("title", ""), card.get("artist", ""))
        card["key"] = key
        self.path.parent.mkdir(parents=True, exist_ok=True)
        rows = self.load_all()
        rows[key] = card
        self.path.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows.values()) + "\n",
            encoding="utf-8",
        )
        return card

    def get(self, *, kind: str, title: str = "", artist: str = "") -> dict[str, Any] | None:
        return self.load_all().get(knowledge_key(kind, title, artist))

    def search_terms(self, terms: Iterable[str], *, limit: int = 10) -> list[dict[str, Any]]:
        needles = [str(term or "").strip().casefold() for term in terms if str(term or "").strip()]
        if not needles:
            return []
        hits: list[tuple[int, dict[str, Any]]] = []
        for card in self.load_all().values():
            haystack = " ".join(
                [
                    str(card.get("title") or ""),
                    str(card.get("artist") or ""),
                    str(card.get("summary") or ""),
                    " ".join(card.get("facts") or []),
                ]
            ).casefold()
            score = sum(1 for needle in needles if needle in haystack)
            if score:
                hits.append((score, card))
        hits.sort(key=lambda item: (-item[0], item[1].get("key", "")))
        return [card for _score, card in hits[: max(1, int(limit))]]
