"""Offline web-to-knowledge-card enrichment for music catalog facts."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Any, Mapping

import aiohttp

from services.catalog_enrichment import build_artist_knowledge_query, build_song_knowledge_query, clamp_confidence
from services.music_knowledge_store import MusicKnowledgeStore
from tools.web_search_aggregator import fetch_searxng_search, fetch_tavily_search, fetch_zhipu_search


STYLE_KEYWORDS = {
    "Rock": ("rock", "摇滚", "alternative", "punk", "guitar"),
    "Folk": ("folk", "民谣", "acoustic", "singer-songwriter"),
    "R&B": ("r&b", "rnb", "soul", "节奏布鲁斯"),
    "Hip-Hop": ("hip-hop", "hip hop", "rap", "说唱", "嘻哈"),
    "Electronic": ("electronic", "electronica", "synth", "edm", "电子"),
    "Pop": ("pop", "流行"),
    "Indie": ("indie", "独立"),
    "Jazz": ("jazz", "爵士"),
    "Classical": ("classical", "古典"),
    "Metal": ("metal", "金属"),
    "Dream Pop": ("dream pop", "shoegaze", "梦幻流行"),
    "Lo-fi": ("lo-fi", "lofi", "低保真"),
}


@dataclass(frozen=True)
class WebSnippet:
    title: str
    content: str
    url: str
    source: str


def normalize_snippets(raw_results: list[Mapping[str, Any]]) -> list[WebSnippet]:
    snippets: list[WebSnippet] = []
    seen: set[str] = set()
    for item in raw_results:
        url = str(item.get("url") or "").strip()
        title = " ".join(str(item.get("title") or "").split())[:220]
        content = " ".join(str(item.get("content") or item.get("snippet") or "").split())[:900]
        source = str(item.get("source") or "web").strip()[:80]
        key = url or f"{title}\0{content[:80]}"
        if not content or key in seen:
            continue
        seen.add(key)
        snippets.append(WebSnippet(title=title, content=content, url=url, source=source))
    return snippets


async def fetch_music_knowledge_snippets(query: str) -> list[WebSnippet]:
    """Run federated web search and return structured snippets."""

    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(
            fetch_zhipu_search(query, session),
            fetch_tavily_search(query, session),
            fetch_searxng_search(query, session),
            return_exceptions=True,
        )
    merged: list[Mapping[str, Any]] = []
    for result in results:
        if isinstance(result, list):
            merged.extend(result)
    return normalize_snippets(merged)


def infer_style_tags(text: str, *, limit: int = 6) -> list[str]:
    lower = str(text or "").casefold()
    tags = [
        tag
        for tag, aliases in STYLE_KEYWORDS.items()
        if any(alias.casefold() in lower for alias in aliases)
    ]
    return tags[:limit]


def extract_release_year(text: str) -> int | None:
    for match in re.finditer(r"\b(19\d{2}|20\d{2}|2100)\b", str(text or "")):
        year = int(match.group(1))
        if 1900 <= year <= 2100:
            return year
    return None


def build_card_from_snippets(
    *,
    kind: str,
    title: str = "",
    artist: str = "",
    snippets: list[WebSnippet],
) -> dict[str, Any] | None:
    """Create a conservative card from web snippets without inventing facts."""

    useful = [snippet for snippet in snippets if snippet.content]
    if not useful:
        return None
    primary = useful[0]
    merged_text = " ".join(f"{snippet.title}. {snippet.content}" for snippet in useful[:5])
    facts = []
    for snippet in useful[:5]:
        sentence = re.split(r"(?<=[。.!?！？])\s*", snippet.content)[0].strip()
        if sentence and sentence not in facts:
            facts.append(sentence[:220])
    style_tags = infer_style_tags(merged_text)
    confidence = 0.72 if primary.url else 0.6
    if len(useful) >= 3:
        confidence += 0.05
    card = {
        "kind": "artist" if kind == "artist" else "song",
        "title": title,
        "artist": artist,
        "summary": primary.content[:900],
        "facts": facts[:8],
        "source": primary.source or "web",
        "source_url": primary.url,
        "confidence": clamp_confidence(confidence),
        "style_tags": style_tags,
        "source_title": primary.title,
    }
    if kind == "song":
        card["release_year"] = extract_release_year(merged_text)
    return card


async def enrich_artist_card(artist: str, *, store: MusicKnowledgeStore | None = None, dry_run: bool = False) -> dict[str, Any] | None:
    query = build_artist_knowledge_query(artist)
    snippets = await fetch_music_knowledge_snippets(query)
    card = build_card_from_snippets(kind="artist", artist=artist, title=artist, snippets=snippets)
    if card and not dry_run:
        store = store or MusicKnowledgeStore()
        store.upsert_artist_card(
            artist=artist,
            summary=card["summary"],
            style_tags=card.get("style_tags", []),
            facts=card.get("facts", []),
            source_url=card.get("source_url", ""),
            source_title=card.get("source_title", ""),
            source_provider=card.get("source", "web"),
            confidence=card.get("confidence", 0.6),
        )
    return card


async def enrich_song_card(
    title: str,
    artist: str = "",
    *,
    store: MusicKnowledgeStore | None = None,
    dry_run: bool = False,
) -> dict[str, Any] | None:
    query = build_song_knowledge_query(title, artist)
    snippets = await fetch_music_knowledge_snippets(query)
    card = build_card_from_snippets(kind="song", title=title, artist=artist, snippets=snippets)
    if card and not dry_run:
        store = store or MusicKnowledgeStore()
        store.upsert_song_card(
            title=title,
            artist=artist,
            summary=card["summary"],
            release_year=card.get("release_year"),
            style_tags=card.get("style_tags", []),
            facts=card.get("facts", []),
            source_url=card.get("source_url", ""),
            source_title=card.get("source_title", ""),
            source_provider=card.get("source", "web"),
            confidence=card.get("confidence", 0.6),
        )
    return card
