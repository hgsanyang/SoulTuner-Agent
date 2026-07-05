"""Offline web-to-knowledge-card enrichment for music catalog facts."""

from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import aiohttp

from services.catalog_enrichment import build_artist_knowledge_query, build_song_knowledge_query, clamp_confidence
from config.settings import settings
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

PROJECT_ENV_FILE = Path(__file__).resolve().parents[1] / ".env"


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


def _extract_json_object(text: str) -> dict[str, Any] | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    try:
        loaded = json.loads(raw)
        return loaded if isinstance(loaded, dict) else None
    except json.JSONDecodeError:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        try:
            loaded = json.loads(raw[start : end + 1])
            return loaded if isinstance(loaded, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def _load_dashscope_env() -> str:
    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ENV_FILE, override=False)
    except Exception:
        pass
    return os.getenv("DASHSCOPE_API_KEY", "").strip()


def _normalise_llm_card(
    *,
    kind: str,
    title: str = "",
    artist: str = "",
    parsed: Mapping[str, Any],
    source_url: str = "",
    source_title: str = "",
    source: str = "dashscope_web_search",
) -> dict[str, Any] | None:
    summary = str(parsed.get("summary") or "").strip()
    if not summary:
        return None
    sources = parsed.get("sources") or parsed.get("source_urls") or []
    if isinstance(sources, str):
        sources = [sources]
    first_source = source_url or next((str(item).strip() for item in sources if str(item).strip()), "")
    return {
        "kind": "artist" if kind == "artist" else "song",
        "title": title,
        "artist": artist,
        "summary": summary[:900],
        "facts": [str(item)[:220] for item in parsed.get("facts") or [] if str(item).strip()][:8],
        "source": source,
        "source_url": first_source,
        "confidence": clamp_confidence(parsed.get("confidence"), default=0.74),
        "style_tags": [str(item)[:80] for item in parsed.get("style_tags") or [] if str(item).strip()][:8],
        "source_title": source_title or "Qwen web search",
        "release_year": parsed.get("release_year"),
    }


def _dashscope_openai_client() -> Any | None:
    api_key = _load_dashscope_env()
    if not api_key:
        return None
    try:
        from openai import OpenAI

        return OpenAI(
            api_key=api_key,
            base_url=os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        )
    except Exception:
        return None


def _knowledge_llm_model() -> str:
    model = str(getattr(settings, "llm_default_model", "") or "").strip()
    return model if model.startswith("qwen3.") else "qwen3.7-plus"


def _response_text(response: Any) -> str:
    text = getattr(response, "output_text", "")
    if text:
        return str(text)
    try:
        data = response.model_dump()
    except Exception:
        data = {}
    chunks: list[str] = []
    for item in data.get("output") or []:
        for content in item.get("content") or []:
            if content.get("type") in {"output_text", "text"}:
                chunks.append(str(content.get("text") or ""))
    return "\n".join(chunks)


def _llm_web_card(
    *,
    kind: str,
    query: str,
    title: str = "",
    artist: str = "",
) -> dict[str, Any] | None:
    """Ask Qwen3.7-plus to perform web search and return a sourced card."""

    client = _dashscope_openai_client()
    if client is None or not hasattr(client, "responses"):
        return None
    subject = f"artist={artist or title}" if kind == "artist" else f"title={title}, artist={artist}"
    prompt = f"""
你是音乐资料整理助手。请调用联网搜索核对资料，只整理和音乐相关的事实，不要编造。
对象: {kind}; {subject}
搜索意图: {query}

请输出严格 JSON:
{{
  "summary": "120字以内中文摘要，覆盖风格、背景、代表性信息",
  "facts": ["最多5条可由联网结果支持的事实"],
  "style_tags": ["最多6个音乐风格/类型/场景标签"],
  "release_year": 歌曲首发年份或 null，歌手卡可为 null,
  "confidence": 0.0到1.0,
  "sources": ["至少1个用于支撑摘要的网页URL"]
}}

规则:
- 只输出 JSON，不要 Markdown。
- 如果同名歌曲/歌手有歧义，降低 confidence，并在 facts 里说明歧义。
- 发行年份优先原曲首发年份，不把重制版、精选集、Live 专辑年份当作首发年份。
""".strip()
    try:
        response = client.responses.create(
            model=_knowledge_llm_model(),
            input=prompt,
            tools=[{"type": "web_search"}],
            extra_body={"enable_thinking": True},
        )
        parsed = _extract_json_object(_response_text(response))
        if not parsed:
            return None
        return _normalise_llm_card(
            kind=kind,
            title=title,
            artist=artist,
            parsed=parsed,
            source="dashscope_web_search",
        )
    except Exception:
        return None


def _llm_card_from_snippets(
    *,
    kind: str,
    title: str = "",
    artist: str = "",
    snippets: list[WebSnippet],
) -> dict[str, Any] | None:
    """Use Qwen/DashScope to structure already-fetched snippets."""

    client = _dashscope_openai_client()
    if client is None or not snippets:
        return None
    try:
        evidence = "\n".join(
            f"[{idx + 1}] title={snippet.title}\nurl={snippet.url}\ncontent={snippet.content}"
            for idx, snippet in enumerate(snippets[:6])
        )
        subject = f"artist={artist or title}" if kind == "artist" else f"title={title}, artist={artist}"
        prompt = f"""
你是音乐资料整理助手。只根据下面联网搜索片段整理知识卡，不要编造。
对象: {kind}; {subject}

输出严格 JSON:
{{
  "summary": "120字以内中文摘要",
  "facts": ["最多5条可由片段支持的事实"],
  "style_tags": ["最多6个音乐风格/类型/场景标签"],
  "release_year": 发行年份或 null,
  "confidence": 0.0到1.0
}}

搜索片段:
{evidence}
""".strip()
        response = client.responses.create(
            model=_knowledge_llm_model(),
            input=f"你只输出 JSON，不输出解释。\n\n{prompt}",
            extra_body={"enable_thinking": True},
        )
        content = _response_text(response)
        parsed = _extract_json_object(content)
        if not parsed:
            return None
        primary = snippets[0]
        return _normalise_llm_card(
            kind=kind,
            title=title,
            artist=artist,
            parsed=parsed,
            source_url=primary.url,
            source_title=primary.title,
            source=primary.source or "web",
        )
    except Exception:
        return None


def build_card_from_snippets(
    *,
    kind: str,
    title: str = "",
    artist: str = "",
    snippets: list[WebSnippet],
    use_llm_summary: bool = False,
) -> dict[str, Any] | None:
    """Create a conservative card from web snippets without inventing facts."""

    useful = [snippet for snippet in snippets if snippet.content]
    if not useful:
        return None
    if use_llm_summary:
        llm_card = _llm_card_from_snippets(kind=kind, title=title, artist=artist, snippets=useful)
        if llm_card and llm_card.get("summary") and llm_card.get("source_url"):
            return llm_card
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


async def enrich_artist_card(
    artist: str,
    *,
    store: MusicKnowledgeStore | None = None,
    dry_run: bool = False,
    use_llm_summary: bool = False,
) -> dict[str, Any] | None:
    query = build_artist_knowledge_query(artist)
    card = None
    if use_llm_summary:
        card = _llm_web_card(kind="artist", query=query, artist=artist, title=artist)
    if card is None:
        snippets = await fetch_music_knowledge_snippets(query)
        card = build_card_from_snippets(
            kind="artist",
            artist=artist,
            title=artist,
            snippets=snippets,
            use_llm_summary=use_llm_summary,
        )
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
    use_llm_summary: bool = False,
) -> dict[str, Any] | None:
    query = build_song_knowledge_query(title, artist)
    card = None
    if use_llm_summary:
        card = _llm_web_card(kind="song", query=query, title=title, artist=artist)
    if card is None:
        snippets = await fetch_music_knowledge_snippets(query)
        card = build_card_from_snippets(
            kind="song",
            title=title,
            artist=artist,
            snippets=snippets,
            use_llm_summary=use_llm_summary,
        )
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
