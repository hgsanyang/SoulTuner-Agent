"""Offline web-to-knowledge-card enrichment for music catalog facts."""

from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

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
BLOCKED_SOURCE_HOSTS = {
    "tavily.com",
    "www.tavily.com",
    "google.com",
    "www.google.com",
    "bing.com",
    "www.bing.com",
    "baidu.com",
    "www.baidu.com",
    "search.brave.com",
}
BLOCKED_SOURCE_PARENT_HOSTS = {"tavily.com"}


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
    """Run legacy federated web search and return structured snippets.

    This is kept for explicit diagnostics only.  Production knowledge-card
    enrichment uses DashScope/Qwen web_search as the default source of truth.
    """

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


def _is_traceable_source_url(url: str) -> bool:
    raw = str(url or "").strip()
    if not raw:
        return False
    try:
        parsed = urlparse(raw)
    except ValueError:
        return False
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return False
    host = parsed.netloc.casefold()
    if host in BLOCKED_SOURCE_HOSTS:
        return False
    if any(host.endswith("." + blocked) for blocked in BLOCKED_SOURCE_PARENT_HOSTS):
        return False
    if "/search" in parsed.path.casefold() and host in {"google.com", "www.google.com", "bing.com", "www.bing.com"}:
        return False
    return True


def _first_traceable_source_url(items: Any) -> str:
    if isinstance(items, str):
        items = [items]
    for item in items or []:
        raw = str(item or "").strip()
        if _is_traceable_source_url(raw):
            return raw
    return ""


def _details_from_parsed(kind: str, parsed: Mapping[str, Any]) -> dict[str, Any]:
    raw = parsed.get("details") or {}
    if not isinstance(raw, Mapping):
        raw = {}
    details = dict(raw)
    if kind == "song":
        release_year = parsed.get("release_year") or details.get("original_release_year") or details.get("release_year")
        if release_year:
            details.setdefault("original_release_year", release_year)
    return details


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
    first_source = source_url if _is_traceable_source_url(source_url) else _first_traceable_source_url(sources)
    if not first_source:
        return None
    details = _details_from_parsed(kind, parsed)
    release_year = parsed.get("release_year")
    if kind == "song" and not release_year:
        release_year = details.get("original_release_year") or details.get("release_year")
    return {
        "kind": "artist" if kind == "artist" else "song",
        "title": title,
        "artist": artist,
        "summary": summary[:900],
        "facts": [str(item)[:220] for item in parsed.get("facts") or [] if str(item).strip()][:8],
        "details": details,
        "source": source,
        "source_url": first_source,
        "confidence": clamp_confidence(parsed.get("confidence"), default=0.74),
        "style_tags": [str(item)[:80] for item in parsed.get("style_tags") or [] if str(item).strip()][:8],
        "source_title": source_title or "Qwen web search",
        "release_year": release_year,
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
            timeout=float(os.getenv("MUSIC_KNOWLEDGE_LLM_TIMEOUT_SECONDS", "120")),
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
你是音乐资料整理助手。你必须实际调用联网搜索工具核对资料，只整理搜索结果能支持的音乐事实，不要凭模型记忆编造。
对象: {kind}; {subject}
搜索意图: {query}

请输出严格 JSON:
{{
  "summary": "120字以内中文摘要，覆盖风格、背景、代表性信息",
  "facts": ["最多5条可由联网结果支持的事实"],
  "style_tags": ["最多6个音乐风格/类型/场景标签"],
  "release_year": 歌曲首发年份或 null，歌手卡可为 null,
  "details": {{}},
  "confidence": 0.0到1.0,
  "sources": ["至少1个用于支撑摘要的网页URL"]
}}

规则:
- 只输出 JSON，不要 Markdown。
- 必须使用联网搜索结果；如果搜索不到可靠来源，summary 为空字符串，release_year 为 null，confidence <= 0.3。
- sources 必须填写真实网页 URL，不要填写搜索聚合页、空字符串或无法追溯的来源。
- 如果同名歌曲/歌手有歧义，降低 confidence，并在 facts 里说明歧义。
- 发行年份优先原曲首发年份，不把重制版、精选集、Live 专辑年份当作首发年份。
- artist details 建议字段: aliases, artist_type, country_or_region, active_years, members, genres, styles, languages, representative_works, achievements, similar_artists, sound_traits, lyrical_themes。
- song details 建议字段: album, original_release_year, release_type, version_note, writers, composers, lyricists, producers, genres, styles, moods, themes, scenarios, language, era, region, instrumentation, vocal_style, energy_descriptor, tempo_descriptor, lyrical_theme, known_context。
- 冷门歌曲资料不足时，不要硬填；可把 details 留空或只填能被来源支持的字段。
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


def _llm_web_song_cards_batch(songs: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Ask Qwen to enrich several songs in one web-search request.

    This is for offline backlog reduction only.  Each returned card still needs
    its own traceable source URL; cards without source evidence are dropped.
    """

    client = _dashscope_openai_client()
    if client is None or not hasattr(client, "responses") or not songs:
        return []
    seed_lines = []
    for idx, song in enumerate(songs, start=1):
        seed_lines.append(
            f"{idx}. title={str(song.get('title') or '').strip()} | artist={str(song.get('artist') or '').strip()}"
        )
    prompt = f"""
你是音乐资料整理助手。你必须实际调用联网搜索工具核对资料，只整理搜索结果能支持的音乐事实，不要凭模型记忆编造。

请为下面每一首歌曲分别整理知识卡。每首歌都必须独立保留 source_url；找不到可靠来源的歌曲可以省略，不要硬填。
歌曲列表:
{chr(10).join(seed_lines)}

请输出严格 JSON:
{{
  "cards": [
    {{
      "title": "歌曲名，保持输入中的对应歌曲名",
      "artist": "歌手，保持输入中的对应歌手",
      "summary": "120字以内中文摘要，覆盖风格、背景、代表性信息",
      "facts": ["最多5条可由联网结果支持的事实"],
      "style_tags": ["最多6个音乐风格/类型/场景标签"],
      "release_year": 歌曲首发年份或 null,
      "details": {{}},
      "confidence": 0.0到1.0,
      "sources": ["至少1个用于支撑该歌曲摘要的网页URL"]
    }}
  ]
}}

规则:
- 只输出 JSON，不要 Markdown。
- 必须使用联网搜索结果；没有可靠来源的歌曲不要放进 cards。
- sources 必须是真实网页 URL，不要填写搜索聚合页、空字符串或无法追溯的来源。
- 同名歌曲/版本有歧义时降低 confidence，并在 facts/details.version_note 说明。
- 发行年份优先原曲首发年份，不把重制版、精选集、Live 专辑年份当作首发年份。
- details 建议字段: album, original_release_year, release_type, version_note, writers, composers, lyricists, producers, genres, styles, moods, themes, scenarios, language, era, region, instrumentation, vocal_style, energy_descriptor, tempo_descriptor, lyrical_theme, known_context。
""".strip()
    try:
        response = client.responses.create(
            model=_knowledge_llm_model(),
            input=prompt,
            tools=[{"type": "web_search"}],
            extra_body={"enable_thinking": True},
        )
        parsed = _extract_json_object(_response_text(response))
        raw_cards = parsed.get("cards") if isinstance(parsed, Mapping) else None
        if not isinstance(raw_cards, list):
            return []
        seeds_by_key = {
            (str(song.get("title") or "").casefold(), str(song.get("artist") or "").casefold()): song
            for song in songs
        }
        cards: list[dict[str, Any]] = []
        for raw in raw_cards:
            if not isinstance(raw, Mapping):
                continue
            title = str(raw.get("title") or "").strip()
            artist = str(raw.get("artist") or "").strip()
            seed = seeds_by_key.get((title.casefold(), artist.casefold()))
            if seed is None and len(songs) == 1:
                seed = songs[0]
            normalized = _normalise_llm_card(
                kind="song",
                title=str((seed or raw).get("title") or title),
                artist=str((seed or raw).get("artist") or artist),
                parsed=raw,
                source="dashscope_web_search_batch",
            )
            if normalized:
                cards.append(normalized)
        return cards
    except Exception:
        return []


def _llm_web_artist_cards_batch(artists: list[str]) -> list[dict[str, Any]]:
    """Ask Qwen to enrich several artists in one web-search request."""

    client = _dashscope_openai_client()
    cleaned_artists = [str(artist or "").strip() for artist in artists if str(artist or "").strip()]
    if client is None or not hasattr(client, "responses") or not cleaned_artists:
        return []
    seed_lines = [f"{idx}. artist={artist}" for idx, artist in enumerate(cleaned_artists, start=1)]
    prompt = f"""
你是音乐资料整理助手。你必须实际调用联网搜索工具核对资料，只整理搜索结果能支持的音乐事实，不要凭模型记忆编造。

请为下面每一位歌手/乐队分别整理知识卡。每位艺人都必须独立保留 source_url；找不到可靠来源的艺人可以省略，不要硬填。
艺人列表:
{chr(10).join(seed_lines)}

请输出严格 JSON:
{{
  "cards": [
    {{
      "artist": "艺人名，保持输入中的对应艺人名",
      "summary": "120字以内中文摘要，覆盖风格、背景、代表性信息",
      "facts": ["最多5条可由联网结果支持的事实"],
      "style_tags": ["最多6个音乐风格/类型/场景标签"],
      "release_year": null,
      "details": {{}},
      "confidence": 0.0到1.0,
      "sources": ["至少1个用于支撑该艺人摘要的网页URL"]
    }}
  ]
}}

规则:
- 只输出 JSON，不要 Markdown。
- 必须使用联网搜索结果；没有可靠来源的艺人不要放进 cards。
- sources 必须是真实网页 URL，不要填写搜索聚合页、空字符串或无法追溯的来源。
- 同名艺人/乐队有歧义时降低 confidence，并在 facts/details.version_note 说明。
- details 建议字段: aliases, artist_type, country_or_region, active_years, members, genres, styles, languages, representative_works, achievements, similar_artists, influences, sound_traits, lyrical_themes。
- 冷门艺人资料不足时，不要硬填；可把 details 留空或只填能被来源支持的字段。
""".strip()
    try:
        response = client.responses.create(
            model=_knowledge_llm_model(),
            input=prompt,
            tools=[{"type": "web_search"}],
            extra_body={"enable_thinking": True},
        )
        parsed = _extract_json_object(_response_text(response))
        raw_cards = parsed.get("cards") if isinstance(parsed, Mapping) else None
        if not isinstance(raw_cards, list):
            return []
        seeds = {artist.casefold(): artist for artist in cleaned_artists}
        cards: list[dict[str, Any]] = []
        for raw in raw_cards:
            if not isinstance(raw, Mapping):
                continue
            artist = str(raw.get("artist") or raw.get("title") or "").strip()
            seed_artist = seeds.get(artist.casefold()) or artist
            normalized = _normalise_llm_card(
                kind="artist",
                title=seed_artist,
                artist=seed_artist,
                parsed=raw,
                source="dashscope_web_search_batch",
            )
            if normalized:
                cards.append(normalized)
        return cards
    except Exception:
        return []


async def _llm_web_card_async(**kwargs: Any) -> dict[str, Any] | None:
    timeout = float(os.getenv("MUSIC_KNOWLEDGE_LLM_TIMEOUT_SECONDS", "120")) + 10.0
    try:
        return await asyncio.wait_for(asyncio.to_thread(_llm_web_card, **kwargs), timeout=timeout)
    except Exception:
        return None


async def _llm_web_song_cards_batch_async(songs: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    timeout = float(os.getenv("MUSIC_KNOWLEDGE_LLM_TIMEOUT_SECONDS", "120")) + 30.0
    try:
        return await asyncio.wait_for(asyncio.to_thread(_llm_web_song_cards_batch, songs), timeout=timeout)
    except Exception:
        return []


async def _llm_web_artist_cards_batch_async(artists: list[str]) -> list[dict[str, Any]]:
    timeout = float(os.getenv("MUSIC_KNOWLEDGE_LLM_TIMEOUT_SECONDS", "120")) + 30.0
    try:
        return await asyncio.wait_for(asyncio.to_thread(_llm_web_artist_cards_batch, artists), timeout=timeout)
    except Exception:
        return []


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
    allow_snippet_fallback: bool = False,
) -> dict[str, Any] | None:
    query = build_artist_knowledge_query(artist)
    card = None
    if use_llm_summary:
        card = await _llm_web_card_async(kind="artist", query=query, artist=artist, title=artist)
    if card is None and allow_snippet_fallback:
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
            details=card.get("details") or {},
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
    allow_snippet_fallback: bool = False,
) -> dict[str, Any] | None:
    query = build_song_knowledge_query(title, artist)
    card = None
    if use_llm_summary:
        card = await _llm_web_card_async(kind="song", query=query, title=title, artist=artist)
    if card is None and allow_snippet_fallback:
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
            details=card.get("details") or {},
            source_url=card.get("source_url", ""),
            source_title=card.get("source_title", ""),
            source_provider=card.get("source", "web"),
            confidence=card.get("confidence", 0.6),
        )
    return card


async def enrich_song_cards_batch(
    songs: list[Mapping[str, Any]],
    *,
    store: MusicKnowledgeStore | None = None,
    dry_run: bool = False,
    use_llm_summary: bool = True,
) -> list[dict[str, Any]]:
    """Batch-enrich songs with one Qwen web-search request per chunk."""

    if not songs or not use_llm_summary:
        return []
    cards = await _llm_web_song_cards_batch_async(songs)
    if cards and not dry_run:
        store = store or MusicKnowledgeStore()
        for card in cards:
            store.upsert_song_card(
                title=card.get("title", ""),
                artist=card.get("artist", ""),
                summary=card["summary"],
                release_year=card.get("release_year"),
                style_tags=card.get("style_tags", []),
                facts=card.get("facts", []),
                details=card.get("details") or {},
                source_url=card.get("source_url", ""),
                source_title=card.get("source_title", ""),
                source_provider=card.get("source", "dashscope_web_search_batch"),
                confidence=card.get("confidence", 0.6),
            )
    return cards


async def enrich_artist_cards_batch(
    artists: list[str],
    *,
    store: MusicKnowledgeStore | None = None,
    dry_run: bool = False,
    use_llm_summary: bool = True,
) -> list[dict[str, Any]]:
    """Batch-enrich artists with one Qwen web-search request per chunk."""

    if not artists or not use_llm_summary:
        return []
    cards = await _llm_web_artist_cards_batch_async(artists)
    if cards and not dry_run:
        store = store or MusicKnowledgeStore()
        for card in cards:
            artist = card.get("artist") or card.get("title") or ""
            store.upsert_artist_card(
                artist=artist,
                summary=card["summary"],
                style_tags=card.get("style_tags", []),
                facts=card.get("facts", []),
                details=card.get("details") or {},
                source_url=card.get("source_url", ""),
                source_title=card.get("source_title", ""),
                source_provider=card.get("source", "dashscope_web_search_batch"),
                confidence=card.get("confidence", 0.6),
            )
    return cards
