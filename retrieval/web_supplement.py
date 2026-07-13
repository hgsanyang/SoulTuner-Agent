"""Parallel web-discovery lane built on the planner model's native web search.

The local recommendation chain stays untouched; this lane runs concurrently
and, when it finishes in time, contributes a few evidence-backed online
songs. The LLM (qwen with DashScope ``enable_search``) decides what evidence
to look for — charts, community consensus, high-quality playlists — and must
name its evidence per song. Deterministic code only handles playability
resolution, candidate/hit consistency checks, dedup and bounded scoring;
it never picks songs by keyword rules.

Toggles: requires the per-request web switch (MUSIC_WEB_SEARCH_ENABLED) and
the lane switch MUSIC_WEB_SUPPLEMENT_ENABLED (default on). Fully local
deployments simply keep the web switch off.
"""

from __future__ import annotations

import asyncio
import difflib
import hashlib
import json
import logging
import os
import re
from typing import Any, Awaitable, Callable

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

MAX_CANDIDATES = 6
MAX_RESOLVED = 4
# Mid-range bounded score: web supplements enrich the slate but never outrank
# strong local relevance by fiat (the old path hard-coded 9.5).
SUPPLEMENT_BASE_SCORE = 6.5
DUPLICATE_SIMILARITY = 0.85

WEB_SUPPLEMENT_SYSTEM_PROMPT = """你是一位选曲人，可以联网搜索。用户刚发出一个听歌请求，
本地曲库已经在并行出一批结果；你的任务是从互联网上补充少量真正合适的歌。

要求：
1. 用联网搜索找证据，优先级：权威榜单/音乐平台数据 > 乐评与社区高赞共识
   （豆瓣/Reddit/B站评论区等）> 高质量主题歌单。不要凭记忆报歌。
2. 每首歌必须给出 evidence：一句话说明来自哪类来源、为什么适合当前请求。
   没有可信证据的歌不要输出。
3. 数量宁缺毋滥，最多 {max_candidates} 首；找不到合适的就返回空列表。
4. 不要输出用户明确要避开的方向；歌名和歌手用原文（中文歌用中文名）。

输出 JSON：{{"songs":[{{"title":"...","artist":"...","evidence":"...","source_kind":"chart|community|playlist"}}]}}
"""


class WebSongCandidate(BaseModel):
    title: str = Field(min_length=1, max_length=120)
    artist: str = Field(min_length=1, max_length=80)
    evidence: str = Field(default="", max_length=200)
    source_kind: str = Field(default="", max_length=20)


class WebSongDiscovery(BaseModel):
    songs: list[WebSongCandidate] = Field(default_factory=list)


_PAREN_RE = re.compile(r"[（(\[【][^）)\]】]*[）)\]】]")
_NOISE_RE = re.compile(
    r"\b(feat\.?|ft\.?|live|remaster(?:ed)?(?:\s*\d{4})?|acoustic|demo|cover|"
    r"remix|version|ver\.?|edit|explicit|instrumental)\b",
    flags=re.IGNORECASE,
)
_PUNCT_RE = re.compile(r"[\s\W_]+", flags=re.UNICODE)


def normalize_song_text(text: Any) -> str:
    """Normalize a title/artist for duplicate comparison."""
    value = _PAREN_RE.sub(" ", str(text or ""))
    value = _NOISE_RE.sub(" ", value)
    value = _PUNCT_RE.sub("", value.casefold())
    return value


def is_similar_text(left: Any, right: Any, *, threshold: float = DUPLICATE_SIMILARITY) -> bool:
    a, b = normalize_song_text(left), normalize_song_text(right)
    if not a or not b:
        return False
    if a == b or a in b or b in a:
        return True
    return difflib.SequenceMatcher(None, a, b).ratio() >= threshold


def is_duplicate_song(
    title: Any,
    artist: Any,
    existing: list[tuple[Any, Any]],
) -> bool:
    """Fuzzy duplicate check on (title, artist) against existing songs."""
    for existing_title, existing_artist in existing:
        if is_similar_text(title, existing_title) and is_similar_text(artist, existing_artist):
            return True
        # Same normalized title with no artist info still counts as duplicate.
        if (
            normalize_song_text(title)
            and normalize_song_text(title) == normalize_song_text(existing_title)
            and not normalize_song_text(existing_artist)
        ):
            return True
    return False


def supplement_enabled() -> bool:
    return (
        os.environ.get("MUSIC_WEB_SEARCH_ENABLED", "1") != "0"
        and os.environ.get("MUSIC_WEB_SUPPLEMENT_ENABLED", "1") != "0"
    )


async def _default_resolver(candidate: WebSongCandidate) -> dict[str, Any] | None:
    """Resolve a candidate to a playable hit whose identity actually matches."""
    from tools.music_fetch_tool import execute_search_online_music

    fetched = await execute_search_online_music(f"{candidate.artist} {candidate.title}")
    if not getattr(fetched, "success", False) or not fetched.data:
        return None
    for hit in fetched.data[:5]:
        # First hits on open platforms are often karaoke/white-noise uploads
        # with unrelated names; only accept hits that match the candidate.
        if is_similar_text(hit.get("title"), candidate.title) and (
            is_similar_text(hit.get("artist"), candidate.artist)
            or not str(hit.get("artist") or "").strip()
        ):
            return hit
    return None


class WebSongSupplement:
    """LLM-with-web-search discovery, resolved to playable, deduped songs.

    ``generator`` (payload -> WebSongDiscovery) and ``resolver``
    (candidate -> playable hit dict | None) are injectable for tests.
    """

    def __init__(
        self,
        *,
        generator: Callable[[dict[str, Any]], Any] | None = None,
        resolver: Callable[[WebSongCandidate], Awaitable[dict[str, Any] | None]] | None = None,
        timeout_seconds: float = 30.0,
    ):
        self.generator = generator
        self.resolver = resolver or _default_resolver
        self.timeout_seconds = float(timeout_seconds)

    @staticmethod
    def prompt_hash() -> str:
        return hashlib.sha256(WEB_SUPPLEMENT_SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:16]

    async def discover(
        self,
        *,
        query: str,
        plan_summary: dict[str, Any] | None = None,
        avoid: list[str] | None = None,
        limit: int = MAX_RESOLVED,
    ) -> list[dict[str, Any]]:
        """Return playable supplement items shaped like existing web_playable."""
        try:
            return await asyncio.wait_for(
                self._discover(query=query, plan_summary=plan_summary, avoid=avoid, limit=limit),
                timeout=self.timeout_seconds,
            )
        except Exception as exc:
            logger.warning("[WebSupplement] 联网补充失败（fail-soft 跳过）: %s", exc)
            return []

    async def _discover(
        self,
        *,
        query: str,
        plan_summary: dict[str, Any] | None,
        avoid: list[str] | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        payload = {
            "user_request": str(query or "")[:400],
            "structured_plan": plan_summary or {},
            "explicitly_avoided": [str(item)[:60] for item in (avoid or [])][:10],
        }
        discovery = await self._invoke(payload)
        candidates = discovery.songs[:MAX_CANDIDATES]

        seen: list[tuple[str, str]] = []
        unique: list[WebSongCandidate] = []
        for candidate in candidates:
            if is_duplicate_song(candidate.title, candidate.artist, seen):
                continue
            seen.append((candidate.title, candidate.artist))
            unique.append(candidate)

        resolved: list[dict[str, Any]] = []
        hits = await asyncio.gather(
            *(self.resolver(candidate) for candidate in unique),
            return_exceptions=True,
        )
        for candidate, hit in zip(unique, hits):
            if isinstance(hit, Exception) or not hit:
                continue
            rank = len(resolved)
            reason = f"🌐 联网补充：{candidate.evidence}" if candidate.evidence else "🌐 联网补充"
            resolved.append(
                {
                    "song": {
                        "title": hit.get("title") or candidate.title,
                        "artist": hit.get("artist") or candidate.artist,
                        "preview_url": hit.get("play_url") or hit.get("preview_url"),
                        "cover_url": hit.get("cover_url"),
                        "album": hit.get("album", "未知"),
                        "genre": "Web Discovery",
                        "source": "web_supplement",
                        "recall_sources": ["web"],
                        "recall_source_labels": ["联网"],
                        "web_evidence": candidate.evidence,
                        "web_source_kind": candidate.source_kind,
                    },
                    "reason": reason,
                    "similarity_score": SUPPLEMENT_BASE_SCORE - rank * 0.1,
                    "_recall_sources": ["web"],
                    "_recall_source_labels": ["联网"],
                }
            )
            if len(resolved) >= max(1, limit):
                break
        return resolved

    async def _invoke(self, payload: dict[str, Any]) -> WebSongDiscovery:
        if self.generator is not None:
            result = self.generator(payload)
            if asyncio.iscoroutine(result) or isinstance(result, Awaitable):
                result = await result
            if isinstance(result, WebSongDiscovery):
                return result
            return WebSongDiscovery.model_validate(result)

        from config.settings import settings
        from llms.chat_models import get_chat_model

        provider = settings.intent_llm_provider or settings.llm_default_provider
        model_name = settings.intent_llm_model or settings.llm_default_model
        llm = get_chat_model(
            provider=provider,
            model_name=model_name,
            temperature=0.0,
            max_tokens=1200,
            enable_web_search=True,
        )
        try:
            structured = llm.with_structured_output(
                WebSongDiscovery, include_raw=True, method="json_mode"
            )
        except (TypeError, ValueError):
            structured = llm.with_structured_output(WebSongDiscovery, include_raw=True)
        messages = [
            ("system", WEB_SUPPLEMENT_SYSTEM_PROMPT.format(max_candidates=MAX_CANDIDATES)),
            ("human", json.dumps(payload, ensure_ascii=False, separators=(",", ":"))),
        ]
        result = await structured.ainvoke(messages)
        if isinstance(result, WebSongDiscovery):
            return result
        if isinstance(result, dict) and isinstance(result.get("parsed"), WebSongDiscovery):
            return result["parsed"]
        raw = result.get("raw") if isinstance(result, dict) else result
        content = str(getattr(raw, "content", raw) or "").strip()
        start, end = content.find("{"), content.rfind("}")
        if start >= 0 and end > start:
            content = content[start : end + 1]
        return WebSongDiscovery.model_validate(json.loads(content))


_default_supplement: WebSongSupplement | None = None


def get_web_supplement() -> WebSongSupplement:
    global _default_supplement
    if _default_supplement is None:
        _default_supplement = WebSongSupplement()
    return _default_supplement
