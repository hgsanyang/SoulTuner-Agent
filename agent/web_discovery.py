"""External music discovery helpers.

Search engines discover candidate titles/artists; Netease resolves those
candidates into playable rows.  These helpers stay deterministic so they can be
unit tested without network access.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping


@dataclass(frozen=True)
class SongCandidate:
    title: str
    artist: str = ""
    evidence: str = ""

    @property
    def query(self) -> str:
        return " ".join(part for part in (self.title, self.artist) if part).strip()


def build_web_discovery_query(
    user_query: str,
    retrieval_plan: Mapping[str, Any] | None = None,
    gap: Mapping[str, Any] | None = None,
) -> str:
    plan = dict(retrieval_plan or {})
    hard = dict(plan.get("hard_constraints") or {})
    soft = dict(plan.get("soft_intent") or {})
    hints = dict(plan.get("hints") or {})
    parts = [
        user_query,
        str(plan.get("web_search_keywords") or ""),
        str(soft.get("goal") or ""),
        str(soft.get("vibe") or ""),
        "歌名 歌手 发行年份 推荐",
    ]
    language = hard.get("language")
    if language:
        parts.append(f"{language} songs")
    genres = hints.get("genres")
    if isinstance(genres, list):
        parts.extend(str(item) for item in genres[:3])
    elif genres:
        parts.append(str(genres))
    reasons = ", ".join((gap or {}).get("reasons") or [])
    if reasons:
        parts.append(reasons)
    query = " ".join(part.strip() for part in parts if str(part).strip())
    return re.sub(r"\s+", " ", query).strip()


def _clean(value: str) -> str:
    value = re.sub(r"\[[^\]]+\]|\([^\)]*(?:official|mv|lyrics|live|remaster|cover)[^\)]*\)", "", value, flags=re.I)
    value = re.sub(r"^[\d\-\*\s.、:：]+", "", value)
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"[的\s]+$", "", value)
    return value.strip(" -—–|｜:：，,。.;；")


def _valid_title(value: str) -> bool:
    value = _clean(value)
    if not (1 <= len(value) <= 80):
        return False
    banned = ("摘要", "搜索结果", "spotify", "youtube", "网易云", "推荐", "playlist")
    return not any(term in value.casefold() for term in banned)


def extract_song_candidates(web_text: str, *, max_candidates: int = 12) -> list[SongCandidate]:
    """Extract likely (title, artist) pairs from search snippets.

    The extractor is intentionally conservative: it prefers quoted Chinese
    titles and simple "Title - Artist" / "Artist - Title" lines, then leaves
    final validation to Netease resolver and catalog filters.
    """
    text = str(web_text or "")
    if not text or "网络搜索未能找到" in text:
        return []
    candidates: list[SongCandidate] = []
    seen: set[str] = set()

    def add(title: str, artist: str = "", evidence: str = "") -> None:
        title = _clean(title)
        artist = _clean(artist)
        if not _valid_title(title):
            return
        if len(artist) > 60:
            artist = ""
        key = f"{title.casefold()}\0{artist.casefold()}"
        if key in seen:
            return
        seen.add(key)
        candidates.append(SongCandidate(title=title, artist=artist, evidence=evidence[:180]))

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in lines:
        if len(candidates) >= max_candidates:
            break
        compact = re.sub(r"\s+", " ", line)
        for match in re.finditer(r"(?P<artist>[\w\u4e00-\u9fff·・.&、\s-]{1,50})[的\s]*《(?P<title>[^》]{1,80})》", compact):
            add(match.group("title"), match.group("artist"), compact)
            if len(candidates) >= max_candidates:
                break
        if len(candidates) >= max_candidates:
            break
        for match in re.finditer(r"《(?P<title>[^》]{1,80})》\s*(?:-|—|–|/|by|演唱|歌手|来自)?\s*(?P<artist>[\w\u4e00-\u9fff·・.&、\s-]{0,50})", compact, flags=re.I):
            add(match.group("title"), match.group("artist"), compact)
            if len(candidates) >= max_candidates:
                break
        if len(candidates) >= max_candidates:
            break
        if any(cue in compact.casefold() for cue in ("song", "歌曲", "歌手", "artist", "track")):
            match = re.search(
                r"(?P<title>[\w\u4e00-\u9fff·・.&'’\s]{2,60})\s[-—–]\s(?P<artist>[\w\u4e00-\u9fff·・.&、\s]{2,50})",
                compact,
            )
            if match:
                add(match.group("title"), match.group("artist"), compact)

    return candidates[:max_candidates]
