"""Pure helpers for inventory-aware online fallback and exclusion filtering."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable, Mapping, Sequence

from agent.netease_query import artist_matches


@dataclass(frozen=True)
class FallbackDecision:
    required: bool
    reason: str = ""
    inventory_count: int = 0


def _normalize(value: Any) -> str:
    return re.sub(r"[\W_]+", "", str(value or "").casefold())


_LANGUAGE_ALIASES = {
    "chinese": {"chinese", "mandarin", "中文", "华语", "国语"},
    "cantonese": {"cantonese", "粤语", "广东话"},
    "english": {"english", "英文", "英语"},
    "japanese": {"japanese", "日语", "日文", "jpop"},
    "korean": {"korean", "韩语", "韩文", "kpop"},
    "instrumental": {"instrumental", "纯音乐", "器乐", "无人声"},
}


def _canonical_language(value: Any) -> str:
    normalized = _normalize(value)
    for canonical, aliases in _LANGUAGE_ALIASES.items():
        if normalized in {_normalize(alias) for alias in aliases}:
            return canonical
    return normalized


def _language_matches(value: Any, expected: Any) -> bool:
    actual = _canonical_language(value)
    wanted = _canonical_language(expected)
    return bool(actual and wanted and actual == wanted)


def _literal_avoid_candidates(term: Any) -> list[str]:
    """Return conservative literal cores from conversational avoid phrases."""
    text = str(term or "").strip()
    if not text:
        return []
    candidates = [text]
    core = re.sub(r"^(?:但)?(?:不要|不是|别要|别给我|排除|避开)", "", text).strip()
    core = re.sub(r"(?:那一首|那首|那一版|这个版本|的歌|的歌曲|原唱)$", "", core).strip()
    if len(_normalize(core)) >= 2 and core != text:
        candidates.append(core)
    return candidates


def layered_constraints(plan: Mapping[str, Any] | None) -> tuple[list[str], list[str], dict[str, Any]]:
    plan = dict(plan or {})
    hard = dict(plan.get("hard_constraints") or {})
    artists = list(hard.get("artist_entities") or plan.get("graph_artist_entities") or [])
    songs = list(hard.get("song_entities") or plan.get("graph_song_entities") or [])
    return artists, songs, hard


def _song_dict(item: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = item.get("song")
    return nested if isinstance(nested, Mapping) else item


def _song_entities_are_similarity_seeds(
    retrieval_plan: Mapping[str, Any] | None,
    query: str = "",
) -> bool:
    plan = dict(retrieval_plan or {})
    soft = dict(plan.get("soft_intent") or {})
    context = " ".join(
        [
            query,
            str(soft.get("goal") or ""),
            str(soft.get("vibe") or ""),
            str(plan.get("vector_acoustic_query") or ""),
            str(plan.get("web_search_keywords") or ""),
        ]
    ).casefold()
    return any(
        term in context
        for term in (
            "类似",
            "相似",
            "听感",
            "像",
            "同类",
            "similar",
            "same vibe",
            "sounds like",
            "like this",
        )
    )


def decide_online_fallback(
    search_results: Sequence[Mapping[str, Any]],
    retrieval_plan: Mapping[str, Any] | None,
    query: str = "",
) -> FallbackDecision:
    """Decide fallback from final local inventory, independent of intent label."""
    artists, songs, hard = layered_constraints(retrieval_plan)
    song_entities_are_reference = bool(songs) and _song_entities_are_similarity_seeds(
        retrieval_plan,
        query,
    )
    count = len(search_results)
    has_explicit_constraint = bool(
        artists
        or (songs and not song_entities_are_reference)
        or hard.get("language")
        or hard.get("region")
        or hard.get("instrumental")
    )
    if not search_results:
        return FallbackDecision(
            required=has_explicit_constraint,
            reason="local_inventory_empty" if has_explicit_constraint else "",
            inventory_count=0,
        )

    if artists and not songs:
        checked = []
        for item in search_results:
            artist = str(_song_dict(item).get("artist") or "").strip()
            if artist and artist != "互联网最新情报":
                checked.append(artist)
        matched = sum(artist_matches(artist, tuple(artists)) for artist in checked)
        if not checked or matched / len(checked) < 0.7:
            return FallbackDecision(True, "local_artist_match_insufficient", count)

    if songs and not song_entities_are_reference:
        normalized_titles = [_normalize(_song_dict(item).get("title")) for item in search_results]
        title_matched = any(
            normalized_term
            and any(normalized_term in title for title in normalized_titles if title)
            for term in songs
            if (normalized_term := _normalize(term))
        )
        if not title_matched:
            return FallbackDecision(True, "local_song_match_missing", count)

    language = hard.get("language")
    if language and _canonical_language(language) != "instrumental":
        known_languages = []
        matched_languages = 0
        for item in search_results:
            value = _song_dict(item).get("language")
            normalized = _normalize(value)
            if not normalized or normalized in {"unknown", "none", "null", "未知", "未标注"}:
                continue
            known_languages.append(value)
            if _language_matches(value, language):
                matched_languages += 1
        match_ratio = matched_languages / max(len(known_languages), 1)
        if matched_languages < 3 or match_ratio < 0.5:
            return FallbackDecision(True, "local_language_match_insufficient", count)

    return FallbackDecision(False, "", count)


def _query_avoid_terms(query: str = "") -> list[str]:
    """Extract conservative literal avoid terms from the raw user query."""
    text = str(query or "").strip()
    if not text:
        return []

    terms: list[str] = []
    for match in re.finditer(
        r"(?:不是|不要|别(?:是|给我)?|排除|避开)([^，。,；;！!？?\n]{2,24})",
        text,
    ):
        term = match.group(1).strip()
        term = re.sub(r"^(?:太|很|特别)", "", term).strip()
        term = re.sub(r"(?:那一首|那首|那一版|这个版本|的歌|的歌曲|原唱)$", "", term).strip()
        if len(_normalize(term)) >= 2:
            terms.append(term)
    return terms


def avoid_terms(retrieval_plan: Mapping[str, Any] | None, query: str = "") -> list[str]:
    soft = dict((retrieval_plan or {}).get("soft_intent") or {})
    terms = [str(term).strip() for term in soft.get("avoid") or [] if str(term).strip()]
    terms.extend(_query_avoid_terms(query))
    return list(dict.fromkeys(terms))


def filter_results_by_avoid(
    results: Sequence[Mapping[str, Any]],
    terms: Iterable[str],
) -> tuple[list[Mapping[str, Any]], int]:
    """Remove literal title/artist matches; subjective avoids remain soft signals."""
    normalized_terms = [
        _normalize(candidate)
        for term in terms
        for candidate in _literal_avoid_candidates(term)
    ]
    normalized_terms = [term for term in normalized_terms if len(term) >= 2]
    if not normalized_terms:
        return list(results), 0

    kept = []
    excluded = 0
    for item in results:
        song = _song_dict(item)
        title = _normalize(song.get("title") or song.get("name"))
        artist = _normalize(song.get("artist"))
        if not artist:
            artists = song.get("artists") or []
            artist = _normalize(" ".join(str(a.get("name", "")) for a in artists if isinstance(a, Mapping)))
        if any(term in title or term in artist for term in normalized_terms):
            excluded += 1
            continue
        kept.append(item)
    return kept, excluded


def filter_results_by_requested_language(
    results: Sequence[Mapping[str, Any]],
    language: Any,
) -> tuple[list[Mapping[str, Any]], int]:
    """Keep only web rows whose script gives high-confidence language evidence.

    Netease search rows do not expose a language field.  We currently apply a
    strict script check only for Korean, where Hangul is unambiguous.  Other
    languages stay untouched instead of receiving fabricated labels.
    """
    canonical = _canonical_language(language)
    if canonical != "korean":
        return list(results), 0

    kept = []
    excluded = 0
    for item in results:
        song = _song_dict(item)
        title = str(song.get("title") or song.get("name") or "")
        artist = str(song.get("artist") or "")
        if not artist:
            artists = song.get("artists") or []
            artist = " ".join(
                str(value.get("name", ""))
                for value in artists
                if isinstance(value, Mapping)
            )
        if re.search(r"[\uac00-\ud7af]", f"{title} {artist}"):
            enriched = dict(item)
            enriched["_inferred_language"] = "Korean"
            kept.append(enriched)
        else:
            excluded += 1
    return kept, excluded


def fallback_query(retrieval_plan: Mapping[str, Any] | None, default_query: str) -> str:
    artists, songs, hard = layered_constraints(retrieval_plan)
    avoids = avoid_terms(retrieval_plan)
    if artists and songs:
        return f"{artists[0]} {songs[0]}".strip()
    if artists:
        return artists[0]
    if songs:
        suffix = " 翻唱" if avoids else ""
        return f"{songs[0]}{suffix}".strip()
    language = _canonical_language(hard.get("language"))
    if language == "korean":
        return "韩国 抒情歌"
    if language == "cantonese":
        return "粤语 经典歌曲"
    return default_query
