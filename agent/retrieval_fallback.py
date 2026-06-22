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


def layered_constraints(plan: Mapping[str, Any] | None) -> tuple[list[str], list[str], dict[str, Any]]:
    plan = dict(plan or {})
    hard = dict(plan.get("hard_constraints") or {})
    artists = list(hard.get("artist_entities") or plan.get("graph_artist_entities") or [])
    songs = list(hard.get("song_entities") or plan.get("graph_song_entities") or [])
    return artists, songs, hard


def _song_dict(item: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = item.get("song")
    return nested if isinstance(nested, Mapping) else item


def decide_online_fallback(
    search_results: Sequence[Mapping[str, Any]],
    retrieval_plan: Mapping[str, Any] | None,
) -> FallbackDecision:
    """Decide fallback from final local inventory, independent of intent label."""
    artists, songs, hard = layered_constraints(retrieval_plan)
    count = len(search_results)
    has_explicit_constraint = bool(
        artists
        or songs
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

    if songs:
        normalized_titles = [_normalize(_song_dict(item).get("title")) for item in search_results]
        title_matched = any(
            normalized_term
            and any(normalized_term in title for title in normalized_titles if title)
            for term in songs
            if (normalized_term := _normalize(term))
        )
        if not title_matched:
            return FallbackDecision(True, "local_song_match_missing", count)

    return FallbackDecision(False, "", count)


def avoid_terms(retrieval_plan: Mapping[str, Any] | None) -> list[str]:
    soft = dict((retrieval_plan or {}).get("soft_intent") or {})
    return [str(term).strip() for term in soft.get("avoid") or [] if str(term).strip()]


def filter_results_by_avoid(
    results: Sequence[Mapping[str, Any]],
    terms: Iterable[str],
) -> tuple[list[Mapping[str, Any]], int]:
    """Remove literal title/artist matches; subjective avoids remain soft signals."""
    normalized_terms = [_normalize(term) for term in terms]
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


def fallback_query(retrieval_plan: Mapping[str, Any] | None, default_query: str) -> str:
    artists, songs, _hard = layered_constraints(retrieval_plan)
    avoids = avoid_terms(retrieval_plan)
    if artists and songs:
        return f"{artists[0]} {songs[0]}".strip()
    if artists:
        return artists[0]
    if songs:
        suffix = " 翻唱" if avoids else ""
        return f"{songs[0]}{suffix}".strip()
    return default_query
