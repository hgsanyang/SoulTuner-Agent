"""Independent local recall sources used by the R1 retrieval pipeline."""

from __future__ import annotations

import json
from typing import Any, Iterable, List, Mapping
from urllib.parse import quote

from config.settings import settings
from retrieval.neo4j_client import get_neo4j_client
from retrieval.retrieval_fusion import normalize_text


def _clean_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item) for item in value if item]
    if value and value != "Unknown":
        return [str(value)]
    return []


def _public_url(path: str | None) -> str | None:
    if not path:
        return None
    encoded = "/".join(quote(part, safe="") for part in str(path).split("/"))
    return f"{settings.api_base_url}{encoded}"


def _playable_song_where(alias: str = "s") -> str:
    return (
        f"coalesce(properties({alias})['unplayable_stub'], false) <> true "
        f"AND {alias}.audio_url IS NOT NULL "
        f"AND trim(toString({alias}.audio_url)) <> ''"
    )


def _record_to_song(record: Mapping[str, Any]) -> dict:
    genres = _clean_list(record.get("genres"))
    moods = _clean_list(record.get("moods"))
    themes = _clean_list(record.get("themes"))
    scenarios = _clean_list(record.get("scenarios"))
    display_parts = genres[:2] + moods[:1] + scenarios[:1]
    return {
        "title": record.get("title") or "未知标题",
        "artist": record.get("artist")
        or "、".join(_clean_list(record.get("artists")))
        or "未知艺术家",
        "album": record.get("album") or "未知",
        "genre": "/".join(display_parts) if display_parts else "",
        "genres": genres,
        "moods": moods,
        "themes": themes,
        "scenarios": scenarios,
        "language": record.get("language") or "Unknown",
        "region": record.get("region") or "Unknown",
        "is_instrumental": bool(record.get("is_instrumental") or False),
        "has_vocal": record.get("has_vocal"),
        "has_drums": record.get("has_drums"),
        "energy_level": record.get("energy_level"),
        "acoustic_vocalness": record.get("acoustic_vocalness"),
        "acoustic_drumness": record.get("acoustic_drumness"),
        "acoustic_energy": record.get("acoustic_energy"),
        "acoustic_probe_version": record.get("acoustic_probe_version"),
        "preview_url": _public_url(record.get("audio_url")),
        "cover_url": _public_url(record.get("cover_url")),
        "lrc_url": _public_url(record.get("lrc_url")),
    }


def _records_to_json(records: Iterable[Mapping[str, Any]], score_field: str) -> str:
    results = []
    for record in records:
        item = _record_to_song(record)
        item["similarity_score"] = float(record.get(score_field) or 0.0)
        results.append(item)
    return json.dumps(results, ensure_ascii=False)


def _norm_terms(values: Iterable[Any]) -> List[str]:
    terms: List[str] = []
    for value in values:
        text = normalize_text(value)
        if text and text not in terms:
            terms.append(text)
    return terms


def graph_candidate_recall(
    hard_constraints: Mapping[str, Any],
    hints: Mapping[str, Any],
    *,
    limit: int,
) -> str:
    """Rank graph entities and optional tags without treating soft tags as filters."""
    artists = list(hard_constraints.get("artist_entities") or [])
    songs = list(hard_constraints.get("song_entities") or [])
    genres = list(hints.get("genres") or [])
    moods = [hints.get("mood")] if hints.get("mood") else []
    scenarios = [hints.get("scenario")] if hints.get("scenario") else []
    language = hard_constraints.get("language")
    region = hard_constraints.get("region")
    instrumental = bool(hard_constraints.get("instrumental"))
    if normalize_text(language) in {"instrumental", "纯音乐", "器乐"}:
        language = None
        instrumental = True
    if not any((artists, songs, genres, moods, scenarios, language, region, instrumental)):
        return "[]"

    client = get_neo4j_client()
    candidate_limit = max(limit * 4, 80)

    def _merge_eids(current: list[str], rows: list[Mapping[str, Any]]) -> list[str]:
        seen = set(current)
        for row in rows or []:
            eid = str(row.get("eid") or "")
            if eid and eid not in seen:
                current.append(eid)
                seen.add(eid)
        return current[:candidate_limit]

    candidate_eids: list[str] = []

    def _run_candidate_query(cypher: str, params: Mapping[str, Any]) -> None:
        nonlocal candidate_eids
        if len(candidate_eids) >= candidate_limit:
            return
        rows = client.execute_query(cypher, {**params, "candidate_limit": candidate_limit})
        candidate_eids = _merge_eids(candidate_eids, rows)

    playable = _playable_song_where("s")
    song_terms = _norm_terms(songs)
    artist_terms = _norm_terms(artists)
    genre_terms = _norm_terms(genres)
    mood_terms = _norm_terms(moods)
    theme_terms = _norm_terms(genres + moods + scenarios)
    scenario_terms = _norm_terms(scenarios)

    if song_terms:
        _run_candidate_query(
            f"""
            MATCH (s:Song)
            WHERE {playable}
              AND any(term IN $terms WHERE toLower(toString(coalesce(s.title, ''))) CONTAINS term)
            RETURN DISTINCT elementId(s) AS eid
            LIMIT $candidate_limit
            """,
            {"terms": song_terms},
        )
    if artist_terms:
        _run_candidate_query(
            f"""
            MATCH (s:Song)-[:PERFORMED_BY]->(a:Artist)
            WHERE {playable}
              AND any(term IN $terms WHERE toLower(toString(coalesce(a.name, ''))) CONTAINS term)
            RETURN DISTINCT elementId(s) AS eid
            LIMIT $candidate_limit
            """,
            {"terms": artist_terms},
        )
    for rel, label, terms in (
        ("BELONGS_TO_GENRE", "Genre", genre_terms),
        ("HAS_MOOD", "Mood", mood_terms),
        ("HAS_THEME", "Theme", theme_terms),
        ("FITS_SCENARIO", "Scenario", scenario_terms),
    ):
        if not terms:
            continue
        _run_candidate_query(
            f"""
            MATCH (s:Song)-[:{rel}]->(n:{label})
            WHERE {playable}
              AND any(term IN $terms WHERE toLower(toString(coalesce(n.name, ''))) CONTAINS term)
            RETURN DISTINCT elementId(s) AS eid
            LIMIT $candidate_limit
            """,
            {"terms": terms},
        )
    if language:
        _run_candidate_query(
            f"""
            MATCH (s:Song)
            WHERE {playable}
              AND toLower(toString(coalesce(s.language, ''))) = $language
            RETURN DISTINCT elementId(s) AS eid
            LIMIT $candidate_limit
            """,
            {"language": normalize_text(language)},
        )
    if region:
        _run_candidate_query(
            f"""
            MATCH (s:Song)
            WHERE {playable}
              AND toLower(toString(coalesce(s.region, ''))) = $region
            RETURN DISTINCT elementId(s) AS eid
            LIMIT $candidate_limit
            """,
            {"region": normalize_text(region)},
        )
    if instrumental:
        _run_candidate_query(
            f"""
            MATCH (s:Song)
            WHERE {playable}
              AND (
                coalesce(properties(s)['is_instrumental'], properties(s)['instrumental'], false)
                OR toLower(toString(coalesce(s.language, ''))) CONTAINS 'instrumental'
                OR toString(coalesce(s.language, '')) CONTAINS '纯音乐'
                OR toString(coalesce(s.language, '')) CONTAINS '器乐'
              )
            RETURN DISTINCT elementId(s) AS eid
            LIMIT $candidate_limit
            """,
            {},
        )

    if not candidate_eids:
        return "[]"

    query = """
    UNWIND $eids AS eid
    MATCH (s)
    WHERE elementId(s) = eid
    OPTIONAL MATCH (s)-[:PERFORMED_BY]->(a:Artist)
    OPTIONAL MATCH (s)-[:BELONGS_TO_GENRE]->(g:Genre)
    OPTIONAL MATCH (s)-[:HAS_MOOD]->(m:Mood)
    OPTIONAL MATCH (s)-[:HAS_THEME]->(t:Theme)
    OPTIONAL MATCH (s)-[:FITS_SCENARIO]->(sc:Scenario)
    RETURN elementId(s) AS eid, s.title AS title,
           collect(DISTINCT a.name) AS artists, s.album AS album,
           s.audio_url AS audio_url, s.cover_url AS cover_url, s.lrc_url AS lrc_url,
           coalesce(s.language, 'Unknown') AS language,
           coalesce(s.region, 'Unknown') AS region,
           (
             coalesce(properties(s)['is_instrumental'], properties(s)['instrumental'], false)
             OR toLower(toString(coalesce(s.language, ''))) CONTAINS 'instrumental'
             OR toString(coalesce(s.language, '')) CONTAINS '纯音乐'
             OR toString(coalesce(s.language, '')) CONTAINS '器乐'
           ) AS is_instrumental,
           properties(s)['has_vocal'] AS has_vocal,
           properties(s)['has_drums'] AS has_drums,
           properties(s)['energy_level'] AS energy_level,
           properties(s)['acoustic_vocalness'] AS acoustic_vocalness,
           properties(s)['acoustic_drumness'] AS acoustic_drumness,
           properties(s)['acoustic_energy'] AS acoustic_energy,
           properties(s)['acoustic_probe_version'] AS acoustic_probe_version,
           collect(DISTINCT g.name) AS genres,
           collect(DISTINCT m.name) AS moods,
           collect(DISTINCT t.name) AS themes,
           collect(DISTINCT sc.name) AS scenarios,
           coalesce(s.updated_at, 0) AS updated_at
    """
    rows = client.execute_query(query, {"eids": candidate_eids})

    def _contains(value: str, options: List[str]) -> bool:
        normalized = normalize_text(value)
        return bool(
            normalized
            and any(
                option in normalized or normalized in option
                for option in (normalize_text(item) for item in options)
                if option
            )
        )

    for row in rows:
        title = str(row.get("title") or "")
        artist_names = _clean_list(row.get("artists"))
        title_exact = any(normalize_text(title) == normalize_text(item) for item in songs)
        artist_exact = any(
            normalize_text(name) == normalize_text(item)
            for name in artist_names
            for item in artists
        )
        tags = (
            _clean_list(row.get("genres"))
            + _clean_list(row.get("themes"))
            + _clean_list(row.get("moods"))
            + _clean_list(row.get("scenarios"))
        )
        language_match = bool(
            language
            and normalize_text(row.get("language")) == normalize_text(language)
        )
        region_match = bool(
            region
            and normalize_text(row.get("region")) == normalize_text(region)
        )
        instrumental_match = bool(
            instrumental
            and (
                normalize_text(row.get("language")) == "instrumental"
                or "instrumental" in normalize_text(row.get("language"))
                or "纯音乐" in str(row.get("language") or "")
                or "器乐" in str(row.get("language") or "")
            )
        )

        # Hard label constraints are also recall signals.  Without this, a
        # language-only query never enters graph recall and sparse languages
        # may be absent from the RRF pool even when they exist in the catalog.
        if language and not language_match:
            continue
        if region and not region_match:
            continue
        row["recall_score"] = (
            (8.0 if title_exact else 6.0 if _contains(title, songs) else 0.0)
            + (
                7.0
                if artist_exact
                else 5.0
                if any(_contains(name, artists) for name in artist_names)
                else 0.0
            )
            + (1.5 if any(_contains(tag, genres) for tag in tags) else 0.0)
            + (
                1.0
                if any(_contains(tag, moods) for tag in _clean_list(row.get("moods")))
                else 0.0
            )
            + (
                1.0
                if any(
                    _contains(tag, scenarios)
                    for tag in _clean_list(row.get("scenarios"))
                )
                else 0.0
            )
            + (4.0 if language_match else 0.0)
            + (3.0 if region_match else 0.0)
            + (4.0 if instrumental_match else 0.0)
        )
    rows = sorted(
        (row for row in rows if float(row.get("recall_score") or 0.0) > 0),
        key=lambda row: (
            float(row.get("recall_score") or 0.0),
            int(row.get("updated_at") or 0),
            str(row.get("title") or ""),
        ),
        reverse=True,
    )[:limit]
    return _records_to_json(rows, "recall_score")
