"""Independent local recall sources used by the R1 retrieval pipeline."""

from __future__ import annotations

import json
import math
import os
import re
import threading
import time
import unicodedata
from pathlib import Path
from typing import Any, Iterable, List, Mapping
from urllib.parse import quote

from config.logging_config import get_logger
from config.settings import settings
from retrieval.neo4j_client import get_neo4j_client
from retrieval.retrieval_fusion import normalize_text


logger = get_logger(__name__)
CATALOG_CACHE_TTL_SECONDS = 300

_catalog_lock = threading.Lock()
_catalog_cache: List[dict] = []
_catalog_cache_at = 0.0
_bm25_cache: Any = None
_bm25_tokens: List[List[str]] = []


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
    if not any((artists, songs, genres, moods, scenarios, language, region, instrumental)):
        return "[]"

    query = """
    MATCH (s:Song)
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
           collect(DISTINCT g.name) AS genres,
           collect(DISTINCT m.name) AS moods,
           collect(DISTINCT t.name) AS themes,
           collect(DISTINCT sc.name) AS scenarios,
           coalesce(s.updated_at, 0) AS updated_at
    """
    client = get_neo4j_client()
    rows = client.execute_query(query)

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
            and normalize_text(row.get("language")) == "instrumental"
        )

        # Hard label constraints are also recall signals.  Without this, a
        # language-only query never enters graph recall and sparse languages
        # may be absent from the RRF pool even when they exist in the catalog.
        if language and not language_match:
            continue
        if region and not region_match:
            continue
        if instrumental and not instrumental_match:
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


def _tokenize(text: str) -> List[str]:
    normalized = unicodedata.normalize("NFKC", text or "").casefold()
    latin = re.findall(r"[a-z0-9]+", normalized)
    cjk_runs = re.findall(r"[\u3400-\u9fff]+", normalized)
    cjk = []
    for run in cjk_runs:
        cjk.extend(run)
        cjk.extend(run[index : index + 2] for index in range(len(run) - 1))
    return latin + cjk


def _lyrics_roots() -> List[Path]:
    configured = (
        os.getenv("MUSIC_DATA_ROOT")
        or os.getenv("MUSIC_DATA_PATH")
        or "data"
    )
    root = Path(configured).expanduser()
    return [root, root / "processed_audio", root / "online_acquired"]


def _read_lyrics(lrc_url: str | None) -> str:
    if not lrc_url:
        return ""
    relative = str(lrc_url).split("?", 1)[0].lstrip("/")
    if relative.startswith("static/"):
        relative = relative[len("static/") :]
    for root in _lyrics_roots():
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root.resolve())
        except ValueError:
            continue
        if candidate.is_file():
            try:
                text = candidate.read_text(encoding="utf-8", errors="ignore")
                return re.sub(r"\[[^\]]+\]", " ", text)[:20000]
            except OSError:
                return ""
    return ""


def _load_catalog() -> List[dict]:
    global _catalog_cache, _catalog_cache_at, _bm25_cache, _bm25_tokens
    now = time.monotonic()
    if _catalog_cache and now - _catalog_cache_at < CATALOG_CACHE_TTL_SECONDS:
        return _catalog_cache

    with _catalog_lock:
        now = time.monotonic()
        if _catalog_cache and now - _catalog_cache_at < CATALOG_CACHE_TTL_SECONDS:
            return _catalog_cache

        query = """
        MATCH (s:Song)
        OPTIONAL MATCH (s)-[:PERFORMED_BY]->(a:Artist)
        OPTIONAL MATCH (s)-[:BELONGS_TO_GENRE]->(g:Genre)
        OPTIONAL MATCH (s)-[:HAS_MOOD]->(m:Mood)
        OPTIONAL MATCH (s)-[:HAS_THEME]->(t:Theme)
        OPTIONAL MATCH (s)-[:FITS_SCENARIO]->(sc:Scenario)
        RETURN s.title AS title, collect(DISTINCT a.name) AS artists,
               s.album AS album, s.audio_url AS audio_url,
               s.cover_url AS cover_url, s.lrc_url AS lrc_url,
               coalesce(s.language, 'Unknown') AS language,
               coalesce(s.region, 'Unknown') AS region,
               collect(DISTINCT g.name) AS genres,
               collect(DISTINCT m.name) AS moods,
               collect(DISTINCT t.name) AS themes,
               collect(DISTINCT sc.name) AS scenarios
        """
        rows = get_neo4j_client().execute_query(query)
        documents = []
        tokens = []
        for row in rows:
            lyrics = _read_lyrics(row.get("lrc_url"))
            title_tokens = _tokenize(str(row.get("title") or ""))
            artist_tokens = _tokenize(" ".join(_clean_list(row.get("artists"))))
            tag_text = " ".join(
                _clean_list(row.get("genres"))
                + _clean_list(row.get("moods"))
                + _clean_list(row.get("themes"))
                + _clean_list(row.get("scenarios"))
            )
            document_tokens = (
                title_tokens * 4
                + artist_tokens * 3
                + _tokenize(tag_text)
                + _tokenize(lyrics)
            )
            documents.append(dict(row))
            tokens.append(document_tokens or ["__empty__"])

        try:
            from rank_bm25 import BM25Okapi

            bm25 = BM25Okapi(tokens)
        except ImportError:
            bm25 = None
            logger.warning("[BM25] rank_bm25 未安装，使用内置 Okapi 兼容实现")

        _catalog_cache = documents
        _catalog_cache_at = now
        _bm25_cache = bm25
        _bm25_tokens = tokens
        logger.info("[BM25] 目录索引已刷新: %d 首，TTL=%ds", len(documents), CATALOG_CACHE_TTL_SECONDS)
        return _catalog_cache


def _fallback_bm25_scores(query_tokens: List[str]) -> List[float]:
    documents = _bm25_tokens
    if not documents or not query_tokens:
        return [0.0] * len(documents)
    doc_count = len(documents)
    avg_len = sum(len(doc) for doc in documents) / max(doc_count, 1)
    frequencies = {}
    for token in set(query_tokens):
        frequencies[token] = sum(1 for doc in documents if token in doc)

    scores = []
    k1, b = 1.5, 0.75
    for doc in documents:
        term_counts = {token: doc.count(token) for token in set(query_tokens)}
        score = 0.0
        for token, tf in term_counts.items():
            if tf == 0:
                continue
            df = frequencies.get(token, 0)
            idf = math.log(1 + (doc_count - df + 0.5) / (df + 0.5))
            denominator = tf + k1 * (1 - b + b * len(doc) / max(avg_len, 1.0))
            score += idf * (tf * (k1 + 1)) / denominator
        scores.append(score)
    return scores


def lexical_bm25_recall(query: str, *, limit: int) -> str:
    catalog = _load_catalog()
    query_tokens = _tokenize(query)
    if not catalog or not query_tokens:
        return "[]"

    if _bm25_cache is not None:
        scores = list(_bm25_cache.get_scores(query_tokens))
    else:
        scores = _fallback_bm25_scores(query_tokens)

    ranked = sorted(
        ((score, index) for index, score in enumerate(scores) if score > 0),
        key=lambda pair: pair[0],
        reverse=True,
    )[:limit]
    results = []
    for score, index in ranked:
        item = _record_to_song(catalog[index])
        item["similarity_score"] = float(score)
        results.append(item)
    return json.dumps(results, ensure_ascii=False)
