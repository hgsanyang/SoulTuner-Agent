"""Pure retrieval planning, RRF fusion, and hard-constraint filtering."""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, Iterable, List, Mapping


RRF_K = 60

SOURCE_LABELS = {
    "graph": "知识图谱",
    "dense": "稠密语义",
    "lexical": "BM25词法",
    "personal": "个性化",
    "cold": "冷启动",
}

DEFAULT_RECALL_WEIGHTS = {
    "graph": 1.0,
    "dense": 1.0,
    "lexical": 1.0,
    "personal": 0.45,
    "cold": 0.25,
}

INTENT_RECALL_WEIGHTS = {
    "graph_search": {
        "graph": 1.45,
        "dense": 0.70,
        "lexical": 1.50,
        "personal": 0.35,
        "cold": 0.20,
    },
    "hybrid_search": {
        "graph": 1.10,
        "dense": 1.20,
        "lexical": 1.10,
        "personal": 0.50,
        "cold": 0.30,
    },
    "vector_search": {
        "graph": 0.60,
        "dense": 1.50,
        "lexical": 0.70,
        "personal": 0.70,
        "cold": 0.45,
    },
}


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold().strip()
    return re.sub(r"[,，、/\\\s()（）【】\[\]《》'\"`]+", "", text)


def normalize_song_key(title: str, artist: str) -> str:
    return f"{normalize_text(title)}_{normalize_text(artist)}"


def recall_weights_for_intent(intent_type: str | None) -> Dict[str, float]:
    weights = dict(DEFAULT_RECALL_WEIGHTS)
    weights.update(INTENT_RECALL_WEIGHTS.get(str(intent_type or ""), {}))
    return weights


def _merge_song_metadata(target: dict, incoming: dict) -> None:
    list_fields = {"genres", "moods", "themes", "scenarios"}
    for field, value in incoming.items():
        if field in list_fields:
            combined = []
            for item in list(target.get(field) or []) + list(value or []):
                if item and item not in combined:
                    combined.append(item)
            target[field] = combined
            continue
        current = target.get(field)
        if current in (None, "", "Unknown", "未知", "未知艺术家", "未知标题") and value not in (
            None,
            "",
            "Unknown",
            "未知",
        ):
            target[field] = value


def weighted_rrf(
    source_items: Mapping[str, List[dict]],
    weights: Mapping[str, float],
    *,
    k: int = RRF_K,
) -> List[dict]:
    """Fuse ranked recall lists while preserving source ranks and metadata."""
    fused: Dict[str, dict] = {}

    for source, items in source_items.items():
        weight = max(float(weights.get(source, 1.0)), 0.0)
        if weight == 0:
            continue
        for fallback_rank, item in enumerate(items, start=1):
            song = dict(item.get("song") or {})
            title = song.get("title", "")
            artist = song.get("artist", "")
            key = item.get("key") or normalize_song_key(title, artist)
            if not normalize_text(title):
                continue

            rank = int(item.get("rank", fallback_rank - 1)) + 1
            contribution = weight / (k + rank)
            entry = fused.setdefault(
                key,
                {
                    "song": song,
                    "similarity_score": 0.0,
                    "_rrf_score": 0.0,
                    "_source_ranks": {},
                    "_recall_sources": [],
                },
            )
            _merge_song_metadata(entry["song"], song)
            entry["_rrf_score"] += contribution
            entry["similarity_score"] = entry["_rrf_score"]
            entry["_source_ranks"][source] = rank
            if source not in entry["_recall_sources"]:
                entry["_recall_sources"].append(source)

    results = list(fused.values())
    for item in results:
        labels = [SOURCE_LABELS.get(source, source) for source in item["_recall_sources"]]
        item["reason"] = "RRF召回来源: " + " + ".join(labels)
        item["_both_engines"] = len(labels) > 1
        item["_rrf_score"] = round(item["_rrf_score"], 8)
        item["similarity_score"] = item["_rrf_score"]

    results.sort(
        key=lambda item: (
            item.get("_rrf_score", 0.0),
            -min(item.get("_source_ranks", {}).values(), default=10**9),
        ),
        reverse=True,
    )
    return results


def _matches_any(value: Any, aliases: Iterable[str]) -> bool:
    normalized = normalize_text(value)
    wanted = [normalize_text(alias) for alias in aliases if normalize_text(alias)]
    return bool(normalized and any(alias in normalized or normalized in alias for alias in wanted))


def _language_matches(song: Mapping[str, Any], language: str) -> bool:
    expected = normalize_text(language)
    actual = normalize_text(song.get("language"))
    if expected == "instrumental":
        return actual == "instrumental" or bool(song.get("is_instrumental"))
    return bool(actual and actual == expected)


def _has_known_label(value: Any) -> bool:
    normalized = normalize_text(value)
    return bool(normalized and normalized not in {"unknown", "none", "null", "na", "n/a", "未知", "未标注"})


def apply_hard_filters(
    candidates: List[dict],
    hard_constraints: Mapping[str, Any] | None,
    disliked_titles: Iterable[str] = (),
    *,
    limit: int | None = None,
    logger: Any | None = None,
) -> List[dict]:
    """Apply the only exclusion stage: request hard constraints plus safety.

    DISLIKES, explicit entities, and instrumental requests stay strict. Sparse
    language/region metadata is only relaxed when strict language/region filtering
    would empty the result set.
    """
    hard = dict(hard_constraints or {})
    artist_entities = list(hard.get("artist_entities") or [])
    song_entities = list(hard.get("song_entities") or [])
    language = hard.get("language")
    region = hard.get("region")
    instrumental = bool(hard.get("instrumental"))
    disliked = {normalize_text(title) for title in disliked_titles if normalize_text(title)}

    safety_filtered = []
    for item in candidates:
        song = item.get("song") or {}
        if normalize_text(song.get("title", "")) not in disliked:
            safety_filtered.append(item)

    entity_filtered = []
    for item in safety_filtered:
        song = item.get("song") or {}
        title = song.get("title", "")
        artist = song.get("artist", "")

        if artist_entities and not _matches_any(artist, artist_entities):
            continue
        if song_entities and not _matches_any(title, song_entities):
            continue
        if instrumental and not _language_matches(song, "Instrumental"):
            continue
        entity_filtered.append(item)

    def _language_region_status(song: Mapping[str, Any]) -> str:
        """Return match, unknown, or conflict for sparse catalog labels."""
        has_unknown = False
        if language:
            if _has_known_label(song.get("language")):
                if not _language_matches(song, str(language)):
                    return "conflict"
            else:
                has_unknown = True
        if region:
            if _has_known_label(song.get("region")):
                if normalize_text(song.get("region")) != normalize_text(region):
                    return "conflict"
            else:
                has_unknown = True
        return "unknown" if has_unknown else "match"

    has_sparse_label_constraint = bool(language or region)
    if not has_sparse_label_constraint:
        return entity_filtered

    strict_matches = []
    unknown_labels = []
    conflicts = []
    for item in entity_filtered:
        status = _language_region_status(item.get("song") or {})
        if status == "match":
            strict_matches.append(item)
        elif status == "unknown":
            unknown_labels.append(item)
        else:
            conflicts.append(item)

    # Known matches are most trustworthy; missing labels remain eligible instead
    # of being mistaken for conflicts.
    preferred = strict_matches + unknown_labels

    min_required = max(int(limit or 0), 8) if limit is not None else 0
    if not min_required or len(preferred) >= min_required:
        return preferred

    # Preserve entity/instrumental/safety constraints, but fill a sparse language
    # or region result from the remaining RRF candidates rather than returning an
    # unusably short or empty list.
    needed = min_required - len(preferred)
    relaxed_filtered = preferred + conflicts[:needed]
    if logger is not None:
        logger.warning(
            "[HardFilterFallback] language/region filtering left %d/%d preferred "
            "candidates; filled to %d from RRF while keeping entity/safety strict",
            len(preferred),
            len(entity_filtered),
            len(relaxed_filtered),
        )
    return relaxed_filtered
