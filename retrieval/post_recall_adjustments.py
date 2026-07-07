"""Post-recall score adjustments for personalization, freshness, and exposure.

This module deliberately does not recall any new songs.  It only annotates and
slightly adjusts candidates that were already found by content recall.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
import time
from typing import Any, Mapping


DAY_MS = 86_400_000


@dataclass(frozen=True)
class PostRecallAdjustmentConfig:
    personal_weight: float = 0.06
    freshness_weight: float = 0.035
    longtail_weight: float = 0.025
    exposure_penalty_weight: float = 0.06
    semantic_preference_weight: float = 0.035
    semantic_conflict_weight: float = 0.055
    delta_limit: float = 0.08
    freshness_half_life_days: float = 21.0
    exposure_half_life_days: float = 7.0
    exposure_penalty_pivot: float = 3.0


DEFAULT_CONFIG = PostRecallAdjustmentConfig()


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, "", "Unknown"):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _epoch_ms(value: Any) -> float:
    raw = _to_float(value, 0.0)
    if raw <= 0:
        return 0.0
    # Neo4j timestamp() is milliseconds.  Some tests/tools may pass seconds.
    return raw * 1000.0 if raw < 10_000_000_000 else raw


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _normalise(values: list[float], neutral: float = 0.5) -> list[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if math.isclose(lo, hi):
        return [neutral for _ in values]
    span = hi - lo
    return [(value - lo) / span for value in values]


def freshness_score(updated_at_ms: Any, *, now_ms: float, half_life_days: float) -> float:
    updated = _epoch_ms(updated_at_ms)
    if updated <= 0 or now_ms <= updated:
        return 0.0 if updated <= 0 else 1.0
    age_days = (now_ms - updated) / DAY_MS
    half_life = max(float(half_life_days), 0.1)
    return _clamp(0.5 ** (age_days / half_life), 0.0, 1.0)


def decayed_exposure_count(
    ts_beta: Any,
    *,
    last_exposed_at_ms: Any = 0,
    now_ms: float,
    half_life_days: float,
) -> float:
    # beta starts at 1.0.  The part above 1.0 is our exposure count proxy.
    exposure_count = max(_to_float(ts_beta, 1.0) - 1.0, 0.0)
    last_exposed = _epoch_ms(last_exposed_at_ms)
    if exposure_count <= 0 or last_exposed <= 0 or now_ms <= last_exposed:
        return exposure_count
    age_days = (now_ms - last_exposed) / DAY_MS
    half_life = max(float(half_life_days), 0.1)
    return exposure_count * (0.5 ** (age_days / half_life))


def exposure_penalty(effective_exposure: float, *, pivot: float) -> float:
    exposure = max(float(effective_exposure), 0.0)
    pivot = max(float(pivot), 0.1)
    return _clamp(exposure / (exposure + pivot), 0.0, 1.0)


def longtail_score(effective_exposure: float) -> float:
    return _clamp(1.0 / (1.0 + max(float(effective_exposure), 0.0)), 0.0, 1.0)


def _iter_terms(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value is None:
        return []
    text = str(value).strip()
    return [text] if text else []


def _semantic_terms(
    soft_intent: Mapping[str, Any] | None = None,
    hints: Mapping[str, Any] | None = None,
) -> tuple[list[str], list[str]]:
    soft = soft_intent or {}
    hint = hints or {}
    positive: list[str] = []
    for value in (
        hint.get("genres"),
        hint.get("mood"),
        hint.get("scenario"),
    ):
        for term in _iter_terms(value):
            if term and term not in positive:
                positive.append(term)

    conflicts: list[str] = []
    for term in _iter_terms(soft.get("avoid")):
        if term and term not in conflicts:
            conflicts.append(term)
    return positive, conflicts


def _normalise_tag(value: Any) -> str:
    return re.sub(r"[\W_]+", "", str(value or "").casefold())


def _song_tokens(song: Mapping[str, Any]) -> set[str]:
    tokens: set[str] = set()
    for field in ("genre", "genres", "moods", "themes", "scenarios", "language", "region"):
        raw = song.get(field)
        values = raw if isinstance(raw, list) else [raw]
        for item in values:
            text = str(item or "").strip()
            if not text:
                continue
            for piece in re.split(r"[/,，;；|｜\s]+", text):
                normalized = _normalise_tag(piece)
                if normalized:
                    tokens.add(normalized)
            compact = _normalise_tag(text)
            if compact:
                tokens.add(compact)
    return tokens


def _contains_token(tokens: set[str], wanted: str) -> bool:
    target = _normalise_tag(wanted)
    if not target:
        return False
    for token in tokens:
        if token == target:
            return True
        if len(token) >= 4 and len(target) >= 4 and (token in target or target in token):
            return True
    return False


def semantic_fit_scores(
    song: Mapping[str, Any],
    *,
    query_text: str = "",
    soft_intent: Mapping[str, Any] | None = None,
    hints: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return bounded semantic fit evidence from objective catalog tags.

    This is a small post-recall nudge, not a new recall route.  It deliberately
    does not infer user intent from fixed phrase lists.  It only compares the
    LLM-produced plan fields (`hints` and `soft_intent.avoid`) with objective
    catalog tags already present on a recalled candidate.
    """
    del query_text
    positive_terms, conflict_terms = _semantic_terms(soft_intent, hints)
    if not positive_terms and not conflict_terms:
        return {"active": False, "positive": 0.0, "conflict": 0.0, "positive_hits": [], "conflict_hits": []}

    tokens = _song_tokens(song)
    positive_hits = sorted(term for term in positive_terms if _contains_token(tokens, term))
    conflict_hits = sorted(term for term in conflict_terms if _contains_token(tokens, term))
    positive = _clamp(len(positive_hits) / 3.0, 0.0, 1.0)
    conflict = _clamp(len(conflict_hits) / 2.0, 0.0, 1.0)
    return {
        "active": True,
        "positive": positive,
        "conflict": conflict,
        "positive_hits": positive_hits,
        "conflict_hits": conflict_hits,
    }


def _metadata_for(item: Mapping[str, Any], metadata_by_title: Mapping[str, Mapping[str, Any]]) -> Mapping[str, Any]:
    song = item.get("song") or {}
    title = str(song.get("title") or "")
    return metadata_by_title.get(title) or {}


def _base_score(item: Mapping[str, Any], score_field: str) -> float:
    if score_field in item:
        return _to_float(item.get(score_field), 0.0)
    return _to_float(item.get("similarity_score"), 0.0)


def apply_post_recall_adjustments(
    candidates: list[dict],
    *,
    metadata_by_title: Mapping[str, Mapping[str, Any]] | None = None,
    query_text: str = "",
    soft_intent: Mapping[str, Any] | None = None,
    hints: Mapping[str, Any] | None = None,
    score_field: str = "similarity_score",
    output_score_field: str = "_post_recall_score",
    apply_to_similarity: bool = False,
    config: PostRecallAdjustmentConfig = DEFAULT_CONFIG,
    now_ms: float | None = None,
) -> list[dict]:
    """Annotate already-recalled candidates with bounded score adjustments.

    The output delta is intentionally small and clipped to +/- ``delta_limit``.
    This lets objective content matching stay dominant while still allowing
    personalization, freshness, long-tail rescue, and exposure fatigue to
    nudge ordering.
    """
    if not candidates:
        return candidates

    metadata = metadata_by_title or {}
    now = float(now_ms if now_ms is not None else time.time() * 1000.0)
    graph_values = [_to_float(item.get("_graph_affinity"), 0.0) for item in candidates]
    personal_scores = _normalise(graph_values, neutral=0.5)
    base_scores = [_base_score(item, score_field) for item in candidates]
    base_normalised = _normalise(base_scores, neutral=0.5)

    for item, personal, base_norm in zip(candidates, personal_scores, base_normalised):
        meta = _metadata_for(item, metadata)
        song = item.get("song") or {}
        updated_at = meta.get("updated_at", song.get("updated_at", 0))
        ts_alpha = _to_float(meta.get("ts_alpha", song.get("ts_alpha", 1.0)), 1.0)
        ts_beta = _to_float(meta.get("ts_beta", song.get("ts_beta", 1.0)), 1.0)
        last_exposed = meta.get("ts_last_exposed_at", song.get("ts_last_exposed_at", 0))

        effective_exposure = decayed_exposure_count(
            ts_beta,
            last_exposed_at_ms=last_exposed,
            now_ms=now,
            half_life_days=config.exposure_half_life_days,
        )
        freshness = freshness_score(
            updated_at,
            now_ms=now,
            half_life_days=config.freshness_half_life_days,
        )
        semantic = semantic_fit_scores(
            song,
            query_text=query_text,
            soft_intent=soft_intent,
            hints=hints,
        )
        exposure = exposure_penalty(
            effective_exposure,
            pivot=config.exposure_penalty_pivot,
        )
        longtail = longtail_score(effective_exposure)
        delta = (
            config.personal_weight * (personal - 0.5)
            + config.freshness_weight * freshness
            + config.longtail_weight * longtail
            - config.exposure_penalty_weight * exposure
            + config.semantic_preference_weight * float(semantic["positive"])
            - config.semantic_conflict_weight * float(semantic["conflict"])
        )
        delta = _clamp(delta, -config.delta_limit, config.delta_limit)
        adjusted = _clamp(base_norm + delta, 0.0, 1.0)

        item["_post_personal_score"] = round(personal, 4)
        item["_post_freshness_score"] = round(freshness, 4)
        item["_post_longtail_score"] = round(longtail, 4)
        item["_post_exposure_penalty"] = round(exposure, 4)
        item["_post_semantic_positive_score"] = round(float(semantic["positive"]), 4)
        item["_post_semantic_conflict_score"] = round(float(semantic["conflict"]), 4)
        item["_post_semantic_positive_hits"] = semantic["positive_hits"]
        item["_post_semantic_conflict_hits"] = semantic["conflict_hits"]
        item["_post_effective_exposure"] = round(effective_exposure, 4)
        item["_post_ts_alpha"] = round(ts_alpha, 4)
        item["_post_ts_beta"] = round(ts_beta, 4)
        item["_post_recall_delta"] = round(delta, 6)
        item[output_score_field] = round(adjusted, 6)

        song["post_recall_adjustments"] = {
            "personal": item["_post_personal_score"],
            "freshness": item["_post_freshness_score"],
            "longtail": item["_post_longtail_score"],
            "exposure_penalty": item["_post_exposure_penalty"],
            "semantic_positive": item["_post_semantic_positive_score"],
            "semantic_conflict": item["_post_semantic_conflict_score"],
            "delta": item["_post_recall_delta"],
        }

        if apply_to_similarity:
            original = _base_score(item, score_field)
            item["_pre_post_recall_score"] = round(original, 6)
            item["similarity_score"] = round(_clamp(original + delta, 0.0, 1.0), 6)

    return candidates
