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
    acoustic_preference_weight: float = 0.04
    acoustic_conflict_weight: float = 0.06
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
    for field in ("genre", "genres", "moods", "themes", "scenarios", "language", "region", "energy_level"):
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
    if song.get("is_instrumental") or song.get("instrumental"):
        tokens.add("instrumental")
        tokens.add("withoutvocals")
    if song.get("has_vocal") is False:
        tokens.add("instrumental")
        tokens.add("withoutvocals")
        tokens.add("novocals")
    elif song.get("has_vocal") is True:
        tokens.add("vocal")
        tokens.add("vocals")
    if song.get("has_drums") is False:
        tokens.add("nodrums")
        tokens.add("withoutdrums")
    elif song.get("has_drums") is True:
        tokens.add("drums")
    if _normalise_tag(song.get("energy_level")) in {"low", "lowenergy", "quiet", "calm"}:
        tokens.add("lowenergy")
        tokens.add("quiet")
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


_NO_VOCAL_TERMS = (
    "不要人声",
    "无人声",
    "无歌词",
    "without vocals",
    "no vocals",
    "instrumental",
    "纯音乐",
    "器乐",
)

_VOCAL_TERMS = ("vocals", "vocal", "singing", "人声", "演唱", "歌声")
_DRUM_TERMS = ("drums", "drum", "percussion", "鼓", "鼓点", "打击乐")
_NO_DRUM_TERMS = ("no drums", "without drums", "少鼓", "弱鼓", "不要鼓", "无鼓")
_LOW_ENERGY_TERMS = (
    "low energy",
    "very low energy",
    "低能量",
    "低动态",
    "安静",
    "quiet",
    "sleep",
    "睡前",
    "soft",
    "gentle",
    "calm",
    "放松",
    "舒缓",
)
_HIGH_ENERGY_TERMS = (
    "high energy",
    "energetic",
    "loud",
    "party",
    "edm",
    "driving",
    "aggressive",
    "突然变响",
    "太吵",
    "炸",
    "蹦迪",
)


def _plan_text_sections(
    soft_intent: Mapping[str, Any] | None = None,
    hints: Mapping[str, Any] | None = None,
) -> tuple[str, str]:
    soft = soft_intent or {}
    hint = hints or {}
    positive_parts: list[str] = []
    for value in (
        soft.get("goal"),
        soft.get("trajectory"),
        soft.get("vibe"),
        hint.get("genres"),
        hint.get("mood"),
        hint.get("scenario"),
    ):
        positive_parts.extend(_iter_terms(value))
    avoid_parts = _iter_terms(soft.get("avoid"))
    return (
        " ".join(positive_parts).casefold(),
        " ".join(avoid_parts).casefold(),
    )


def _has_term(text: str, terms: tuple[str, ...]) -> bool:
    folded = str(text or "").casefold()
    return any(str(term).casefold() in folded for term in terms)


def acoustic_probe_fit_scores(
    song: Mapping[str, Any],
    *,
    metadata: Mapping[str, Any] | None = None,
    soft_intent: Mapping[str, Any] | None = None,
    hints: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return soft acoustic evidence from MuQ-derived probe fields.

    This consumes LLM-produced plan fields, not raw user-query trigger words.
    Scores are soft nudges for post-recall ranking and should never be used as
    hard filters.
    """
    meta = metadata or {}
    positive_text, avoid_text = _plan_text_sections(soft_intent, hints)
    if not positive_text and not avoid_text:
        return {"active": False, "positive": 0.0, "conflict": 0.0, "positive_hits": [], "conflict_hits": []}

    vocalness = _to_float(meta.get("acoustic_vocalness", song.get("acoustic_vocalness", song.get("has_vocal"))), 0.5)
    drumness = _to_float(meta.get("acoustic_drumness", song.get("acoustic_drumness", song.get("has_drums"))), 0.5)
    energy = _to_float(meta.get("acoustic_energy", song.get("acoustic_energy")), 0.5)
    if song.get("has_vocal") is False:
        vocalness = min(vocalness, 0.15)
    elif song.get("has_vocal") is True and "acoustic_vocalness" not in meta and "acoustic_vocalness" not in song:
        vocalness = max(vocalness, 0.85)
    if song.get("has_drums") is False:
        drumness = min(drumness, 0.15)
    elif song.get("has_drums") is True and "acoustic_drumness" not in meta and "acoustic_drumness" not in song:
        drumness = max(drumness, 0.85)

    positive_values: list[float] = []
    conflict_values: list[float] = []
    positive_hits: list[str] = []
    conflict_hits: list[str] = []

    wants_no_vocals = _has_term(positive_text, _NO_VOCAL_TERMS) or _has_term(avoid_text, _VOCAL_TERMS)
    wants_vocals = _has_term(positive_text, _VOCAL_TERMS) or _has_term(avoid_text, _NO_VOCAL_TERMS)
    wants_low_drums = _has_term(positive_text, _NO_DRUM_TERMS) or _has_term(avoid_text, _DRUM_TERMS)
    wants_drums = _has_term(positive_text, _DRUM_TERMS) or _has_term(avoid_text, _NO_DRUM_TERMS)
    wants_low_energy = _has_term(positive_text, _LOW_ENERGY_TERMS) or _has_term(avoid_text, _HIGH_ENERGY_TERMS)
    wants_high_energy = _has_term(positive_text, _HIGH_ENERGY_TERMS) or _has_term(avoid_text, _LOW_ENERGY_TERMS)

    if wants_no_vocals and not wants_vocals:
        positive_values.append(1.0 - vocalness)
        conflict_values.append(vocalness)
        positive_hits.append("no_vocal")
        if vocalness > 0.55:
            conflict_hits.append("vocalness")
    elif wants_vocals and not wants_no_vocals:
        positive_values.append(vocalness)
        conflict_values.append(1.0 - vocalness)
        positive_hits.append("vocal")
        if vocalness < 0.45:
            conflict_hits.append("instrumentalness")

    if wants_low_drums and not wants_drums:
        positive_values.append(1.0 - drumness)
        conflict_values.append(drumness)
        positive_hits.append("low_drums")
        if drumness > 0.55:
            conflict_hits.append("drumness")
    elif wants_drums and not wants_low_drums:
        positive_values.append(drumness)
        conflict_values.append(1.0 - drumness)
        positive_hits.append("drums")
        if drumness < 0.45:
            conflict_hits.append("low_drumness")

    if wants_low_energy and not wants_high_energy:
        positive_values.append(1.0 - energy)
        conflict_values.append(energy)
        positive_hits.append("low_energy")
        if energy > 0.55:
            conflict_hits.append("energy")
    elif wants_high_energy and not wants_low_energy:
        positive_values.append(energy)
        conflict_values.append(1.0 - energy)
        positive_hits.append("high_energy")
        if energy < 0.45:
            conflict_hits.append("low_energy")

    if not positive_values and not conflict_values:
        return {"active": False, "positive": 0.0, "conflict": 0.0, "positive_hits": [], "conflict_hits": []}

    positive = _clamp(sum(positive_values) / max(len(positive_values), 1), 0.0, 1.0)
    conflict = _clamp(sum(conflict_values) / max(len(conflict_values), 1), 0.0, 1.0)
    return {
        "active": True,
        "positive": positive,
        "conflict": conflict,
        "positive_hits": sorted(set(positive_hits)),
        "conflict_hits": sorted(set(conflict_hits)),
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
    enable_acoustic_probe: bool = False,
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
        # Exposure is user-scoped metadata from (User)-[:EXPOSED]->(Song).
        # Never fall back to legacy global Song.ts_* fields, which leak state
        # across users and make offline evaluation order-dependent.
        ts_alpha = _to_float(meta.get("ts_alpha", 1.0), 1.0)
        ts_beta = _to_float(meta.get("ts_beta", 1.0), 1.0)
        last_exposed = meta.get("ts_last_exposed_at", 0)

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
        acoustic = (
            acoustic_probe_fit_scores(
                song,
                metadata=meta,
                soft_intent=soft_intent,
                hints=hints,
            )
            if enable_acoustic_probe
            else {
                "active": False,
                "positive": 0.0,
                "conflict": 0.0,
                "positive_hits": [],
                "conflict_hits": [],
            }
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
            + config.acoustic_preference_weight * float(acoustic["positive"])
            - config.acoustic_conflict_weight * float(acoustic["conflict"])
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
        item["_post_acoustic_positive_score"] = round(float(acoustic["positive"]), 4)
        item["_post_acoustic_conflict_score"] = round(float(acoustic["conflict"]), 4)
        item["_post_acoustic_positive_hits"] = acoustic["positive_hits"]
        item["_post_acoustic_conflict_hits"] = acoustic["conflict_hits"]
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
            "acoustic_positive": item["_post_acoustic_positive_score"],
            "acoustic_conflict": item["_post_acoustic_conflict_score"],
            "delta": item["_post_recall_delta"],
        }

        if apply_to_similarity:
            original = _base_score(item, score_field)
            item["_pre_post_recall_score"] = round(original, 6)
            item["similarity_score"] = round(_clamp(original + delta, 0.0, 1.0), 6)

    return candidates
