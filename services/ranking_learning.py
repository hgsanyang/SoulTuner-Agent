"""Auditable offline ranking learner for SoulTuner exposure feedback.

Only explicit positive/negative events are labels. An exposed-but-untouched
song is never treated as a negative example.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import math
from typing import Any, Iterable


FEATURE_NAMES = (
    "rrf_graph",
    "rrf_dense",
    "rrf_lexical",
    "semantic",
    "acoustic",
    "personal",
    "freshness",
    "longtail",
    "exposure_penalty",
)
POSITIVE_REWARDS = {"full_play": 1.0, "like": 2.0, "save": 2.0, "repeat": 3.0}
NEGATIVE_REWARDS = {"skip": -1.0, "dislike": -3.0}
SLATE_POSITIVE_RATINGS = {"great": 0.35}
SLATE_NEGATIVE_RATINGS = {
    "off": -0.45,
    "wrong_context": -0.45,
    "too_noisy": -0.35,
    "too_sad": -0.35,
    "too_quiet": -0.25,
    "too_generic": -0.30,
    "too_familiar": -0.25,
}
SLATE_NEUTRAL_RATINGS = {"partial", "more_discovery", "more_niche", "closer_to_seed"}
BASELINE_COEFFICIENTS = {
    "rrf_graph": 0.30,
    "rrf_dense": 0.35,
    "rrf_lexical": 0.30,
    "semantic": 0.45,
    "acoustic": 0.30,
    "personal": 0.06,
    "freshness": 0.035,
    "longtail": 0.025,
    "exposure_penalty": -0.06,
}


def _normalise_identity(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


def _song_key(title: Any, artist: Any) -> str:
    return f"{_normalise_identity(title)}\0{_normalise_identity(artist)}"


def _safe_unit(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _rank_feature(source_ranks: dict[str, Any], source: str) -> float:
    try:
        rank = max(1, int(source_ranks.get(source)))
    except (TypeError, ValueError):
        return 0.0
    return 61.0 / (60.0 + rank)


def feature_vector(item: dict[str, Any]) -> dict[str, float]:
    source_ranks = item.get("source_ranks") or {}
    return {
        "rrf_graph": _rank_feature(source_ranks, "graph"),
        "rrf_dense": _rank_feature(source_ranks, "dense"),
        "rrf_lexical": _rank_feature(source_ranks, "lexical"),
        "semantic": _safe_unit(item.get("semantic_score"), 0.5),
        "acoustic": _safe_unit(item.get("acoustic_score"), 0.5),
        "personal": _safe_unit(item.get("personal_score"), 0.5),
        "freshness": _safe_unit(item.get("freshness_score"), 0.0),
        "longtail": _safe_unit(item.get("longtail_score"), 0.0),
        "exposure_penalty": _safe_unit(item.get("exposure_penalty"), 0.0),
    }


def _event_reward(event_type: Any) -> float | None:
    event = str(event_type or "")
    if event in POSITIVE_REWARDS:
        return POSITIVE_REWARDS[event]
    if event in NEGATIVE_REWARDS:
        return NEGATIVE_REWARDS[event]
    return None


def build_strict_labeled_rows(
    exposures: list[dict[str, Any]],
    events: list[dict[str, Any]],
    *,
    attribution_window_ms: int = 86_400_000,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Join events only to their exact exposure and item identity."""
    exposure_map = {
        str(exposure.get("exposure_id")): exposure
        for exposure in exposures
        if exposure.get("exposure_id")
    }
    diagnostics: defaultdict[str, int] = defaultdict(int)
    rows: list[dict[str, Any]] = []
    seen_event_ids: set[str] = set()

    for event in sorted(events, key=lambda row: int(row.get("ts") or 0)):
        reward = _event_reward(event.get("event_type"))
        if reward is None:
            diagnostics["neutral_events"] += 1
            continue
        event_id = str(event.get("event_id") or "")
        if event_id and event_id in seen_event_ids:
            diagnostics["duplicate_events"] += 1
            continue
        if event_id:
            seen_event_ids.add(event_id)

        exposure_id = str(event.get("exposure_id") or "")
        if not exposure_id:
            diagnostics["missing_exposure_id"] += 1
            continue
        exposure = exposure_map.get(exposure_id)
        if exposure is None:
            diagnostics["unknown_exposure_id"] += 1
            continue

        exposure_ts = int(exposure.get("ts") or 0)
        event_ts = int(event.get("ts") or 0)
        if exposure_ts and event_ts and not (
            exposure_ts <= event_ts <= exposure_ts + attribution_window_ms
        ):
            diagnostics["outside_attribution_window"] += 1
            continue

        wanted = _song_key(event.get("title"), event.get("artist"))
        match = next(
            (
                item
                for item in exposure.get("items") or []
                if _song_key(item.get("title"), item.get("artist")) == wanted
            ),
            None,
        )
        if match is None:
            diagnostics["song_not_in_exposure"] += 1
            continue
        rows.append(
            {
                "event_id": event_id,
                "event_type": event.get("event_type"),
                "exposure_id": exposure_id,
                "user_id": str(event.get("user_id") or exposure.get("user_id") or "local_admin"),
                "intent_type": str(exposure.get("intent_type") or ""),
                "ts": event_ts,
                "title": match.get("title"),
                "artist": match.get("artist"),
                "rank": match.get("rank"),
                "reward": reward,
                "label": 1 if reward > 0 else 0,
                "sample_weight": abs(reward),
                "features": feature_vector(match),
            }
        )
        diagnostics["matched_events"] += 1
    diagnostics["positive_rows"] = sum(1 for row in rows if row["label"] == 1)
    diagnostics["negative_rows"] = sum(1 for row in rows if row["label"] == 0)
    return rows, dict(diagnostics)


def _slate_reward(rating: Any) -> float | None:
    value = str(rating or "").strip()
    if value in SLATE_POSITIVE_RATINGS:
        return SLATE_POSITIVE_RATINGS[value]
    if value in SLATE_NEGATIVE_RATINGS:
        return SLATE_NEGATIVE_RATINGS[value]
    return None


def build_slate_feedback_rows(
    exposures: list[dict[str, Any]],
    slate_feedback: list[dict[str, Any]],
    *,
    top_k: int = 5,
    attribution_window_ms: int = 7 * 86_400_000,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Convert high-confidence whole-slate feedback into low-weight rows.

    Slate labels are weaker than song-level events, but they are still explicit
    user feedback about the ranked list.  Ambiguous ratings such as
    ``more_discovery`` are intentionally ignored here and handled by the
    MemoryGateway preference layer instead.
    """
    exposure_map = {
        str(exposure.get("exposure_id")): exposure
        for exposure in exposures
        if exposure.get("exposure_id")
    }
    diagnostics: defaultdict[str, int] = defaultdict(int)
    rows: list[dict[str, Any]] = []

    for feedback in sorted(slate_feedback, key=lambda row: int(row.get("ts") or 0)):
        rating = str(feedback.get("rating") or "").strip()
        reward = _slate_reward(rating)
        if reward is None:
            if rating in SLATE_NEUTRAL_RATINGS:
                diagnostics["neutral_slate_feedback"] += 1
            else:
                diagnostics["unsupported_slate_rating"] += 1
            continue

        exposure_id = str(feedback.get("exposure_id") or "")
        if not exposure_id:
            diagnostics["missing_exposure_id"] += 1
            continue
        exposure = exposure_map.get(exposure_id)
        if exposure is None:
            diagnostics["unknown_exposure_id"] += 1
            continue

        exposure_ts = int(exposure.get("ts") or 0)
        feedback_ts = int(feedback.get("ts") or 0)
        if exposure_ts and feedback_ts and not (
            exposure_ts <= feedback_ts <= exposure_ts + attribution_window_ms
        ):
            diagnostics["outside_attribution_window"] += 1
            continue

        items = sorted(
            list(exposure.get("items") or []),
            key=lambda item: int(item.get("rank") or 999_999),
        )[: max(1, int(top_k))]
        if not items:
            diagnostics["exposure_without_items"] += 1
            continue

        label = 1 if reward > 0 else 0
        for item in items:
            rows.append(
                {
                    "event_id": f"slate:{feedback.get('feedback_id') or exposure_id}:{item.get('rank')}",
                    "event_type": f"slate:{rating}",
                    "exposure_id": exposure_id,
                    "user_id": str(feedback.get("user_id") or exposure.get("user_id") or "local_admin"),
                    "intent_type": str(exposure.get("intent_type") or ""),
                    "ts": feedback_ts,
                    "title": item.get("title"),
                    "artist": item.get("artist"),
                    "rank": item.get("rank"),
                    "reward": reward,
                    "label": label,
                    "sample_weight": abs(reward),
                    "features": feature_vector(item),
                    "label_source": "slate_feedback",
                    "slate_rating": rating,
                }
            )
        diagnostics["matched_slate_feedback"] += 1
        diagnostics["slate_rows"] += len(items)

    diagnostics["positive_rows"] = sum(1 for row in rows if row["label"] == 1)
    diagnostics["negative_rows"] = sum(1 for row in rows if row["label"] == 0)
    return rows, dict(diagnostics)


def build_preference_pairs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create strict positive-vs-negative pairs from the same exposure."""
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["exposure_id"])].append(row)

    pairs: list[dict[str, Any]] = []
    for exposure_id, group in grouped.items():
        positives = [row for row in group if row["label"] == 1]
        negatives = [row for row in group if row["label"] == 0]
        for positive in positives:
            for negative in negatives:
                pairs.append(
                    {
                        "exposure_id": exposure_id,
                        "user_id": positive["user_id"],
                        "positive": positive,
                        "negative": negative,
                        "feature_delta": {
                            name: positive["features"][name] - negative["features"][name]
                            for name in FEATURE_NAMES
                        },
                    }
                )
    return pairs


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _predict(row: dict[str, Any], coefficients: dict[str, float], bias: float) -> float:
    score = bias + sum(coefficients[name] * row["features"][name] for name in FEATURE_NAMES)
    return _sigmoid(score)


def evaluate_rows(
    rows: list[dict[str, Any]],
    coefficients: dict[str, float],
    bias: float,
) -> dict[str, float]:
    if not rows:
        return {"log_loss": 0.0, "accuracy": 0.0}
    loss = 0.0
    correct = 0
    total_weight = 0.0
    for row in rows:
        pred = max(1e-6, min(1.0 - 1e-6, _predict(row, coefficients, bias)))
        label = float(row["label"])
        weight = float(row.get("sample_weight") or 1.0)
        loss += weight * (-(label * math.log(pred) + (1.0 - label) * math.log(1.0 - pred)))
        total_weight += weight
        correct += int((pred >= 0.5) == bool(label))
    return {
        "log_loss": round(loss / max(total_weight, 1e-9), 6),
        "accuracy": round(correct / len(rows), 6),
    }


def evaluate_pairs(
    pairs: list[dict[str, Any]],
    coefficients: dict[str, float],
) -> dict[str, float]:
    if not pairs:
        return {"pair_accuracy": 0.0, "pair_log_loss": 0.0, "pair_count": 0}
    correct = 0
    loss = 0.0
    for pair in pairs:
        margin = sum(
            coefficients[name] * pair["feature_delta"][name]
            for name in FEATURE_NAMES
        )
        probability = max(1e-6, min(1.0 - 1e-6, _sigmoid(margin)))
        correct += int(margin > 0)
        loss += -math.log(probability)
    return {
        "pair_accuracy": round(correct / len(pairs), 6),
        "pair_log_loss": round(loss / len(pairs), 6),
        "pair_count": len(pairs),
    }


def chronological_split(
    rows: list[dict[str, Any]],
    *,
    validation_ratio: float = 0.2,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ordered = sorted(rows, key=lambda row: (int(row.get("ts") or 0), str(row.get("event_id") or "")))
    if len(ordered) < 2:
        return ordered, []
    val_count = max(1, int(round(len(ordered) * validation_ratio)))
    val_count = min(val_count, len(ordered) - 1)
    return ordered[:-val_count], ordered[-val_count:]


def fit_logistic_ranker(
    rows: list[dict[str, Any]],
    *,
    initial: dict[str, float] | None = None,
    learning_rate: float = 0.08,
    epochs: int = 320,
    l2: float = 0.04,
) -> tuple[dict[str, float], float]:
    coefficients = {
        name: float((initial or {}).get(name, 0.0))
        for name in FEATURE_NAMES
    }
    bias = 0.0
    if not rows:
        return coefficients, bias

    for _ in range(max(1, int(epochs))):
        gradient = {name: 0.0 for name in FEATURE_NAMES}
        bias_gradient = 0.0
        total_weight = 0.0
        for row in rows:
            weight = float(row.get("sample_weight") or 1.0)
            error = (_predict(row, coefficients, bias) - float(row["label"])) * weight
            bias_gradient += error
            total_weight += weight
            for name in FEATURE_NAMES:
                gradient[name] += error * row["features"][name]
        scale = 1.0 / max(total_weight, 1e-9)
        bias -= learning_rate * bias_gradient * scale
        for name in FEATURE_NAMES:
            regularized = gradient[name] * scale + l2 * (
                coefficients[name] - BASELINE_COEFFICIENTS[name]
            )
            coefficients[name] -= learning_rate * regularized
            coefficients[name] = max(-4.0, min(4.0, coefficients[name]))
    return coefficients, bias


def _bounded_multiplier(coefficient: float, baseline: float) -> float:
    direction = coefficient - baseline
    return round(max(0.8, min(1.2, math.exp(0.18 * direction))), 6)


def derive_runtime_policy(coefficients: dict[str, float]) -> dict[str, Any]:
    return {
        "rrf_multipliers": {
            source: _bounded_multiplier(
                coefficients[f"rrf_{source}"],
                BASELINE_COEFFICIENTS[f"rrf_{source}"],
            )
            for source in ("graph", "dense", "lexical")
        },
        "content_anchor_multipliers": {
            source: _bounded_multiplier(
                coefficients[source],
                BASELINE_COEFFICIENTS[source],
            )
            for source in ("semantic", "acoustic")
        },
        "post_recall_multipliers": {
            source: _bounded_multiplier(
                coefficients[source],
                BASELINE_COEFFICIENTS[source],
            )
            for source in ("personal", "freshness", "longtail")
        }
        | {
            "exposure_penalty": _bounded_multiplier(
                -coefficients["exposure_penalty"],
                -BASELINE_COEFFICIENTS["exposure_penalty"],
            )
        },
    }


def _has_both_labels(rows: Iterable[dict[str, Any]]) -> bool:
    labels = {int(row["label"]) for row in rows}
    return labels == {0, 1}


def _fit_scope(
    rows: list[dict[str, Any]],
    *,
    min_events: int,
    validation_ratio: float,
    initial: dict[str, float] | None = None,
) -> dict[str, Any]:
    if len(rows) < min_events or not _has_both_labels(rows):
        return {
            "status": "insufficient_data",
            "events": len(rows),
            "positive_events": sum(1 for row in rows if row["label"] == 1),
            "negative_events": sum(1 for row in rows if row["label"] == 0),
        }
    train, validation = chronological_split(rows, validation_ratio=validation_ratio)
    if not validation or not _has_both_labels(validation):
        return {
            "status": "insufficient_validation",
            "events": len(rows),
            "train_events": len(train),
            "validation_events": len(validation),
            "reason": "chronological validation requires both positive and negative events",
        }

    coefficients, bias = fit_logistic_ranker(train, initial=initial or BASELINE_COEFFICIENTS)
    baseline_metrics = evaluate_rows(validation, BASELINE_COEFFICIENTS, 0.0)
    learned_metrics = evaluate_rows(validation, coefficients, bias)
    validation_pairs = build_preference_pairs(validation)
    baseline_pairs = evaluate_pairs(validation_pairs, BASELINE_COEFFICIENTS)
    learned_pairs = evaluate_pairs(validation_pairs, coefficients)
    gate_passed = (
        learned_metrics["log_loss"] <= baseline_metrics["log_loss"]
        and learned_metrics["accuracy"] + 0.02 >= baseline_metrics["accuracy"]
        and (
            not validation_pairs
            or learned_pairs["pair_accuracy"] + 0.02 >= baseline_pairs["pair_accuracy"]
        )
    )
    return {
        "status": "accepted" if gate_passed else "rejected",
        "events": len(rows),
        "train_events": len(train),
        "validation_events": len(validation),
        "coefficients": {name: round(value, 8) for name, value in coefficients.items()},
        "bias": round(bias, 8),
        "runtime_policy": derive_runtime_policy(coefficients),
        "baseline_validation": baseline_metrics | baseline_pairs,
        "learned_validation": learned_metrics | learned_pairs,
        "gate_passed": gate_passed,
    }


def learn_ranking_policy(
    exposures: list[dict[str, Any]],
    events: list[dict[str, Any]],
    slate_feedback: list[dict[str, Any]] | None = None,
    *,
    min_events: int = 20,
    per_user_min_events: int = 30,
    validation_ratio: float = 0.2,
    slate_top_k: int = 5,
) -> dict[str, Any]:
    explicit_rows, explicit_diagnostics = build_strict_labeled_rows(exposures, events)
    slate_rows, slate_diagnostics = build_slate_feedback_rows(
        exposures,
        slate_feedback or [],
        top_k=slate_top_k,
    )
    rows = explicit_rows + slate_rows
    diagnostics = {
        "training_rows": len(rows),
        "explicit_rows": len(explicit_rows),
        "slate_rows": len(slate_rows),
        "explicit": explicit_diagnostics,
        "slate": slate_diagnostics,
    }
    global_model = _fit_scope(
        rows,
        min_events=min_events,
        validation_ratio=validation_ratio,
    )
    user_models: dict[str, Any] = {}
    if global_model.get("status") == "accepted":
        by_user: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_user[row["user_id"]].append(row)
        for user_id, user_rows in by_user.items():
            result = _fit_scope(
                user_rows,
                min_events=per_user_min_events,
                validation_ratio=validation_ratio,
                initial=global_model["coefficients"],
            )
            if result.get("status") == "accepted":
                shrinkage = len(user_rows) / (len(user_rows) + per_user_min_events)
                blended = {
                    name: (
                        shrinkage * result["coefficients"][name]
                        + (1.0 - shrinkage) * global_model["coefficients"][name]
                    )
                    for name in FEATURE_NAMES
                }
                result["shrinkage"] = round(shrinkage, 6)
                result["coefficients"] = {
                    name: round(value, 8) for name, value in blended.items()
                }
                result["runtime_policy"] = derive_runtime_policy(blended)
            user_models[user_id] = result

    return {
        "schema_version": 2,
        "method": "explicit_feedback_logistic_v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": (
            "candidate_accepted"
            if global_model.get("status") == "accepted"
            else global_model.get("status", "rejected")
        ),
        "feature_names": list(FEATURE_NAMES),
        "label_sources": {
            "explicit_song_events": "exact exposure_id + title/artist joins only",
            "slate_feedback": "low-weight top-k rows from high-confidence whole-slate ratings",
            "ignored_slate_ratings": sorted(SLATE_NEUTRAL_RATINGS),
        },
        "diagnostics": diagnostics,
        "strict_preference_pairs": len(build_preference_pairs(rows)),
        "global": global_model,
        "users": user_models,
        "gate_passed": bool(global_model.get("gate_passed")),
    }
