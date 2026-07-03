"""JSONL feedback logs for exposure replay and lightweight rank learning."""

from __future__ import annotations

import json
import hashlib
import math
import os
import time
import uuid
from pathlib import Path
from typing import Any


POSITIVE_EVENTS = {"like", "save", "full_play", "repeat"}
NEGATIVE_EVENTS = {"skip", "dislike"}
WEIGHTS_FILE = "ranking_weights.json"
SLATE_FEEDBACK_FILE = "slate_feedback.jsonl"
FEATURE_FIELDS = {
    "semantic": "semantic_score",
    "acoustic": "acoustic_score",
    "personal": "personal_score",
}


def _first_present(item: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in item and item.get(key) is not None:
            return item.get(key)
    return None


def _feedback_dir() -> Path:
    root = os.getenv("MUSIC_FEEDBACK_DIR")
    path = Path(root) if root else Path("data") / "feedback"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _jsonl_path(name: str) -> Path:
    return _feedback_dir() / name


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")


def _song_identity(song: dict[str, Any]) -> dict[str, str]:
    return {
        "title": str(song.get("title") or "").strip(),
        "artist": str(song.get("artist") or "").strip(),
    }


def _feature_snapshot(item: dict[str, Any], rank: int) -> dict[str, Any]:
    song = item.get("song") if isinstance(item.get("song"), dict) else item
    identity = _song_identity(song)
    source_ranks = (
        item.get("_source_ranks")
        or song.get("_source_ranks")
        or {}
    )
    return {
        **identity,
        "rank": rank,
        "music_id": song.get("music_id") or song.get("id"),
        "source": song.get("source") or song.get("recall_source") or item.get("source") or item.get("recall_source"),
        "recall_sources": song.get("recall_sources") or song.get("_recall_sources") or item.get("_recall_sources") or [],
        "score": _first_present(item, "similarity_score", "_post_final_score"),
        "rrf_score": _first_present(item, "_rrf_score"),
        "source_ranks": source_ranks,
        "semantic_score": _first_present(item, "_semantic_score"),
        "acoustic_score": _first_present(item, "_acoustic_score"),
        "personal_score": _first_present(item, "_post_personal_score", "_personal_score"),
        "freshness_score": _first_present(item, "_post_freshness_score"),
        "longtail_score": _first_present(item, "_post_longtail_score"),
        "exposure_penalty": _first_present(item, "_post_exposure_penalty"),
        "post_recall_delta": _first_present(item, "_post_recall_delta"),
        "is_exploration": bool(item.get("_is_exploration")),
        "language": song.get("language"),
        "genres": song.get("genres") or song.get("genre"),
        "moods": song.get("moods"),
        "scenarios": song.get("scenarios"),
    }


def log_exposure(
    *,
    query: str,
    recommendations: list[dict[str, Any]],
    user_id: str = "local_admin",
    request_id: str | None = None,
    intent_type: str = "",
    retrieval_meta: dict[str, Any] | None = None,
    dialog_state: dict[str, Any] | None = None,
    timings: dict[str, Any] | None = None,
) -> str:
    """Persist one recommendation slate for later offline replay."""
    exposure_id = request_id or str(uuid.uuid4())
    rows = [
        _feature_snapshot(song if isinstance(song, dict) else {}, rank=i + 1)
        for i, song in enumerate(recommendations or [])
    ]
    payload = {
        "type": "exposure",
        "exposure_id": exposure_id,
        "ts": int(time.time() * 1000),
        "user_id": user_id,
        "query_hash": hashlib.sha256(str(query or "").encode("utf-8")).hexdigest(),
        "intent_type": intent_type,
        "count": len(rows),
        "items": rows,
        "retrieval_meta": retrieval_meta or {},
        "dialog_state": dialog_state or {},
        "timings": timings or {},
    }
    if os.getenv("FEEDBACK_LOG_RAW_QUERY", "0").lower() in {"1", "true", "yes"}:
        payload["query"] = query
    _append_jsonl(_jsonl_path("exposures.jsonl"), payload)
    return exposure_id


def log_user_event(
    *,
    event_type: str,
    song_title: str,
    artist: str,
    user_id: str = "local_admin",
    exposure_id: str | None = None,
    extra: Any = None,
) -> str:
    extra_payload = extra if isinstance(extra, dict) else {"value": extra} if extra is not None else {}
    event_id = str(uuid.uuid4())
    payload = {
        "type": "event",
        "event_id": event_id,
        "ts": int(time.time() * 1000),
        "user_id": user_id,
        "event_type": event_type,
        "title": str(song_title or "").strip(),
        "artist": str(artist or "").strip(),
        "exposure_id": exposure_id,
        "extra": extra_payload,
        "position": extra_payload.get("position"),
        "play_duration_ms": extra_payload.get("play_duration_ms"),
        "progress_ratio": extra_payload.get("progress_ratio"),
        "session_id": extra_payload.get("session_id"),
    }
    _append_jsonl(_jsonl_path("events.jsonl"), payload)
    return event_id


def log_slate_feedback(
    *,
    exposure_id: str,
    rating: str,
    reasons: list[str] | None = None,
    note: str = "",
    user_id: str = "local_admin",
    extra: dict[str, Any] | None = None,
) -> str:
    """Persist feedback for an entire recommendation slate.

    Song-level feedback tells us which item worked.  Slate-level feedback tells
    us whether the whole ranked list satisfied the current intent, which is the
    signal needed for offline replay and future ranking-policy learning.
    """
    feedback_id = str(uuid.uuid4())
    payload = {
        "type": "slate_feedback",
        "feedback_id": feedback_id,
        "ts": int(time.time() * 1000),
        "user_id": user_id,
        "exposure_id": str(exposure_id or "").strip(),
        "rating": str(rating or "").strip(),
        "reasons": [str(item).strip() for item in (reasons or []) if str(item).strip()],
        "note": str(note or "").strip()[:1000],
        "extra": extra or {},
    }
    _append_jsonl(_jsonl_path(SLATE_FEEDBACK_FILE), payload)
    return feedback_id


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def learned_weights_path() -> Path:
    return _feedback_dir() / WEIGHTS_FILE


def load_learned_tri_anchor_weights() -> dict[str, float] | None:
    path = learned_weights_path()
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        weights = payload.get("weights", payload)
        values = {
            "semantic": float(weights.get("semantic", 0.0)),
            "acoustic": float(weights.get("acoustic", 0.0)),
            "personal": float(weights.get("personal", 0.0)),
        }
        total = sum(max(0.0, v) for v in values.values())
        if total <= 0:
            return None
        return {key: max(0.0, value) / total for key, value in values.items()}
    except Exception:
        return None


def _event_label(event_type: str) -> int | None:
    if event_type in POSITIVE_EVENTS:
        return 1
    if event_type in NEGATIVE_EVENTS:
        return 0
    return None


def _safe_feature(value: Any) -> float:
    if value is None:
        return 0.5
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.5


def build_feedback_training_rows(
    exposures: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Join user feedback events to exposed items and return audit-friendly rows."""
    by_exposure: dict[str, list[dict[str, Any]]] = {
        str(row.get("exposure_id")): list(row.get("items") or [])
        for row in exposures
        if row.get("exposure_id")
    }
    all_items: list[dict[str, Any]] = [item for row in exposures for item in (row.get("items") or [])]
    rows: list[dict[str, Any]] = []

    for event in events:
        label = _event_label(str(event.get("event_type") or ""))
        if label is None:
            continue
        title = str(event.get("title") or "").casefold()
        artist = str(event.get("artist") or "").casefold()
        items = by_exposure.get(str(event.get("exposure_id"))) or all_items
        match = next(
            (
                item
                for item in items
                if str(item.get("title") or "").casefold() == title
                and str(item.get("artist") or "").casefold() == artist
            ),
            None,
        )
        if not match:
            continue
        features = {
            key: _safe_feature(match.get(field))
            for key, field in FEATURE_FIELDS.items()
        }
        rows.append({
            "label": label,
            "event_type": event.get("event_type"),
            "exposure_id": event.get("exposure_id"),
            "title": match.get("title"),
            "artist": match.get("artist"),
            "rank": match.get("rank"),
            "features": features,
            "feature_present": {
                key: match.get(field) is not None
                for key, field in FEATURE_FIELDS.items()
            },
        })
    return rows


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _log_loss(rows: list[dict[str, Any]], coefficients: dict[str, float], bias: float) -> float:
    if not rows:
        return 0.0
    total = 0.0
    for row in rows:
        score = bias + sum(coefficients[key] * row["features"][key] for key in FEATURE_FIELDS)
        pred = max(1e-6, min(1.0 - 1e-6, _sigmoid(score)))
        label = float(row["label"])
        total += -(label * math.log(pred) + (1.0 - label) * math.log(1.0 - pred))
    return total / len(rows)


def _coefficients_to_weights(coefficients: dict[str, float]) -> dict[str, float]:
    # Softmax keeps every anchor non-negative and makes the learned file safe to load.
    largest = max(coefficients.values()) if coefficients else 0.0
    exp_values = {
        key: math.exp(float(value) - largest)
        for key, value in coefficients.items()
    }
    total = sum(exp_values.values()) or 1.0
    return {key: round(value / total, 4) for key, value in exp_values.items()}


def learn_tri_anchor_weights(
    exposures: list[dict[str, Any]],
    events: list[dict[str, Any]],
    *,
    min_events: int = 8,
    learning_rate: float = 0.15,
    epochs: int = 240,
    l2: float = 0.02,
) -> dict[str, Any]:
    """Learn tri-anchor weights from explicit feedback with an audit trail."""
    rows = build_feedback_training_rows(exposures, events)
    positives = sum(1 for row in rows if row["label"] == 1)
    negatives = sum(1 for row in rows if row["label"] == 0)
    feature_coverage = {
        key: round(
            sum(1 for row in rows if row["feature_present"][key]) / len(rows),
            4,
        ) if rows else 0.0
        for key in FEATURE_FIELDS
    }

    audit: dict[str, Any] = {
        "method": "logistic_tri_anchor_v1",
        "status": "ok",
        "matched_events": len(rows),
        "positive_events": positives,
        "negative_events": negatives,
        "feature_coverage": feature_coverage,
        "min_events": min_events,
    }
    if len(rows) < min_events or positives == 0 or negatives == 0:
        audit.update({
            "status": "insufficient_data",
            "reason": "needs at least min_events with both positive and negative labels",
        })
        return audit

    coefficients = {key: 0.0 for key in FEATURE_FIELDS}
    bias = 0.0
    baseline_loss = _log_loss(rows, coefficients, bias)
    for _ in range(max(1, int(epochs))):
        grad = {key: 0.0 for key in FEATURE_FIELDS}
        grad_bias = 0.0
        for row in rows:
            score = bias + sum(coefficients[key] * row["features"][key] for key in FEATURE_FIELDS)
            error = _sigmoid(score) - float(row["label"])
            grad_bias += error
            for key in FEATURE_FIELDS:
                grad[key] += error * row["features"][key] + l2 * coefficients[key]
        scale = 1.0 / len(rows)
        bias -= learning_rate * grad_bias * scale
        for key in FEATURE_FIELDS:
            coefficients[key] -= learning_rate * grad[key] * scale

    learned_loss = _log_loss(rows, coefficients, bias)
    audit.update({
        "weights": _coefficients_to_weights(coefficients),
        "coefficients": {key: round(value, 6) for key, value in coefficients.items()},
        "bias": round(bias, 6),
        "baseline_log_loss": round(baseline_loss, 6),
        "learned_log_loss": round(learned_loss, 6),
        "loss_delta": round(baseline_loss - learned_loss, 6),
    })
    return audit


def estimate_tri_anchor_weights(
    exposures: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    """Estimate simple tri-anchor weights from logged events.

    This is intentionally transparent rather than clever: it joins events to
    exposed items by exposure_id when available, otherwise by title+artist, then
    rewards feature dimensions that were high on positive events and low on
    negative events.
    """
    by_exposure: dict[str, list[dict[str, Any]]] = {
        str(row.get("exposure_id")): list(row.get("items") or [])
        for row in exposures
        if row.get("exposure_id")
    }
    all_items: list[dict[str, Any]] = [item for row in exposures for item in (row.get("items") or [])]

    accum = {"semantic": 1.0, "acoustic": 1.0, "personal": 1.0}
    matched = 0
    positives = 0
    negatives = 0

    for event in events:
        title = str(event.get("title") or "").casefold()
        artist = str(event.get("artist") or "").casefold()
        items = by_exposure.get(str(event.get("exposure_id"))) or all_items
        match = next(
            (
                item
                for item in items
                if str(item.get("title") or "").casefold() == title
                and str(item.get("artist") or "").casefold() == artist
            ),
            None,
        )
        if not match:
            continue
        matched += 1
        sign = 1.0 if event.get("event_type") in POSITIVE_EVENTS else -0.6 if event.get("event_type") in NEGATIVE_EVENTS else 0.0
        if sign > 0:
            positives += 1
        elif sign < 0:
            negatives += 1
        for key, field in (
            ("semantic", "semantic_score"),
            ("acoustic", "acoustic_score"),
            ("personal", "personal_score"),
        ):
            value = match.get(field)
            if value is None:
                continue
            score = max(0.0, min(1.0, float(value)))
            accum[key] += sign * (score - 0.5)

    cleaned = {key: max(0.05, value) for key, value in accum.items()}
    total = sum(cleaned.values())
    weights = {key: round(value / total, 4) for key, value in cleaned.items()}
    return {
        "weights": weights,
        "matched_events": matched,
        "positive_events": positives,
        "negative_events": negatives,
        "method": "transparent_event_correlation_v1",
    }
