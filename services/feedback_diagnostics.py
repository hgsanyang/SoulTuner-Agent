"""Non-sensitive diagnostics for the A3 feedback learning loop."""

from __future__ import annotations

from collections import Counter
from typing import Any

from services.ranking_learning import (
    build_preference_pairs,
    build_slate_feedback_rows,
    build_strict_labeled_rows,
)


def summarize_feedback_quality(
    exposures: list[dict[str, Any]],
    events: list[dict[str, Any]],
    slate_feedback: list[dict[str, Any]],
    *,
    min_events: int = 20,
    slate_top_k: int = 5,
) -> dict[str, Any]:
    """Return privacy-preserving readiness diagnostics for replay learning."""
    explicit_rows, explicit_diag = build_strict_labeled_rows(exposures, events)
    slate_rows, slate_diag = build_slate_feedback_rows(
        exposures,
        slate_feedback,
        top_k=slate_top_k,
    )
    rows = explicit_rows + slate_rows
    label_counts = Counter(int(row["label"]) for row in rows)
    event_types = Counter(str(row.get("event_type") or "") for row in events)
    slate_ratings = Counter(str(row.get("rating") or "") for row in slate_feedback)
    exposure_items = sum(len(row.get("items") or []) for row in exposures)
    pair_count = len(build_preference_pairs(rows))
    blockers: list[str] = []

    if not exposures:
        blockers.append("no_exposures")
    if len(rows) < max(1, int(min_events)):
        blockers.append("insufficient_labeled_rows")
    if label_counts.get(1, 0) == 0:
        blockers.append("missing_positive_labels")
    if label_counts.get(0, 0) == 0:
        blockers.append("missing_negative_labels")
    if pair_count == 0:
        blockers.append("no_same_exposure_preference_pairs")

    return {
        "num_exposures": len(exposures),
        "num_exposed_items": exposure_items,
        "num_events": len(events),
        "num_slate_feedback": len(slate_feedback),
        "matched_explicit_rows": len(explicit_rows),
        "matched_slate_rows": len(slate_rows),
        "training_rows": len(rows),
        "positive_rows": label_counts.get(1, 0),
        "negative_rows": label_counts.get(0, 0),
        "same_exposure_pairs": pair_count,
        "event_type_counts": dict(event_types),
        "slate_rating_counts": dict(slate_ratings),
        "explicit_diagnostics": explicit_diag,
        "slate_diagnostics": slate_diag,
        "blockers": blockers,
        "ready_for_replay": not blockers,
    }
