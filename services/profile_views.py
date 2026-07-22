"""Materialized, scope-grouped profile views over the memory ledger.

The product requirement (multi-scene preferences) is served by small
per-scope views instead of one monolithic profile text. Scene labels are
free-form and authored by the consolidation LLM from the user's own
evidence — this module never maps them onto a fixed scene vocabulary; it
only groups and formats effective ledger records. Lifecycle scopes
(global/contextual/temporary) keep stable Chinese titles; every scene
label becomes its own view titled by the label itself, so the scene space
can grow and be renamed across consolidation passes.
"""

from __future__ import annotations

import time
from typing import Any

from services.memory_models import MemoryLayer, MemoryRecord

LIFECYCLE_TITLES: dict[str, str] = {
    "global": "长期稳定偏好",
    "contextual": "情境偏好",
    "temporary": "临时偏好",
}

RECENT_TENDENCY_WINDOW_DAYS = 14


def normalize_scope(raw: Any) -> str:
    """Free-form scene labels pass through; empty falls back to global."""
    return str(raw or "").strip() or "global"


def _view_item(record: MemoryRecord) -> dict[str, Any]:
    payload = record.payload
    evidence_ids = payload.get("evidence_ids") or []
    if not isinstance(evidence_ids, list):
        evidence_ids = []
    return {
        "record_id": record.record_id,
        "memory_key": record.memory_key,
        "layer": record.layer.value,
        "field": str(payload.get("field") or ""),
        "value": str(payload.get("value") or ""),
        "confidence": round(float(record.confidence), 4),
        "source": record.source,
        "valid_from": record.valid_from,
        "expires_at": record.expires_at,
        "evidence_count": len(evidence_ids) + (1 if record.evidence_id else 0),
        "decision_summary": str(payload.get("decision_summary") or ""),
        "why_used": record.why_used,
        "editable": True,
    }


def build_profile_views(
    records: list[MemoryRecord],
    *,
    now_ms: int | None = None,
) -> dict[str, Any]:
    """Group effective L1/L2 records into per-scope editable views."""
    now = int(now_ms if now_ms is not None else time.time() * 1000)
    grouped: dict[str, list[dict[str, Any]]] = {}
    display_labels: dict[str, str] = {}
    recent_tendency: list[dict[str, Any]] = []
    recent_cutoff = now - RECENT_TENDENCY_WINDOW_DAYS * 86_400_000

    for record in records:
        if record.layer not in {MemoryLayer.EXPLICIT, MemoryLayer.INFERRED}:
            continue
        if record.kind != "preference":
            continue
        item = _view_item(record)
        label = normalize_scope(record.payload.get("scope"))
        key = label.casefold()
        grouped.setdefault(key, []).append(item)
        display_labels.setdefault(key, label)
        if record.layer == MemoryLayer.INFERRED and record.created_at >= recent_cutoff:
            recent_tendency.append(item)

    # global first, then scene labels by size (the user's most-lived scenes
    # surface first), lifecycle catch-alls last.
    def _order(key: str) -> tuple[int, int, str]:
        if key == "global":
            return (0, 0, key)
        if key in LIFECYCLE_TITLES:
            return (2, 0, key)
        return (1, -len(grouped[key]), key)

    views: list[dict[str, Any]] = []
    for key in sorted(grouped, key=_order):
        items = grouped[key]
        items.sort(key=lambda it: (it["layer"], -it["confidence"]))
        label = display_labels[key]
        views.append(
            {
                "scope": label,
                "title": LIFECYCLE_TITLES.get(key, label),
                "items": items,
            }
        )

    return {
        "views": views,
        "recent_tendency": {
            "title": f"最近 {RECENT_TENDENCY_WINDOW_DAYS} 天推断倾向",
            "items": sorted(recent_tendency, key=lambda it: -it["confidence"]),
        },
        "generated_at_ms": now,
        "scope_order": [view["scope"] for view in views],
    }
