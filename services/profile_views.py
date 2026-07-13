"""Materialized, scope-grouped profile views over the memory ledger.

The product requirement (multi-scene preferences) is served by small
per-scope views instead of one monolithic profile text: global stable
preferences, per-scene preferences (driving/focus/sleep/...), and a recent
tendency view for short-lived inferred preferences. Every item stays
editable and evidence-bound; this module only groups and formats effective
ledger records — it never invents preferences.
"""

from __future__ import annotations

import time
from typing import Any

from services.memory_models import MemoryLayer, MemoryRecord
from services.memory_retriever import LIFECYCLE_SCOPES, SCENE_SCOPES

# Presentation order and Chinese labels for the UI.
SCOPE_VIEW_ORDER: list[tuple[str, str]] = [
    ("global", "长期稳定偏好"),
    ("driving", "开车"),
    ("commute", "通勤"),
    ("focus", "专注/工作"),
    ("sleep", "睡眠"),
    ("late_night", "深夜"),
    ("rainy", "雨天"),
    ("romantic", "约会/浪漫"),
    ("workout", "运动"),
    ("contextual", "情境偏好"),
    ("temporary", "临时偏好"),
]

RECENT_TENDENCY_WINDOW_DAYS = 14


def normalize_scope(raw: Any) -> str:
    scope = str(raw or "global").strip().casefold()
    if scope in SCENE_SCOPES or scope in LIFECYCLE_SCOPES:
        return scope
    return "global"


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
    recent_tendency: list[dict[str, Any]] = []
    recent_cutoff = now - RECENT_TENDENCY_WINDOW_DAYS * 86_400_000

    for record in records:
        if record.layer not in {MemoryLayer.EXPLICIT, MemoryLayer.INFERRED}:
            continue
        if record.kind != "preference":
            continue
        item = _view_item(record)
        scope = normalize_scope(record.payload.get("scope"))
        grouped.setdefault(scope, []).append(item)
        if record.layer == MemoryLayer.INFERRED and record.created_at >= recent_cutoff:
            recent_tendency.append(item)

    views: list[dict[str, Any]] = []
    for scope, title in SCOPE_VIEW_ORDER:
        items = grouped.get(scope) or []
        if not items:
            continue
        items.sort(key=lambda it: (it["layer"], -it["confidence"]))
        views.append({"scope": scope, "title": title, "items": items})

    return {
        "views": views,
        "recent_tendency": {
            "title": f"最近 {RECENT_TENDENCY_WINDOW_DAYS} 天推断倾向",
            "items": sorted(recent_tendency, key=lambda it: -it["confidence"]),
        },
        "generated_at_ms": now,
        "scope_order": [scope for scope, _ in SCOPE_VIEW_ORDER],
    }
