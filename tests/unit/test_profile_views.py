"""场景化 profile views 的分组与可编辑性测试。"""

from services.memory_models import MemoryLayer, MemoryRecord
from services.profile_views import build_profile_views, normalize_scope

DAY_MS = 86_400_000
NOW = 1_800_000_000_000


def _record(
    record_id: str,
    *,
    layer: MemoryLayer = MemoryLayer.INFERRED,
    scope: str = "global",
    field: str = "add_moods",
    value: str = "Calm",
    kind: str = "preference",
    created_at: int = NOW - DAY_MS,
    confidence: float = 0.85,
) -> MemoryRecord:
    return MemoryRecord(
        record_id=record_id,
        user_id="u1",
        layer=layer,
        kind=kind,
        source="memory_consolidator",
        evidence_id="e1",
        confidence=confidence,
        created_at=created_at,
        valid_from=created_at,
        expires_at=NOW + 30 * DAY_MS,
        payload={
            "field": field,
            "value": value,
            "scope": scope,
            "evidence_ids": ["e1", "e2"],
            "decision_summary": "supported by repeated evidence",
        },
    )


def test_views_group_by_free_scene_labels_global_first_lifecycle_last():
    views = build_profile_views(
        [
            _record("m1", scope="夜里一个人开车", value="Energetic"),
            _record("m1b", scope="夜里一个人开车", value="Bass Heavy"),
            _record("m2", scope="global", value="Calm", layer=MemoryLayer.EXPLICIT),
            _record("m3", scope="雨天在家", value="Quiet"),
            _record("m4", scope="contextual", value="Warm"),
        ],
        now_ms=NOW,
    )
    scopes = [view["scope"] for view in views["views"]]
    # global 最前，场景标签按条数降序，生命周期兜底最后
    assert scopes == ["global", "夜里一个人开车", "雨天在家", "contextual"]
    driving = next(view for view in views["views"] if view["scope"] == "夜里一个人开车")
    assert driving["title"] == "夜里一个人开车"  # 场景视图标题就是 LLM 起的标签
    assert len(driving["items"]) == 2


def test_view_items_carry_evidence_validity_and_editability():
    views = build_profile_views([_record("m1", scope="雨天独处", value="Melancholy")], now_ms=NOW)
    item = views["views"][0]["items"][0]
    assert item["record_id"] == "m1"
    assert item["layer"] == "L2"
    assert item["evidence_count"] == 3  # evidence_ids x2 + evidence_id
    assert item["expires_at"] == NOW + 30 * DAY_MS
    assert item["decision_summary"]
    assert item["editable"] is True


def test_empty_scope_folds_into_global_and_non_preferences_excluded():
    views = build_profile_views(
        [
            _record("m1", scope=""),
            _record("m2", kind="tombstone"),
            _record("m3", layer=MemoryLayer.RAW_EVENT),
        ],
        now_ms=NOW,
    )
    assert [view["scope"] for view in views["views"]] == ["global"]
    assert len(views["views"][0]["items"]) == 1


def test_recent_tendency_only_includes_fresh_inferred():
    views = build_profile_views(
        [
            _record("m-fresh", created_at=NOW - 3 * DAY_MS),
            _record("m-old", created_at=NOW - 40 * DAY_MS),
            _record("m-explicit", layer=MemoryLayer.EXPLICIT, created_at=NOW - DAY_MS),
        ],
        now_ms=NOW,
    )
    recent_ids = [item["record_id"] for item in views["recent_tendency"]["items"]]
    assert recent_ids == ["m-fresh"]


def test_normalize_scope_passes_free_labels_through():
    assert normalize_scope("夜里一个人开车") == "夜里一个人开车"
    assert normalize_scope("temporary") == "temporary"
    assert normalize_scope("") == "global"
    assert normalize_scope(None) == "global"
