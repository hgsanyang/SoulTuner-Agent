"""Local JSONL audit log for LLM planning and ontology feedback.

This module records what the planner decided and which parts may deserve
offline review.  It deliberately does not call another LLM on the hot path and
never mutates the tag ontology automatically.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Mapping
import uuid

from services.runtime_mode import side_effects_disabled


LOG_FILE = "planning_feedback.jsonl"


def feedback_log_enabled() -> bool:
    return not side_effects_disabled() and os.getenv("MUSIC_LLM_FEEDBACK_LOG_ENABLED", "1").lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def feedback_log_dir() -> Path:
    root = os.getenv("MUSIC_LLM_FEEDBACK_DIR") or os.getenv("MUSIC_FEEDBACK_DIR")
    path = Path(root) if root else Path("data") / "feedback" / "llm"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        return dumped if isinstance(dumped, dict) else {}
    return {}


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple | set):
        return list(value)
    if value in ("", "Unknown", "未知"):
        return []
    return [value]


def _compact_text(value: Any, *, limit: int = 600) -> str:
    text = " ".join(str(value or "").split())
    return text[:limit]


def build_planning_feedback(
    *,
    user_input: str,
    plan: Any,
    retrieval_plan: Mapping[str, Any] | None = None,
    provider: str = "",
    model: str = "",
    user_id: str = "",
    dialog_delta: Mapping[str, Any] | None = None,
    refinement_options: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build an audit payload for later ontology and retrieval review."""
    plan_dict = _as_dict(plan)
    rp = dict(retrieval_plan or _as_dict(plan_dict.get("retrieval_plan")))
    hard = _as_dict(rp.get("hard_constraints"))
    soft = _as_dict(rp.get("soft_intent"))
    hints = _as_dict(rp.get("hints"))

    genres = _as_list(hints.get("genres") or rp.get("graph_genre_filter"))
    mood = hints.get("mood") or rp.get("graph_mood_filter")
    scenario = hints.get("scenario") or rp.get("graph_scenario_filter")
    avoid = _as_list(soft.get("avoid"))
    has_soft_text = any(
        _compact_text(soft.get(field))
        for field in ("goal", "trajectory", "vibe")
    )
    has_any_hint = bool(genres or mood or scenario)
    has_entities = bool(_as_list(hard.get("artist_entities")) or _as_list(hard.get("song_entities")))

    observations: list[dict[str, Any]] = []
    missing_information: list[str] = []
    review_suggestions: list[str] = []

    if has_soft_text and not has_any_hint:
        observations.append({
            "type": "soft_intent_without_structured_hints",
            "severity": "medium",
            "detail": "用户需求主要落在自由文本软意图，当前标签体系可能无法完整表达。",
        })
        review_suggestions.append("检查是否需要新增 mood/theme/scenario 标签，或改进现有标签映射。")

    if avoid:
        observations.append({
            "type": "negative_preference_present",
            "severity": "medium",
            "avoid_terms": avoid[:12],
            "detail": "存在否定/避开项，应重点观察结果是否仍混入冲突标签。",
        })
        review_suggestions.append("聚合高频 avoid_terms，决定是否增加负向标签或冲突标签组。")

    if has_entities and str(plan_dict.get("intent_type") or rp.get("_intent_type") or "") == "vector_search":
        observations.append({
            "type": "entity_with_vector_intent",
            "severity": "high",
            "detail": "用户包含实体但意图落到纯向量检索，可能需要人工审查。",
        })

    if has_soft_text and not _compact_text(rp.get("vector_acoustic_query")):
        missing_information.append("vector_acoustic_query")
        review_suggestions.append("软意图存在但声学描述为空，可能影响文搜音召回。")

    if hard.get("language") and not hard.get("region"):
        observations.append({
            "type": "language_without_region",
            "severity": "low",
            "detail": "语言已识别但地区缺失；对粤语/日语/韩语等库存诊断可能需要补 region。",
        })

    if rp.get("use_web_search"):
        observations.append({
            "type": "web_search_requested",
            "severity": "info",
            "detail": "本轮规划需要外部信息或本地库存补充，后续可检查 Catalog Gap 判定是否合理。",
        })

    if not has_entities and not has_any_hint and not has_soft_text:
        missing_information.append("usable_music_intent")
        observations.append({
            "type": "underspecified_music_request",
            "severity": "medium",
            "detail": "缺少实体、标签和软意图，可能更适合澄清反问。",
        })

    tag_feedback = {
        "requested_genres": genres[:8],
        "requested_mood": mood,
        "requested_scenario": scenario,
        "avoid_terms": avoid[:12],
        "needs_tag_review": any(item.get("severity") in {"medium", "high"} for item in observations),
    }

    return {
        "type": "llm_planning_feedback",
        "schema_version": 1,
        "event_id": str(uuid.uuid4()),
        "ts": datetime.now(timezone.utc).isoformat(),
        "provider": provider,
        "model": model,
        "user_id": user_id,
        "user_input": _compact_text(user_input, limit=1000),
        "intent_type": plan_dict.get("intent_type") or rp.get("_intent_type"),
        "reasoning": _compact_text(plan_dict.get("reasoning"), limit=500),
        "context": _compact_text(plan_dict.get("context"), limit=500),
        "retrieval_plan": {
            "hard_constraints": hard,
            "soft_intent": soft,
            "hints": hints,
            "vector_acoustic_query": _compact_text(rp.get("vector_acoustic_query"), limit=1000),
            "use_web_search": bool(rp.get("use_web_search")),
            "web_search_keywords": _compact_text(rp.get("web_search_keywords"), limit=300),
        },
        "dialog_delta": dict(dialog_delta or {}),
        "refinement_options": list(refinement_options or [])[:8],
        "tag_feedback": tag_feedback,
        "missing_information": missing_information,
        "observations": observations,
        "review_suggestions": list(dict.fromkeys(review_suggestions)),
        "decision_policy": {
            "auto_apply_tag_changes": False,
            "hot_path_extra_llm_call": False,
            "purpose": "offline ontology, prompt, and catalog-gap review",
        },
    }


def log_planning_feedback(payload: Mapping[str, Any]) -> Path | None:
    """Append an audit event to local JSONL when enabled."""
    if not feedback_log_enabled():
        return None
    path = feedback_log_dir() / LOG_FILE
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), ensure_ascii=False, default=str) + "\n")
    return path


def load_planning_feedback(path: Path | None = None) -> list[dict[str, Any]]:
    target = path or feedback_log_dir() / LOG_FILE
    if not target.exists():
        return []
    rows = []
    with target.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows
