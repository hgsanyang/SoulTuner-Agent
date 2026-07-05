"""Catalog gap detection for deciding controlled online discovery.

The detector answers one question: can the local catalog *prove* it satisfies
the user's constraints?  If not, the graph can either mix in a few online
candidates, fall back to online discovery, or explain the gap when web is off.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Mapping, Sequence

from agent.retrieval_fallback import FallbackDecision, _canonical_language, layered_constraints


@dataclass(frozen=True)
class CatalogGapDecision:
    action: str = "none"  # none | mix_in | fallback | blocked
    reasons: tuple[str, ...] = ()
    inventory_count: int = 0
    target_web_count: int = 0
    discovery_required: bool = False
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def needs_online(self) -> bool:
        return self.action in {"mix_in", "fallback"}

    def model_dump(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "reasons": list(self.reasons),
            "inventory_count": self.inventory_count,
            "target_web_count": self.target_web_count,
            "discovery_required": self.discovery_required,
            "message": self.message,
            "details": self.details,
        }


_ERA_PATTERNS = (
    r"(?:19[2-9]0|20[0-2]0)年代",
    r"\b(?:[1-9]0)s\b",
    r"(?:八十|80|九十|90|七十|70|六十|60)年代",
    r"(?:老歌|怀旧|复古|oldies|classic old songs|retro)",
)
_RELEASE_PATTERNS = (
    r"(?:发行|首发|原唱|原版|哪一年|年份|年代|release year|original release)",
)
_RECENCY_PATTERNS = (
    r"(?:最新|最近|今年|本周|本月|刚出|刚发|新歌|新曲|新专|2026|latest|new release)",
)
_EXTERNAL_KNOWLEDGE_PATTERNS = (
    r"(?:榜单|排名|获奖|新闻|资讯|巡演|演唱会|代表作|口碑|playlist|chart|award|news)",
    r"(?:创作背景|歌曲背景|歌手背景|背景故事|background info|background story)",
)


def _song_dict(item: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = item.get("song")
    return nested if isinstance(nested, Mapping) else item


def _iter_terms(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value is None:
        return []
    text = str(value).strip()
    return [text] if text else []


def _context_text(query: str, retrieval_plan: Mapping[str, Any] | None) -> str:
    plan = dict(retrieval_plan or {})
    hard = dict(plan.get("hard_constraints") or {})
    soft = dict(plan.get("soft_intent") or {})
    hints = dict(plan.get("hints") or {})
    parts = [
        query,
        str(plan.get("vector_acoustic_query") or ""),
        str(plan.get("web_search_keywords") or ""),
        *(_iter_terms(soft.get("goal"))),
        *(_iter_terms(soft.get("trajectory"))),
        *(_iter_terms(soft.get("vibe"))),
        *(_iter_terms(soft.get("avoid"))),
        *(_iter_terms(hints.get("mood"))),
        *(_iter_terms(hints.get("scenario"))),
        *(_iter_terms(hints.get("genres"))),
        *(_iter_terms(hard.get("language"))),
        *(_iter_terms(hard.get("region"))),
    ]
    return " ".join(part for part in parts if part).casefold()


def _matches_any(text: str, patterns: Sequence[str]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def _metadata_value(song: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = song.get(key)
        if value not in (None, "", "Unknown", "unknown", "未知", "未标注"):
            return value
    return None


def _metadata_coverage(search_results: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    if not search_results:
        return {"release": 0.0, "playable": 0.0, "language": 0.0}
    release_known = 0
    playable = 0
    language_known = 0
    for item in search_results:
        song = _song_dict(item)
        if _metadata_value(song, "release_year", "release_date", "year", "publishTime", "publish_time"):
            release_known += 1
        if _metadata_value(song, "preview_url", "audio_url", "play_url"):
            playable += 1
        language = _metadata_value(song, "language")
        if language and _canonical_language(language):
            language_known += 1
    total = len(search_results)
    return {
        "release": round(release_known / total, 4),
        "playable": round(playable / total, 4),
        "language": round(language_known / total, 4),
    }


def _message_for_reasons(reasons: Sequence[str]) -> str:
    reason_text = "、".join(reasons) if reasons else "本地曲库证据不足"
    return (
        f"我现在只在本地曲库里找，但这次需求涉及 {reason_text}，"
        "本地结果不足以可靠满足。打开联网搜索后，我可以继续查找外部候选和可播放音频。"
    )


def analyze_catalog_gap(
    search_results: Sequence[Mapping[str, Any]],
    retrieval_plan: Mapping[str, Any] | None,
    query: str = "",
    *,
    web_enabled: bool,
    fallback_decision: FallbackDecision | None = None,
    normal_mix_count: int = 4,
    fallback_count: int = 10,
    min_local_results: int = 8,
) -> CatalogGapDecision:
    """Return a generic online-discovery decision for a local recommendation slate."""
    plan = dict(retrieval_plan or {})
    text = _context_text(query, plan)
    artists, songs, hard = layered_constraints(plan)
    count = len(search_results or [])
    coverage = _metadata_coverage(search_results)
    reasons: list[str] = []
    discovery_required = False

    explicit_fallback = fallback_decision if fallback_decision is not None else FallbackDecision(False, "", count)
    if explicit_fallback.required:
        reasons.append(explicit_fallback.reason or "explicit_constraint_gap")

    requires_release = _matches_any(text, _ERA_PATTERNS) or _matches_any(text, _RELEASE_PATTERNS)
    requires_recency = _matches_any(text, _RECENCY_PATTERNS)
    requires_external = _matches_any(text, _EXTERNAL_KNOWLEDGE_PATTERNS)
    if requires_release:
        discovery_required = True
        if coverage["release"] < 0.5:
            reasons.append("metadata_release_year_missing")
    if requires_recency:
        discovery_required = True
        reasons.append("recency_required")
    if requires_external:
        discovery_required = True
        reasons.append("external_knowledge_required")

    if count < min_local_results and (artists or songs or hard or requires_release or requires_recency):
        reasons.append("local_inventory_low")
    if count and coverage["playable"] < 0.5:
        reasons.append("playable_gap")

    reasons = list(dict.fromkeys(reason for reason in reasons if reason))
    strict_gap = bool(reasons)
    if strict_gap and not web_enabled:
        return CatalogGapDecision(
            action="blocked",
            reasons=tuple(reasons),
            inventory_count=count,
            target_web_count=0,
            discovery_required=discovery_required,
            message=_message_for_reasons(reasons),
            details={"coverage": coverage},
        )
    if strict_gap:
        return CatalogGapDecision(
            action="fallback",
            reasons=tuple(reasons),
            inventory_count=count,
            target_web_count=max(1, int(fallback_count)),
            discovery_required=discovery_required,
            details={"coverage": coverage},
        )

    exact_song_request = bool(songs) and not re.search(r"类似|相似|听感|same vibe|sounds like|like this", text)
    if web_enabled and count >= min_local_results and not exact_song_request:
        return CatalogGapDecision(
            action="mix_in",
            reasons=("online_exploration",),
            inventory_count=count,
            target_web_count=max(1, int(normal_mix_count)),
            discovery_required=requires_external or requires_release or requires_recency,
            details={"coverage": coverage},
        )

    return CatalogGapDecision(
        action="none",
        reasons=(),
        inventory_count=count,
        target_web_count=0,
        discovery_required=False,
        details={"coverage": coverage},
    )


def interleave_online_results(
    local_items: Sequence[Mapping[str, Any]],
    online_items: Sequence[Mapping[str, Any]],
    *,
    target_len: int | None = None,
    first_slot: int = 3,
    stride: int = 4,
) -> list[dict[str, Any]]:
    """Interleave online rows into local rows without increasing stale duplicates."""
    from retrieval.retrieval_fusion import normalize_song_key

    target = target_len if target_len is not None else len(local_items)
    target = max(0, int(target))
    seen: set[str] = set()

    def key_for(item: Mapping[str, Any]) -> str:
        song = _song_dict(item)
        return normalize_song_key(str(song.get("title") or ""), str(song.get("artist") or ""))

    online_queue = []
    for item in online_items:
        key = key_for(item)
        if key and key not in seen:
            online_queue.append(dict(item))
            seen.add(key)

    result: list[dict[str, Any]] = []
    online_index = 0
    for local_index, item in enumerate(local_items):
        should_insert = (
            online_index < len(online_queue)
            and len(result) >= first_slot
            and (len(result) - first_slot) % stride == 0
        )
        if should_insert:
            result.append(online_queue[online_index])
            online_index += 1
            if target and len(result) >= target:
                return result[:target]
        key = key_for(item)
        if key and key in {key_for(existing) for existing in result}:
            continue
        result.append(dict(item))
        if target and len(result) >= target:
            return result[:target]

    while online_index < len(online_queue) and (not target or len(result) < target):
        result.append(online_queue[online_index])
        online_index += 1
    return result[:target] if target else result


def unwrap_recommendation_items(recommendations: Any) -> list[Any]:
    """Return recommendation rows from either a ToolOutput-like object or a raw list."""
    items = getattr(recommendations, "data", recommendations)
    return list(items) if isinstance(items, list) else []
