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


def supersede_mix_in(decision: CatalogGapDecision, *, superseded_by: str) -> CatalogGapDecision:
    """Downgrade a mix_in decision when another lane already provides web songs.

    Only mix_in (inventory is sufficient, web songs are optional garnish) is
    superseded; a true fallback (local inventory cannot satisfy the request)
    keeps its online discovery untouched.
    """
    if decision.action != "mix_in":
        return decision
    from dataclasses import replace

    return replace(
        decision,
        action="none",
        details={**(decision.details or {}), "mix_in_superseded_by": superseded_by},
    )


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
_TAG_FIELD_KEYS = {
    "genres": ("genres", "genre"),
    "moods": ("moods", "mood"),
    "scenarios": ("scenarios", "scenario"),
}
_TAG_ALIASES = {
    "r&b": {"r&b", "rnb", "rhythmblues"},
    "hiphop": {"hiphop", "rap"},
    "hip-hop": {"hiphop", "rap"},
    "lofi": {"lofi"},
    "lo-fi": {"lofi"},
    "calm": {"calm", "peaceful", "relaxing", "quiet", "soft", "gentle"},
    "平静": {"calm", "peaceful", "relaxing", "quiet", "soft", "gentle"},
    "安静": {"calm", "peaceful", "relaxing", "quiet", "soft", "gentle"},
    "柔软": {"soft", "gentle", "peaceful", "relaxing"},
    "rainyday": {"rainyday", "rainy", "rain"},
    "rainy day": {"rainyday", "rainy", "rain"},
    "雨天": {"rainyday", "rainy", "rain"},
    "study": {"study", "focus", "work", "reading"},
    "学习": {"study", "focus", "work", "reading"},
    "专注": {"study", "focus", "work", "reading"},
    "sleep": {"sleep", "latenight", "relaxing"},
    "睡眠": {"sleep", "latenight", "relaxing"},
}
_QUERY_TAG_TERMS = {
    "genres": {
        "r&b": "R&B",
        "rnb": "R&B",
        "节奏布鲁斯": "R&B",
        "hip hop": "Hip-Hop",
        "hip-hop": "Hip-Hop",
        "说唱": "Hip-Hop",
        "rap": "Hip-Hop",
        "lo-fi": "Lo-Fi",
        "lofi": "Lo-Fi",
        "摇滚": "Rock",
        "民谣": "Folk",
        "电子": "Electronic",
        "电音": "Electronic",
        "edm": "EDM",
        "爵士": "Jazz",
        "后摇": "Post-Rock",
        "独立": "Indie",
    },
    "moods": {
        "安静": "Calm",
        "柔软": "Soft",
        "温柔": "Gentle",
        "平静": "Peaceful",
        "治愈": "Healing",
        "难过": "Melancholy",
        "伤感": "Melancholy",
        "怀旧": "Nostalgic",
        "复古": "Nostalgic",
        "quiet": "Calm",
        "soft": "Soft",
        "gentle": "Gentle",
        "calm": "Calm",
        "nostalgic": "Nostalgic",
    },
    "scenarios": {
        "雨天": "Rainy Day",
        "下雨": "Rainy Day",
        "专注": "Study",
        "学习": "Study",
        "写代码": "Work",
        "睡前": "Sleep",
        "夜里": "Late Night",
        "深夜": "Late Night",
        "开车": "Driving",
        "通勤": "Commute",
        "rainy": "Rainy Day",
        "study": "Study",
        "focus": "Study",
        "sleep": "Sleep",
        "late night": "Late Night",
        "driving": "Driving",
    },
}
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


def _normalize_tag(value: Any) -> str:
    return re.sub(r"[\W_]+", "", str(value or "").casefold())


def _tag_aliases(value: Any) -> set[str]:
    text = str(value or "").strip()
    if not text:
        return set()
    compact = _normalize_tag(text)
    aliases = {compact}
    aliases.update(_normalize_tag(item) for item in _TAG_ALIASES.get(text.casefold(), set()))
    aliases.update(_normalize_tag(item) for item in _TAG_ALIASES.get(compact, set()))
    return {item for item in aliases if item}


def _iter_label_values(song: Mapping[str, Any], keys: Sequence[str]) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for key in keys:
        raw = song.get(key)
        raw_values = raw if isinstance(raw, list) else [raw]
        for item in raw_values:
            for piece in re.split(r"[/,，;；|｜]", str(item or "")):
                text = piece.strip()
                normalized = _normalize_tag(text)
                if normalized and normalized not in {"unknown", "none", "null", "未知", "未标注"} and normalized not in seen:
                    seen.add(normalized)
                    values.append(text)
    return values


def _tag_matches(requested: Any, labels: Sequence[str]) -> bool:
    requested_aliases = _tag_aliases(requested)
    if not requested_aliases:
        return False
    for label in labels:
        label_aliases = _tag_aliases(label)
        if requested_aliases & label_aliases:
            return True
        if any(req and lab and (req in lab or lab in req) for req in requested_aliases for lab in label_aliases):
            return True
    return False


def _dedupe_terms(terms: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for term in terms:
        text = str(term or "").strip()
        key = text.casefold()
        if text and key not in seen:
            seen.add(key)
            result.append(text)
    return result


def _requested_tag_terms(retrieval_plan: Mapping[str, Any] | None) -> dict[str, list[str]]:
    """Return only LLM-planned tag hints; this is not a second intent parser."""
    plan = dict(retrieval_plan or {})
    hints = dict(plan.get("hints") or {})
    return {
        "genres": _dedupe_terms(_iter_terms(hints.get("genres"))),
        "moods": _dedupe_terms(_iter_terms(hints.get("mood"))),
        "scenarios": _dedupe_terms(_iter_terms(hints.get("scenario"))),
    }


def _tag_evidence(
    search_results: Sequence[Mapping[str, Any]],
    requested_terms: Mapping[str, Sequence[str]],
) -> dict[str, dict[str, Any]]:
    evidence: dict[str, dict[str, Any]] = {}
    total = len(search_results)
    for tag_field, terms in requested_terms.items():
        terms = [str(term).strip() for term in terms if str(term).strip()]
        if not terms:
            continue
        keys = _TAG_FIELD_KEYS[tag_field]
        known = 0
        matched = 0
        for item in search_results:
            labels = _iter_label_values(_song_dict(item), keys)
            if labels:
                known += 1
            if labels and any(_tag_matches(term, labels) for term in terms):
                matched += 1
        evidence[tag_field] = {
            "requested": terms,
            "known": known,
            "matched": matched,
            "total": total,
            "coverage": round(known / total, 4) if total else 0.0,
            "match_ratio": round(matched / known, 4) if known else 0.0,
        }
    return evidence


def _soft_inventory_gap_reasons(
    tag_evidence: Mapping[str, Mapping[str, Any]],
    *,
    min_local_results: int,
) -> list[str]:
    reasons: list[str] = []
    for tag_field, stats in tag_evidence.items():
        total = int(stats.get("total") or 0)
        known = int(stats.get("known") or 0)
        matched = int(stats.get("matched") or 0)
        if total < min_local_results:
            continue
        # Only call a soft catalog gap when the slate has enough usable labels
        # to make the absence meaningful. Sparse labels should not masquerade as
        # proof that the catalog lacks a style.
        enough_label_evidence = known >= max(4, int(total * 0.4))
        if enough_label_evidence and matched == 0:
            reasons.append(f"local_{tag_field}_match_insufficient")
    return reasons


def _requested_language(hard: Mapping[str, Any]) -> str:
    hard_language = _canonical_language(hard.get("language"))
    if hard_language:
        return hard_language
    return ""


def _language_evidence(search_results: Sequence[Mapping[str, Any]], requested_language: str) -> dict[str, Any]:
    requested = _canonical_language(requested_language)
    total = len(search_results)
    if not requested:
        return {"requested": "", "known": 0, "matched": 0, "total": total, "coverage": 0.0, "match_ratio": 0.0}
    known = 0
    matched = 0
    for item in search_results:
        song = _song_dict(item)
        actual = _canonical_language(song.get("language"))
        if actual:
            known += 1
            if actual == requested:
                matched += 1
    return {
        "requested": requested,
        "known": known,
        "matched": matched,
        "total": total,
        "coverage": round(known / total, 4) if total else 0.0,
        "match_ratio": round(matched / known, 4) if known else 0.0,
    }


def _language_gap_reason(
    evidence: Mapping[str, Any],
    *,
    min_local_results: int,
) -> str:
    requested = str(evidence.get("requested") or "")
    total = int(evidence.get("total") or 0)
    known = int(evidence.get("known") or 0)
    matched = int(evidence.get("matched") or 0)
    if not requested or total < min_local_results:
        return ""
    enough_language_evidence = known >= max(4, int(total * 0.4))
    if enough_language_evidence and matched == 0:
        return "local_language_match_insufficient"
    return ""


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


def _metadata_constraints(retrieval_plan: Mapping[str, Any] | None) -> dict[str, Any]:
    plan = dict(retrieval_plan or {})
    metadata = dict(plan.get("metadata_constraints") or {})
    # Be forgiving for older locally generated plans while keeping the source
    # of truth in the LLM plan, not raw query phrase matching.
    for legacy_key in (
        "release_year_from",
        "release_year_to",
        "era",
        "recency_required",
        "external_knowledge_required",
    ):
        if legacy_key not in metadata and legacy_key in plan:
            metadata[legacy_key] = plan.get(legacy_key)
    return metadata


def _has_release_constraint(metadata: Mapping[str, Any]) -> bool:
    return bool(
        metadata.get("era")
        or metadata.get("release_year_from") is not None
        or metadata.get("release_year_to") is not None
        or metadata.get("release_year_required")
    )


def _knowledge_evidence(text: str, search_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Read local offline knowledge cards as gap evidence without networking."""

    try:
        from config.settings import settings
        from services.music_knowledge_store import MusicKnowledgeStore

        if not settings.knowledge_gap_enabled:
            return {"enabled": False}
        store = MusicKnowledgeStore()
        if not store.path.exists():
            return {"enabled": True, "available": False, "query_hits": 0, "local_song_hits": 0}
        min_conf = float(settings.knowledge_gap_min_confidence)
        query_hits = store.search(text, limit=5, min_confidence=min_conf)
        query_vector_hits = sum(1 for card in query_hits if card.get("_vector_score") is not None)
        local_hits = []
        local_artist_hits = []
        seen_artists: set[str] = set()
        release_hits = 0
        for item in list(search_results or [])[:20]:
            song = _song_dict(item)
            title = str(song.get("title") or "").strip()
            artist = str(song.get("artist") or "").strip()
            if not title:
                continue
            card = store.get_song_card(title, artist)
            if card and float(card.get("confidence") or 0) >= min_conf:
                local_hits.append(card)
                if card.get("release_year"):
                    release_hits += 1
            if artist and artist.casefold() not in seen_artists:
                seen_artists.add(artist.casefold())
                artist_card = store.get_artist_card(artist)
                if artist_card and float(artist_card.get("confidence") or 0) >= min_conf:
                    local_artist_hits.append(artist_card)
        return {
            "enabled": True,
            "available": True,
            "query_hits": len(query_hits),
            "query_vector_hits": query_vector_hits,
            "local_song_hits": len(local_hits),
            "local_artist_hits": len(local_artist_hits),
            "local_song_release_year_hits": release_hits,
            "cards": [
                {
                    "kind": card.get("kind"),
                    "title": card.get("title"),
                    "artist": card.get("artist"),
                    "confidence": card.get("confidence"),
                    "release_year": card.get("release_year"),
                    "source_url": card.get("source_url"),
                    "retrieval_source": "qdrant" if card.get("_vector_score") is not None else "sqlite",
                    "vector_score": card.get("_vector_score"),
                    "style_tags": card.get("style_tags", []),
                }
                for card in (query_hits + local_hits + local_artist_hits)[:6]
            ],
        }
    except Exception as exc:
        return {"enabled": True, "available": False, "error": str(exc)[:160], "query_hits": 0, "local_song_hits": 0}


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
    metadata_constraints = _metadata_constraints(plan)
    tag_evidence = _tag_evidence(search_results, _requested_tag_terms(plan))
    language_evidence = _language_evidence(search_results, _requested_language(hard))
    knowledge_evidence = _knowledge_evidence(text, search_results)
    reasons: list[str] = []
    soft_reasons: list[str] = []
    discovery_required = False

    explicit_fallback = fallback_decision if fallback_decision is not None else FallbackDecision(False, "", count)
    if explicit_fallback.required:
        reasons.append(explicit_fallback.reason or "explicit_constraint_gap")

    requires_release = _has_release_constraint(metadata_constraints)
    requires_recency = bool(metadata_constraints.get("recency_required"))
    requires_external = bool(metadata_constraints.get("external_knowledge_required"))
    if requires_release:
        discovery_required = True
        local_release_with_cards = (
            count > 0
            and int(knowledge_evidence.get("local_song_release_year_hits") or 0) >= max(2, int(count * 0.35))
        )
        if coverage["release"] < 0.5 and not local_release_with_cards:
            reasons.append("metadata_release_year_missing")
    if requires_recency:
        discovery_required = True
        reasons.append("recency_required")
    if requires_external:
        discovery_required = True
        local_knowledge_hits = (
            int(knowledge_evidence.get("query_hits") or 0)
            + int(knowledge_evidence.get("local_song_hits") or 0)
            + int(knowledge_evidence.get("local_artist_hits") or 0)
        )
        if local_knowledge_hits <= 0:
            reasons.append("external_knowledge_required")

    if count < min_local_results and (artists or songs or hard or requires_release or requires_recency):
        reasons.append("local_inventory_low")
    if count and coverage["playable"] < 0.5:
        reasons.append("playable_gap")
    language_reason = _language_gap_reason(language_evidence, min_local_results=min_local_results)
    if language_reason:
        reasons.append(language_reason)
    soft_reasons.extend(
        _soft_inventory_gap_reasons(
            tag_evidence,
            min_local_results=min_local_results,
        )
    )

    reasons = list(dict.fromkeys(reason for reason in reasons if reason))
    soft_reasons = list(dict.fromkeys(reason for reason in soft_reasons if reason))
    strict_gap = bool(reasons)
    soft_gap = bool(soft_reasons)
    details = {
        "coverage": coverage,
        "tag_evidence": tag_evidence,
        "language_evidence": language_evidence,
        "knowledge_evidence": knowledge_evidence,
        "metadata_constraints": metadata_constraints,
    }
    if strict_gap and not web_enabled:
        return CatalogGapDecision(
            action="blocked",
            reasons=tuple(reasons),
            inventory_count=count,
            target_web_count=0,
            discovery_required=discovery_required,
            message=_message_for_reasons(reasons),
            details=details,
        )
    if strict_gap:
        return CatalogGapDecision(
            action="fallback",
            reasons=tuple(reasons),
            inventory_count=count,
            target_web_count=max(1, int(fallback_count)),
            discovery_required=discovery_required,
            details=details,
        )
    if soft_gap and not web_enabled:
        return CatalogGapDecision(
            action="blocked",
            reasons=tuple(soft_reasons),
            inventory_count=count,
            target_web_count=0,
            discovery_required=False,
            message=_message_for_reasons(soft_reasons),
            details=details,
        )
    if soft_gap:
        return CatalogGapDecision(
            action="mix_in",
            reasons=tuple(soft_reasons),
            inventory_count=count,
            target_web_count=max(normal_mix_count, min(int(fallback_count), 6)),
            discovery_required=False,
            details=details,
        )

    soft = dict(plan.get("soft_intent") or {})
    hints = dict(plan.get("hints") or {})
    exact_song_request = bool(songs) and not any(
        [
            soft.get("goal"),
            soft.get("trajectory"),
            soft.get("vibe"),
            soft.get("avoid"),
            hints.get("genres"),
            hints.get("mood"),
            hints.get("scenario"),
            plan.get("vector_acoustic_query"),
            plan.get("vector_acoustic_queries"),
        ]
    )
    if web_enabled and count >= min_local_results and not exact_song_request:
        return CatalogGapDecision(
            action="mix_in",
            reasons=("online_exploration",),
            inventory_count=count,
            target_web_count=max(1, int(normal_mix_count)),
            discovery_required=requires_external or requires_release or requires_recency,
            details=details,
        )

    return CatalogGapDecision(
        action="none",
        reasons=(),
        inventory_count=count,
        target_web_count=0,
        discovery_required=False,
        details=details,
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
