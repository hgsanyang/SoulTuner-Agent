"""Unified memory gateway for hot-path behavior and optional episodic memory.

The gateway keeps recommendation code away from concrete memory backends.
Neo4j remains the structured hot path; GraphZep is an optional sidecar; Mem0
can be added later behind the same interface.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from retrieval.user_memory import UserMemoryManager
from services.feedback_logger import log_slate_feedback, log_user_event
from services.memory_consolidator import MemoryConsolidator
from services.memory_event_store import MemoryEventStore
from services.memory_models import MemoryLayer
from services.memory_retriever import MemoryRelevanceRetriever
from services.runtime_mode import side_effects_disabled

logger = logging.getLogger(__name__)


EVENT_TEMPLATES = {
    "like": "用户对《{title}》{artist} 点了赞，表示喜欢这首歌",
    "unlike": "用户取消了对《{title}》{artist} 的点赞，可能不再感兴趣",
    "save": "用户收藏了《{title}》{artist}，非常喜欢这首歌",
    "unsave": "用户取消了《{title}》{artist} 的收藏",
    "skip": "用户在播放《{title}》{artist} 时迅速跳过了，可能不喜欢",
    "full_play": "用户完整听完了《{title}》{artist}，表示认可",
    "repeat": "用户反复播放了《{title}》{artist}，非常喜欢这首歌",
    "dislike": "用户明确表示不喜欢《{title}》{artist}",
    "play_start": "用户开始播放《{title}》{artist}",
}

SUPPORTED_USER_EVENTS = set(EVENT_TEMPLATES) | {"unsave"}


@dataclass(frozen=True)
class MemoryWriteResult:
    success: bool
    description: str = ""
    feedback_event_id: str | None = None
    slate_feedback_id: str | None = None
    preference_update: dict[str, Any] = field(default_factory=dict)
    graphzep_scheduled: bool = False
    consolidation_scheduled: bool = False
    error: str = ""


class MemoryAdapter(Protocol):
    def remember_event(self, event_type: str, title: str, artist: str, user_id: str, extra: dict[str, Any]) -> None:
        ...

    def remember_preference(self, user_id: str, preferences: dict[str, Any]) -> None:
        ...

    def remember_inferred_preference(self, user_id: str, record: dict[str, Any]) -> bool:
        ...

    def get_user_profile(self, user_id: str, limit: int = 30) -> dict[str, Any]:
        ...

    def delete_memory(self, user_id: str, *, title: str = "", artist: str = "", memory_type: str = "") -> bool:
        ...

    def clear_learned_preferences(self, user_id: str) -> bool:
        ...


class EpisodicMemoryAdapter(Protocol):
    name: str

    async def remember_text(self, description: str, *, user_id: str = "local_admin", extra: dict[str, Any] | None = None) -> bool:
        ...

    async def retrieve_context(self, query: str, *, user_id: str = "local_admin", max_facts: int = 8) -> str:
        ...


class Neo4jPreferenceAdapter:
    """Structured behavior and preference memory backed by Neo4j."""

    def __init__(self, manager: UserMemoryManager | None = None):
        self.manager = manager or UserMemoryManager()

    def remember_event(self, event_type: str, title: str, artist: str, user_id: str, extra: dict[str, Any]) -> None:
        self.manager.ensure_user_exists(user_id)
        if event_type == "like":
            self.manager.record_liked_song(user_id, title, artist)
        elif event_type == "save":
            self.manager.record_saved_song(user_id, title, artist)
        elif event_type == "repeat":
            self.manager.record_liked_song(user_id, title, artist)
        elif event_type == "unlike":
            self.manager.remove_like(user_id, title, artist)
        elif event_type == "unsave":
            self.manager.remove_save(user_id, title, artist)
        elif event_type == "skip":
            self.manager.record_skipped(user_id, title, artist)
        elif event_type == "dislike":
            self.manager.record_dislike(user_id, title, artist)
        elif event_type == "full_play":
            duration = int(extra.get("play_duration_ms") or extra.get("duration") or 0)
            self.manager.record_listened_song(user_id, title, artist, duration=duration)

    def remember_preference(self, user_id: str, preferences: dict[str, Any]) -> None:
        if not preferences:
            return
        self.manager.update_semantic_preferences(user_id, preferences)

    def remember_inferred_preference(self, user_id: str, record: dict[str, Any]) -> bool:
        return self.manager.upsert_inferred_preference(user_id, record)

    def get_user_profile(self, user_id: str, limit: int = 30) -> dict[str, Any]:
        prefs = self.manager.get_user_preferences(user_id, limit=limit) or {}
        return {
            "favorite_songs": prefs.get("favorite_songs", []) or [],
            "favorite_genres": prefs.get("favorite_genres", []) or [],
            "favorite_artists": prefs.get("favorite_artists", []) or [],
            "favorite_moods": prefs.get("favorite_moods", []) or [],
            "favorite_themes": prefs.get("favorite_themes", []) or [],
            "favorite_scenarios": prefs.get("favorite_scenarios", []) or [],
            "preferred_genres_explicit": prefs.get("preferred_genres_explicit", []) or [],
            "preferred_artists_explicit": prefs.get("preferred_artists_explicit", []) or [],
            "avoid_genres": prefs.get("avoid_genres", []) or [],
            "avoid_artists": prefs.get("avoid_artists", []) or [],
            "add_moods": prefs.get("add_moods", []) or [],
            "avoid_moods": prefs.get("avoid_moods", []) or [],
            "add_scenarios": prefs.get("add_scenarios", []) or [],
            "avoid_scenarios": prefs.get("avoid_scenarios", []) or [],
            "mood_tendency": prefs.get("mood_tendency", "") or "",
            "activity_contexts": prefs.get("activity_contexts", []) or [],
            "language_preference": prefs.get("language_preference", "") or "",
            "preferred_genres": prefs.get("preferred_genres", []) or [],
            "preferred_moods": prefs.get("preferred_moods", []) or [],
            "preferred_scenarios": prefs.get("preferred_scenarios", []) or [],
            "preferred_languages": prefs.get("preferred_languages", []) or [],
        }

    def delete_memory(self, user_id: str, *, title: str = "", artist: str = "", memory_type: str = "") -> bool:
        if memory_type == "like":
            self.manager.remove_like(user_id, title, artist)
            return True
        if memory_type == "save":
            self.manager.remove_save(user_id, title, artist)
            return True
        return False

    def clear_learned_preferences(self, user_id: str) -> bool:
        return self.manager.clear_inferred_preferences(user_id)


class NoopMemoryAdapter:
    def remember_event(self, event_type: str, title: str, artist: str, user_id: str, extra: dict[str, Any]) -> None:
        return None

    def remember_preference(self, user_id: str, preferences: dict[str, Any]) -> None:
        return None

    def remember_inferred_preference(self, user_id: str, record: dict[str, Any]) -> bool:
        return False

    def get_user_profile(self, user_id: str, limit: int = 30) -> dict[str, Any]:
        return {}

    def delete_memory(self, user_id: str, *, title: str = "", artist: str = "", memory_type: str = "") -> bool:
        return False

    def clear_learned_preferences(self, user_id: str) -> bool:
        return False


class GraphZepAdapter:
    """Optional episodic sidecar. It never blocks the recommendation hot path."""

    name = "graphzep"

    async def remember_text(self, description: str, *, user_id: str = "local_admin", extra: dict[str, Any] | None = None) -> bool:
        try:
            from services.graphzep_client import get_graphzep_client, group_id_for_user

            return await get_graphzep_client().add_user_event(
                event_description=description,
                group_id=group_id_for_user(user_id),
            )
        except Exception as exc:
            logger.debug("[MemoryGateway] GraphZep side-write skipped: %s", exc)
            return False

    async def retrieve_context(self, query: str, *, user_id: str = "local_admin", max_facts: int = 8) -> str:
        try:
            from services.graphzep_client import get_graphzep_client, group_id_for_user

            return await get_graphzep_client().search_facts(
                query=query,
                group_ids=[group_id_for_user(user_id)],
                max_facts=max_facts,
            )
        except Exception:
            return "暂无用户长期记忆（GraphZep 服务不可用）"


class Mem0Adapter:
    """Optional Mem0 sidecar.

    The import is lazy so the project can run without the mem0 package.  This
    adapter is intentionally narrow: it only handles natural-language long-term
    memory while Neo4j remains the deterministic preference hot path.
    """

    name = "mem0"

    def __init__(self):
        self._client: Any | None = None
        self._available = True

    def _get_client(self) -> Any | None:
        if not self._available:
            return None
        if self._client is not None:
            return self._client
        try:
            from mem0 import Memory  # type: ignore

            config_path = os.getenv("MEM0_CONFIG_PATH", "").strip()
            if config_path:
                self._client = Memory.from_config(config_path)
            else:
                self._client = Memory()
            return self._client
        except Exception as exc:
            self._available = False
            logger.info("[MemoryGateway] Mem0 sidecar disabled/unavailable: %s", exc)
            return None

    async def remember_text(self, description: str, *, user_id: str = "local_admin", extra: dict[str, Any] | None = None) -> bool:
        client = self._get_client()
        if client is None:
            return False
        try:
            result = client.add(description, user_id=user_id, metadata=extra or {})
            if asyncio.iscoroutine(result):
                await result
            return True
        except Exception as exc:
            logger.debug("[MemoryGateway] Mem0 side-write skipped: %s", exc)
            return False

    async def retrieve_context(self, query: str, *, user_id: str = "local_admin", max_facts: int = 8) -> str:
        client = self._get_client()
        if client is None:
            return ""
        try:
            result = client.search(query, user_id=user_id, limit=max_facts)
            if asyncio.iscoroutine(result):
                result = await result
            memories = result.get("results", result) if isinstance(result, dict) else result
            lines: list[str] = []
            for item in memories or []:
                if isinstance(item, dict):
                    text = item.get("memory") or item.get("text") or item.get("content") or ""
                else:
                    text = str(item)
                text = str(text).strip()
                if text:
                    lines.append(text)
            return "\n".join(lines[:max_facts])
        except Exception as exc:
            logger.debug("[MemoryGateway] Mem0 retrieve skipped: %s", exc)
            return ""


def _configured_episodic_adapters(enable_graphzep_sidecar: bool = True) -> list[EpisodicMemoryAdapter]:
    raw = os.getenv("MEMORY_EPISODIC_BACKENDS", "").strip()
    names = [name.strip().lower() for name in raw.split(",") if name.strip()]
    adapters: list[EpisodicMemoryAdapter] = []
    if enable_graphzep_sidecar and "graphzep" in names:
        adapters.append(GraphZepAdapter())
    if "mem0" in names:
        adapters.append(Mem0Adapter())
    return adapters


def _clean_list(values: list[Any] | tuple[Any, ...] | set[Any] | None) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        text = str(value or "").strip()
        key = text.casefold()
        if text and key not in seen:
            seen.add(key)
            out.append(text)
    return out


def _profile_list_count(profile: dict[str, Any], key: str) -> int:
    value = profile.get(key)
    return len(value) if isinstance(value, list) else 1 if str(value or "").strip() else 0


def summarize_memory_profile(profile: dict[str, Any], episodic_backends: list[str]) -> dict[str, Any]:
    """Return privacy-preserving counts for editable memory health."""
    positive_fields = (
        "favorite_songs",
        "favorite_genres",
        "favorite_artists",
        "favorite_moods",
        "favorite_themes",
        "favorite_scenarios",
        "preferred_genres_explicit",
        "preferred_artists_explicit",
        "add_moods",
        "add_scenarios",
        "preferred_genres",
        "preferred_moods",
        "preferred_scenarios",
        "preferred_languages",
    )
    negative_fields = (
        "avoid_genres",
        "avoid_artists",
        "avoid_moods",
        "avoid_scenarios",
    )
    context_fields = (
        "mood_tendency",
        "activity_contexts",
        "language_preference",
    )
    positive_count = sum(_profile_list_count(profile, key) for key in positive_fields)
    negative_count = sum(_profile_list_count(profile, key) for key in negative_fields)
    context_count = sum(_profile_list_count(profile, key) for key in context_fields)
    return {
        "positive_preference_count": positive_count,
        "negative_preference_count": negative_count,
        "context_preference_count": context_count,
        "hot_path_has_signal": bool(positive_count or negative_count or context_count),
        "episodic_enabled": bool(episodic_backends),
        "episodic_backends": episodic_backends,
        "needs_more_feedback": not bool(positive_count or negative_count or context_count),
    }


EDITABLE_MEMORY_FIELDS = (
    ("avoid_genres", "避开流派", "avoid"),
    ("avoid_moods", "避开情绪", "avoid"),
    ("avoid_scenarios", "避开场景", "avoid"),
    ("add_moods", "偏好情绪", "positive"),
    ("add_scenarios", "偏好场景", "positive"),
    ("activity_contexts", "探索倾向", "context"),
)


def editable_memory_sections(profile: dict[str, Any]) -> list[dict[str, Any]]:
    """Return UI-ready editable learned-memory sections."""
    sections: list[dict[str, Any]] = []
    for field_name, label, tone in EDITABLE_MEMORY_FIELDS:
        values = _clean_list(profile.get(field_name) if isinstance(profile.get(field_name), list) else [])
        sections.append(
            {
                "field": field_name,
                "label": label,
                "tone": tone,
                "values": values,
                "count": len(values),
                "deletable": True,
            }
        )
    return sections


class MemoryGateway:
    def __init__(
        self,
        primary: MemoryAdapter | None = None,
        episodic: EpisodicMemoryAdapter | None = None,
        episodic_adapters: list[EpisodicMemoryAdapter] | None = None,
        enable_graphzep_sidecar: bool = True,
        event_store: MemoryEventStore | None = None,
        enable_event_ledger: bool | None = None,
        memory_mode: str | None = None,
        consolidator: MemoryConsolidator | None = None,
        relevance_retriever: MemoryRelevanceRetriever | None = None,
        enable_consolidation: bool | None = None,
    ):
        primary_was_injected = primary is not None
        configured_mode = str(memory_mode or os.getenv("MEMORY_MODE", "structured")).strip().lower()
        if episodic_adapters is not None and memory_mode is None:
            configured_mode = "sidecar"
        if configured_mode not in {"off", "structured", "semantic", "sidecar"}:
            configured_mode = "structured"
        self.mode = configured_mode
        self.primary = primary or (NoopMemoryAdapter() if self.mode == "off" else Neo4jPreferenceAdapter())
        ledger_enabled = (not primary_was_injected) if enable_event_ledger is None else enable_event_ledger
        if self.mode == "off":
            ledger_enabled = False
        self.event_store = event_store or (MemoryEventStore() if ledger_enabled else None)
        configured_consolidation = str(os.getenv("MEMORY_CONSOLIDATION_ENABLED", "1")).strip().lower()
        self.consolidation_enabled = (
            enable_consolidation
            if enable_consolidation is not None
            else configured_consolidation in {"1", "true", "yes", "on"}
        )
        self.consolidator = consolidator or (
            MemoryConsolidator(self.event_store)
            if self.event_store is not None and self.mode != "off"
            else None
        )
        self.relevance_retriever = relevance_retriever or MemoryRelevanceRetriever(
            min_relevance=float(os.getenv("MEMORY_RETRIEVAL_MIN_RELEVANCE", "0.08"))
        )
        self._consolidation_last_started: dict[str, float] = {}
        if self.mode != "sidecar":
            self.episodic_adapters = []
        elif episodic_adapters is not None:
            self.episodic_adapters = episodic_adapters
        elif episodic is not None:
            self.episodic_adapters = [episodic]
        else:
            self.episodic_adapters = _configured_episodic_adapters(enable_graphzep_sidecar)

    async def remember_event(
        self,
        *,
        event_type: str,
        title: str,
        artist: str,
        user_id: str = "local_admin",
        exposure_id: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> MemoryWriteResult:
        if side_effects_disabled():
            return MemoryWriteResult(success=True, description="eval_read_only")
        if event_type not in SUPPORTED_USER_EVENTS:
            return MemoryWriteResult(success=False, error="Unsupported user event type")

        extra_payload = extra or {}
        description = EVENT_TEMPLATES.get(event_type, "用户对《{title}》{artist} 执行了操作").format(
            title=title,
            artist=artist,
        )

        self.primary.remember_event(event_type, title, artist, user_id, extra_payload)
        feedback_event_id = log_user_event(
            event_type=event_type,
            song_title=title,
            artist=artist,
            user_id=user_id,
            exposure_id=exposure_id,
            extra=extra_payload,
        )
        self._append_memory_record(
            user_id=user_id,
            layer=MemoryLayer.RAW_EVENT,
            kind=event_type,
            source="user_action",
            evidence_id=feedback_event_id,
            payload={"title": title, "artist": artist, "exposure_id": exposure_id, **extra_payload},
            memory_key=f"song:{title.casefold()}:{artist.casefold()}:{event_type}",
            why_used="Raw behavior evidence; not injected directly into ranking",
        )
        self._invalidate_hot_profile(user_id)
        consolidation_scheduled = self._schedule_consolidation(user_id)

        graphzep_scheduled = False
        if event_type != "play_start":
            graphzep_scheduled = self._schedule_sidecar_write(description, user_id=user_id, extra=extra_payload)

        return MemoryWriteResult(
            success=True,
            description=description,
            feedback_event_id=feedback_event_id,
            graphzep_scheduled=graphzep_scheduled,
            consolidation_scheduled=consolidation_scheduled,
        )

    async def remember_slate_feedback(
        self,
        *,
        exposure_id: str,
        rating: str,
        reasons: list[str] | None = None,
        note: str = "",
        user_id: str = "local_admin",
        extra: dict[str, Any] | None = None,
    ) -> MemoryWriteResult:
        if side_effects_disabled():
            return MemoryWriteResult(success=True, description="eval_read_only")
        feedback_id = log_slate_feedback(
            exposure_id=exposure_id,
            rating=rating,
            reasons=reasons or [],
            note=note,
            user_id=user_id,
            extra=extra or {},
        )
        # Whole-slate feedback is evidence, not an immediately permanent profile
        # mutation. The LLM consolidator may turn repeated evidence into expiring L2.
        self._append_memory_record(
            user_id=user_id,
            layer=MemoryLayer.RAW_EVENT,
            kind="slate_feedback",
            source="slate_feedback",
            evidence_id=feedback_id,
            payload={
                "exposure_id": exposure_id,
                "rating": rating,
                "reasons": _clean_list(reasons),
                "note": str(note or "")[:500],
                **(extra or {}),
            },
            memory_key=f"slate:{feedback_id}",
            why_used="Bounded user feedback evidence for later consolidation",
        )
        consolidation_scheduled = self._schedule_consolidation(user_id)

        description = f"用户评价本次推荐歌单: {rating}"
        if reasons:
            description += "；原因: " + "、".join(_clean_list(reasons))
        if note:
            description += f"；补充: {note[:120]}"

        graphzep_scheduled = self._schedule_sidecar_write(description, user_id=user_id, extra=extra or {})

        return MemoryWriteResult(
            success=True,
            description=description,
            slate_feedback_id=feedback_id,
            preference_update={},
            graphzep_scheduled=graphzep_scheduled,
            consolidation_scheduled=consolidation_scheduled,
        )

    def remember_preference(self, *, user_id: str, preferences: dict[str, Any]) -> MemoryWriteResult:
        """Write user-confirmed L1 preferences. Inference must use consolidate_user."""
        if side_effects_disabled():
            return MemoryWriteResult(success=True, description="eval_read_only")
        self.primary.remember_preference(user_id, preferences)
        self._append_preferences(
            user_id=user_id,
            preferences=preferences,
            layer=MemoryLayer.EXPLICIT,
            source="user_explicit",
            evidence_id="manual_preference",
            confidence=1.0,
        )
        self._invalidate_hot_profile(user_id)
        return MemoryWriteResult(success=True, preference_update=preferences)

    def remember_conversation_evidence(
        self,
        *,
        user_id: str,
        user_text: str,
        scene: str = "",
        time_label: str = "",
        recommended_songs: list[str] | None = None,
    ) -> MemoryWriteResult:
        """Store user-only L0 evidence and debounce long-term consolidation."""
        if side_effects_disabled():
            return MemoryWriteResult(success=True, description="eval_read_only")
        text = str(user_text or "").strip()
        if not text:
            return MemoryWriteResult(success=False, error="user_text is required")
        record = self._append_memory_record(
            user_id=user_id,
            layer=MemoryLayer.RAW_EVENT,
            kind="conversation_statement",
            source="user_statement",
            evidence_id="",
            payload={
                "user_text": text[:1200],
                "scene": str(scene or "")[:120],
                "time_label": str(time_label or "")[:60],
                "recommended_songs": _clean_list(recommended_songs)[:10],
            },
            memory_key="",
            why_used="User-authored evidence; assistant output is deliberately excluded",
        )
        scheduled = self._schedule_consolidation(user_id)
        return MemoryWriteResult(
            success=record is not None,
            description="conversation_evidence_recorded",
            feedback_event_id=getattr(record, "record_id", None),
            consolidation_scheduled=scheduled,
        )

    async def consolidate_user(self, *, user_id: str, force: bool = False) -> dict[str, Any]:
        """Run bounded LLM consolidation and project accepted L2 records."""
        if side_effects_disabled() or self.event_store is None or self.consolidator is None:
            return {"user_id": user_id, "skipped": True, "reason": "memory_ledger_disabled"}
        if not force and not self._consolidation_ready(user_id):
            return {"user_id": user_id, "skipped": True, "reason": "debounced_or_insufficient_evidence"}

        self._consolidation_last_started[user_id] = time.monotonic()
        try:
            report = await self.consolidator.consolidate(user_id=user_id)
            projected: list[str] = []
            for candidate in report.accepted:
                record = self._append_inferred_candidate(user_id=user_id, candidate=candidate)
                if record is None:
                    continue
                payload = {
                    **record.payload,
                    "memory_key": record.memory_key,
                    "confidence": record.confidence,
                    "created_at": record.created_at,
                    "expires_at": record.expires_at,
                    "ledger_record_id": record.record_id,
                    "source": record.source,
                }
                projector = getattr(self.primary, "remember_inferred_preference", None)
                if callable(projector) and projector(user_id, payload):
                    projected.append(record.memory_key)

            audit = report.model_dump()
            audit["projected_memory_keys"] = projected
            self._append_memory_record(
                user_id=user_id,
                layer=MemoryLayer.RAW_EVENT,
                kind="consolidation_audit",
                source="memory_consolidator",
                evidence_id="",
                payload=audit,
                memory_key="",
                why_used="Auditable decision summary; excluded from future evidence",
            )
            if report.accepted:
                self._invalidate_hot_profile(user_id)
            return {**audit, "skipped": False}
        except Exception as exc:
            logger.warning("[MemoryV2] consolidation failed for %s: %s", user_id, exc)
            return {"user_id": user_id, "skipped": True, "reason": str(exc)}

    async def remember_text(
        self,
        *,
        description: str,
        user_id: str = "local_admin",
        extra: dict[str, Any] | None = None,
    ) -> MemoryWriteResult:
        if side_effects_disabled():
            return MemoryWriteResult(success=True, description="eval_read_only")
        scheduled = self._schedule_sidecar_write(description, user_id=user_id, extra=extra or {})
        self._append_memory_record(
            user_id=user_id,
            layer=MemoryLayer.EPISODIC,
            kind="episode_summary",
            source=str((extra or {}).get("source") or "conversation"),
            evidence_id=str((extra or {}).get("evidence_id") or ""),
            payload={"description": description[:2000], **(extra or {})},
            confidence=float((extra or {}).get("confidence") or 0.65),
            ttl_days=90,
            why_used="Low-frequency episodic context; explicit preferences take precedence",
        )
        return MemoryWriteResult(
            success=True,
            description=description,
            graphzep_scheduled=scheduled,
        )

    async def retrieve_context(self, *, query: str, user_id: str = "local_admin", max_facts: int = 8) -> dict[str, Any]:
        profile = self.get_user_profile(user_id)
        backend_results: dict[str, str] = {}
        retrieved_records: list[dict[str, Any]] = []
        if self.episodic_adapters:
            results = await asyncio.gather(
                *[
                    adapter.retrieve_context(query, user_id=user_id, max_facts=max_facts)
                    for adapter in self.episodic_adapters
                ],
                return_exceptions=True,
            )
            for adapter, result in zip(self.episodic_adapters, results):
                if isinstance(result, Exception):
                    logger.debug("[MemoryGateway] %s retrieve failed: %s", adapter.name, result)
                    backend_results[adapter.name] = ""
                else:
                    backend_results[adapter.name] = str(result or "").strip()
        lines: list[str] = []
        if self.mode != "off" and self.event_store is not None:
            records = self.event_store.effective_records(user_id=user_id, limit=200)
            selected = self.relevance_retriever.retrieve(
                query=query,
                records=records,
                max_facts=max_facts,
                include_episodic=self.mode in {"semantic", "sidecar"},
            )
            ledger_lines: list[str] = []
            for item in selected:
                record = item.record
                field = str(record.payload.get("field") or "").strip()
                value = str(
                    record.payload.get("value")
                    or record.payload.get("user_text")
                    or record.payload.get("description")
                    or ""
                ).strip()
                if not value:
                    continue
                prefix = f"{field}=" if field else ""
                ledger_lines.append(
                    f"[{record.layer.value}; confidence={record.confidence:.2f}] {prefix}{value}"
                )
                retrieved_records.append(item.model_dump())
            if ledger_lines:
                backend_results["memory_v2"] = "\n".join(ledger_lines)
        for name, text in backend_results.items():
            if text and "暂无用户长期记忆" not in text:
                lines.append(f"[{name}] {text}")
        return {
            "profile": profile,
            "episodic": "\n".join(lines),
            "episodic_backends": backend_results,
            "retrieved_records": retrieved_records,
        }

    def get_user_profile(self, user_id: str = "local_admin", limit: int = 30) -> dict[str, Any]:
        try:
            return self.primary.get_user_profile(user_id, limit=limit)
        except Exception as exc:
            logger.warning("[MemoryGateway] get_user_profile failed: %s", exc)
            return {}

    def delete_memory(self, *, user_id: str, title: str = "", artist: str = "", memory_type: str = "") -> bool:
        ok = self.primary.delete_memory(user_id, title=title, artist=artist, memory_type=memory_type)
        if self.event_store is not None:
            for record in self.event_store.effective_records(user_id=user_id, limit=500):
                payload = record.payload
                if (
                    str(payload.get("title") or "").casefold() == title.casefold()
                    and str(payload.get("artist") or "").casefold() == artist.casefold()
                    and (not memory_type or record.kind == memory_type)
                ):
                    self.event_store.tombstone(user_id=user_id, target_record_id=record.record_id)
        self._invalidate_hot_profile(user_id)
        return ok

    def explain_memory(self, *, user_id: str = "local_admin") -> dict[str, Any]:
        profile = self.get_user_profile(user_id)
        episodic_backends = [adapter.name for adapter in self.episodic_adapters]
        return {
            "user_id": user_id,
            "mode": self.mode,
            "hot_path": {
                "likes_and_saves": "Neo4j user-song relations",
                "explicit_preferences": "Neo4j User properties (L1)",
                "inferred_preferences": "Expiring Neo4j InferredPreference projection (L2)",
                "slate_feedback": "L0 evidence; consolidated only after validation",
            },
            "episodic_backends": episodic_backends,
            "profile": profile,
            "editable_sections": editable_memory_sections(profile),
            "diagnostics": summarize_memory_profile(profile, episodic_backends),
            "records": self.list_memory_records(user_id=user_id),
        }

    def forget_preference_item(self, *, user_id: str, field: str, value: str) -> bool:
        manager = getattr(self.primary, "manager", None)
        if manager is None:
            return False
        explicit_ok = bool(
            hasattr(manager, "remove_semantic_preference")
            and manager.remove_semantic_preference(user_id, field, value)
        )
        inferred_ok = bool(
            hasattr(manager, "delete_inferred_preference")
            and manager.delete_inferred_preference(user_id, field=field, value=value)
        )
        ok = explicit_ok or inferred_ok
        if ok:
            self._tombstone_preference(user_id=user_id, field=field, value=value)
            self._invalidate_hot_profile(user_id)
        return ok

    def clear_learned_preferences(self, *, user_id: str) -> bool:
        ok = bool(self.primary.clear_learned_preferences(user_id))
        if ok:
            if self.event_store is not None:
                for record in self.event_store.effective_records(user_id=user_id, limit=1000):
                    if record.layer == MemoryLayer.INFERRED:
                        self.event_store.tombstone(user_id=user_id, target_record_id=record.record_id)
            self._invalidate_hot_profile(user_id)
        return ok

    def list_memory_records(self, *, user_id: str, limit: int = 200) -> list[dict[str, Any]]:
        if self.event_store is None:
            return []
        return [record.model_dump() for record in self.event_store.effective_records(user_id=user_id, limit=limit)]

    def delete_memory_record(self, *, user_id: str, record_id: str) -> bool:
        if self.event_store is None:
            return False
        record = self.event_store.get(user_id=user_id, record_id=record_id)
        if record is None:
            return False
        if record.layer == MemoryLayer.EXPLICIT:
            field = str(record.payload.get("field") or "")
            value = str(record.payload.get("value") or "")
            manager = getattr(self.primary, "manager", None)
            if field and value and manager is not None and hasattr(manager, "remove_semantic_preference"):
                manager.remove_semantic_preference(user_id, field, value)
        elif record.layer == MemoryLayer.INFERRED:
            manager = getattr(self.primary, "manager", None)
            if manager is not None and hasattr(manager, "delete_inferred_preference"):
                manager.delete_inferred_preference(
                    user_id,
                    memory_key=record.memory_key,
                )
        tombstone = self.event_store.tombstone(user_id=user_id, target_record_id=record_id)
        if tombstone is not None:
            self._invalidate_hot_profile(user_id)
        return tombstone is not None

    def _append_preferences(
        self,
        *,
        user_id: str,
        preferences: dict[str, Any],
        layer: MemoryLayer,
        source: str,
        evidence_id: str,
        confidence: float,
        ttl_days: int | None = None,
    ) -> None:
        for preference_field, raw_value in (preferences or {}).items():
            values = raw_value if isinstance(raw_value, list) else [raw_value]
            for value in values:
                text = str(value or "").strip()
                if not text:
                    continue
                self._append_memory_record(
                    user_id=user_id,
                    layer=layer,
                    kind="preference",
                    source=source,
                    evidence_id=evidence_id,
                    payload={"field": preference_field, "value": text},
                    confidence=confidence,
                    ttl_days=ttl_days,
                    memory_key=f"preference:{preference_field}:{text.casefold()}",
                    why_used=(
                        "Explicit user preference overrides inferred memory"
                        if layer == MemoryLayer.EXPLICIT
                        else "Recent inferred preference; expires unless reinforced"
                    ),
                )

    def _append_memory_record(self, *, ttl_days: int | None = None, **kwargs: Any):
        if self.event_store is None:
            return None
        expires_at = None
        if ttl_days is not None:
            expires_at = int(time.time() * 1000) + ttl_days * 24 * 60 * 60 * 1000
        return self.event_store.append(expires_at=expires_at, **kwargs)

    def _append_inferred_candidate(self, *, user_id: str, candidate):
        evidence_ids = list(candidate.evidence_ids)
        return self._append_memory_record(
            user_id=user_id,
            layer=MemoryLayer.INFERRED,
            kind="preference",
            source="memory_consolidator",
            evidence_id=evidence_ids[0] if evidence_ids else "",
            payload={
                "field": candidate.field,
                "value": candidate.value,
                "scope": candidate.scope,
                "evidence_ids": evidence_ids,
                "counter_evidence_ids": list(candidate.counter_evidence_ids),
                "retrieval_cues": list(candidate.retrieval_cues),
                "decision_summary": candidate.decision_summary,
            },
            confidence=float(candidate.confidence),
            ttl_days=int(candidate.ttl_days),
            memory_key=MemoryConsolidator.memory_key(candidate.field, candidate.value),
            why_used="LLM-proposed preference passed deterministic evidence validation",
        )

    def _consolidation_ready(self, user_id: str) -> bool:
        if not self.consolidation_enabled or self.event_store is None:
            return False
        min_events = max(2, int(os.getenv("MEMORY_CONSOLIDATION_MIN_EVENTS", "5")))
        if self.event_store.pending_evidence_count(user_id=user_id) < min_events:
            return False
        cooldown = max(0, int(os.getenv("MEMORY_CONSOLIDATION_COOLDOWN_SECONDS", "900")))
        last_started = self._consolidation_last_started.get(user_id, 0.0)
        return time.monotonic() - last_started >= cooldown

    def _schedule_consolidation(self, user_id: str) -> bool:
        if not self._consolidation_ready(user_id):
            return False
        try:
            asyncio.get_running_loop().create_task(self.consolidate_user(user_id=user_id, force=True))
            self._consolidation_last_started[user_id] = time.monotonic()
            return True
        except RuntimeError:
            return False

    def _tombstone_preference(self, *, user_id: str, field: str, value: str) -> None:
        if self.event_store is None:
            return
        key = f"preference:{field}:{value.casefold()}"
        for record in self.event_store.effective_records(user_id=user_id, limit=500):
            if record.memory_key == key:
                self.event_store.tombstone(user_id=user_id, target_record_id=record.record_id)

    def _schedule_sidecar_write(self, description: str, *, user_id: str, extra: dict[str, Any] | None = None) -> bool:
        scheduled = False
        for adapter in self.episodic_adapters:
            asyncio.create_task(adapter.remember_text(description, user_id=user_id, extra=extra or {}))
            scheduled = True
        return scheduled

    @staticmethod
    def _invalidate_hot_profile(user_id: str) -> None:
        try:
            from retrieval.hybrid_retrieval import invalidate_user_pref_cache

            invalidate_user_pref_cache(user_id)
        except Exception:
            pass
        try:
            from services.policy_memory import invalidate_policy_memory_cache

            invalidate_policy_memory_cache(user_id)
        except Exception:
            pass


_gateway: MemoryGateway | None = None


def get_memory_gateway() -> MemoryGateway:
    global _gateway
    if _gateway is None:
        _gateway = MemoryGateway()
    return _gateway


def reset_memory_gateway_for_tests(gateway: MemoryGateway | None = None) -> None:
    global _gateway
    _gateway = gateway
