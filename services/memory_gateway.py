"""Unified memory gateway for hot-path behavior and optional episodic memory.

The gateway keeps recommendation code away from concrete memory backends.
Neo4j remains the structured hot path; GraphZep is an optional sidecar; Mem0
can be added later behind the same interface.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Protocol

from retrieval.user_memory import UserMemoryManager
from services.feedback_logger import log_slate_feedback, log_user_event

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
    error: str = ""


class MemoryAdapter(Protocol):
    def remember_event(self, event_type: str, title: str, artist: str, user_id: str, extra: dict[str, Any]) -> None:
        ...

    def remember_preference(self, user_id: str, preferences: dict[str, Any]) -> None:
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
        return self.manager.clear_semantic_preferences(user_id)


class NoopMemoryAdapter:
    def remember_event(self, event_type: str, title: str, artist: str, user_id: str, extra: dict[str, Any]) -> None:
        return None

    def remember_preference(self, user_id: str, preferences: dict[str, Any]) -> None:
        return None

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
            from services.graphzep_client import get_graphzep_client

            return await get_graphzep_client().add_user_event(event_description=description)
        except Exception as exc:
            logger.debug("[MemoryGateway] GraphZep side-write skipped: %s", exc)
            return False

    async def retrieve_context(self, query: str, *, user_id: str = "local_admin", max_facts: int = 8) -> str:
        try:
            from services.graphzep_client import get_graphzep_client

            return await get_graphzep_client().search_facts(query=query, max_facts=max_facts)
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


def _merge_pref(target: dict[str, Any], key: str, values: list[str]) -> None:
    cleaned = _clean_list(values)
    if cleaned:
        target[key] = _clean_list([*(target.get(key) or []), *cleaned])


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


def derive_preferences_from_slate_feedback(
    *,
    rating: str,
    reasons: list[str] | None = None,
    note: str = "",
) -> dict[str, Any]:
    """Map whole-slate feedback to conservative hot-path preference updates."""

    reason_text = " ".join([rating, *(reasons or []), note]).casefold()
    update: dict[str, Any] = {}

    if rating == "too_noisy" or any(token in reason_text for token in ["太吵", "刺耳", "土嗨", "edm"]):
        _merge_pref(update, "avoid_genres", ["EDM", "Dance", "Hardcore", "Phonk"])
        _merge_pref(update, "avoid_moods", ["Energetic", "Aggressive", "Party", "Driving"])
        update["mood_tendency"] = "偏好更安静、柔软、低动态的推荐"

    if rating == "too_quiet" or any(token in reason_text for token in ["太安静", "没劲", "更有劲"]):
        _merge_pref(update, "add_moods", ["Energetic", "Upbeat", "Driving"])
        _merge_pref(update, "add_scenarios", ["Workout", "Driving"])

    if rating == "too_sad" or any(token in reason_text for token in ["太丧", "太悲", "苦情", "悲伤"]):
        _merge_pref(update, "avoid_moods", ["Sad", "Heartbreak", "Melancholy", "Lonely"])
        _merge_pref(update, "add_moods", ["Healing", "Warm", "Hopeful"])

    if rating == "more_niche" or any(token in reason_text for token in ["更小众", "冷门", "新歌", "发现更多"]):
        _merge_pref(update, "activity_contexts", ["discovery", "longtail", "less_familiar"])

    if rating == "too_familiar" or "旧歌单" in reason_text or "已收藏" in reason_text:
        _merge_pref(update, "activity_contexts", ["less_familiar", "avoid_overexposed"])

    if rating == "closer_to_seed" or "贴近刚才" in reason_text:
        _merge_pref(update, "activity_contexts", ["closer_to_seed_song"])

    if rating == "wrong_context" or "场景不贴合" in reason_text:
        _merge_pref(update, "activity_contexts", ["needs_context_refinement"])

    if rating == "too_generic" or "太普通" in reason_text:
        _merge_pref(update, "activity_contexts", ["more_distinctive", "less_generic"])

    return update


class MemoryGateway:
    def __init__(
        self,
        primary: MemoryAdapter | None = None,
        episodic: EpisodicMemoryAdapter | None = None,
        episodic_adapters: list[EpisodicMemoryAdapter] | None = None,
        enable_graphzep_sidecar: bool = True,
    ):
        self.primary = primary or Neo4jPreferenceAdapter()
        if episodic_adapters is not None:
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
        self._invalidate_hot_profile(user_id)

        graphzep_scheduled = False
        if event_type != "play_start":
            graphzep_scheduled = self._schedule_sidecar_write(description, user_id=user_id, extra=extra_payload)

        return MemoryWriteResult(
            success=True,
            description=description,
            feedback_event_id=feedback_event_id,
            graphzep_scheduled=graphzep_scheduled,
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
        feedback_id = log_slate_feedback(
            exposure_id=exposure_id,
            rating=rating,
            reasons=reasons or [],
            note=note,
            user_id=user_id,
            extra=extra or {},
        )
        preference_update = derive_preferences_from_slate_feedback(
            rating=rating,
            reasons=reasons or [],
            note=note,
        )
        if preference_update:
            self.primary.remember_preference(user_id, preference_update)
            self._invalidate_hot_profile(user_id)

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
            preference_update=preference_update,
            graphzep_scheduled=graphzep_scheduled,
        )

    def remember_preference(self, *, user_id: str, preferences: dict[str, Any]) -> MemoryWriteResult:
        self.primary.remember_preference(user_id, preferences)
        self._invalidate_hot_profile(user_id)
        return MemoryWriteResult(success=True, preference_update=preferences)

    async def remember_text(
        self,
        *,
        description: str,
        user_id: str = "local_admin",
        extra: dict[str, Any] | None = None,
    ) -> MemoryWriteResult:
        scheduled = self._schedule_sidecar_write(description, user_id=user_id, extra=extra or {})
        return MemoryWriteResult(
            success=True,
            description=description,
            graphzep_scheduled=scheduled,
        )

    async def retrieve_context(self, *, query: str, user_id: str = "local_admin", max_facts: int = 8) -> dict[str, Any]:
        profile = self.get_user_profile(user_id)
        backend_results: dict[str, str] = {}
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
        for name, text in backend_results.items():
            if text and "暂无用户长期记忆" not in text:
                lines.append(f"[{name}] {text}")
        return {
            "profile": profile,
            "episodic": "\n".join(lines),
            "episodic_backends": backend_results,
        }

    def get_user_profile(self, user_id: str = "local_admin", limit: int = 30) -> dict[str, Any]:
        try:
            return self.primary.get_user_profile(user_id, limit=limit)
        except Exception as exc:
            logger.warning("[MemoryGateway] get_user_profile failed: %s", exc)
            return {}

    def delete_memory(self, *, user_id: str, title: str = "", artist: str = "", memory_type: str = "") -> bool:
        ok = self.primary.delete_memory(user_id, title=title, artist=artist, memory_type=memory_type)
        self._invalidate_hot_profile(user_id)
        return ok

    def explain_memory(self, *, user_id: str = "local_admin") -> dict[str, Any]:
        profile = self.get_user_profile(user_id)
        episodic_backends = [adapter.name for adapter in self.episodic_adapters]
        return {
            "user_id": user_id,
            "hot_path": {
                "likes_and_saves": "Neo4j user-song relations",
                "explicit_preferences": "Neo4j User properties",
                "slate_feedback": "JSONL + deterministic preference updates",
            },
            "episodic_backends": episodic_backends,
            "profile": profile,
            "diagnostics": summarize_memory_profile(profile, episodic_backends),
        }

    def forget_preference_item(self, *, user_id: str, field: str, value: str) -> bool:
        manager = getattr(self.primary, "manager", None)
        if manager is None or not hasattr(manager, "remove_semantic_preference"):
            return False
        ok = bool(manager.remove_semantic_preference(user_id, field, value))
        if ok:
            self._invalidate_hot_profile(user_id)
        return ok

    def clear_learned_preferences(self, *, user_id: str) -> bool:
        ok = bool(self.primary.clear_learned_preferences(user_id))
        if ok:
            self._invalidate_hot_profile(user_id)
        return ok

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


_gateway: MemoryGateway | None = None


def get_memory_gateway() -> MemoryGateway:
    global _gateway
    if _gateway is None:
        _gateway = MemoryGateway()
    return _gateway


def reset_memory_gateway_for_tests(gateway: MemoryGateway | None = None) -> None:
    global _gateway
    _gateway = gateway
