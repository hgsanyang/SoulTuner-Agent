"""Production adapters for SoulTuner's bounded ToolPlan registry.

The registry is intentionally built per request so user identity and current
retrieval context are injected by trusted application code, never by the LLM.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Mapping

from agent.catalog_gap import analyze_catalog_gap
from agent.tool_orchestrator import ToolRegistry
from retrieval.recall_sources import graph_candidate_recall
from schemas.tool_plan import ToolName, ToolObservation
from services.memory_gateway import get_memory_gateway
from tools.music_fetch_tool import execute_search_online_music
from tools.semantic_search import semantic_search


def _decode_songs(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if isinstance(value, dict):
        value = value.get("songs") or value.get("data") or []
    return [dict(item) for item in value if isinstance(item, Mapping)] if isinstance(value, list) else []


def _dependency_songs(dependencies: Mapping[str, ToolObservation]) -> list[dict[str, Any]]:
    songs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for observation in dependencies.values():
        for song in _decode_songs(observation.data):
            key = str(song.get("music_id") or song.get("id") or f"{song.get('title')}::{song.get('artist')}")
            if key and key not in seen:
                seen.add(key)
                songs.append(song)
    return songs


def build_music_tool_registry(
    *,
    user_id: str,
    query: str,
    retrieval_plan: Mapping[str, Any] | None = None,
    web_enabled: bool = True,
) -> ToolRegistry:
    """Build request-scoped executors without exposing identity or credentials."""

    registry = ToolRegistry()
    trusted_plan = dict(retrieval_plan or {})

    async def retrieve_memory(arguments: dict[str, Any], _deps: dict[str, ToolObservation]) -> Any:
        gateway = get_memory_gateway()
        return await gateway.retrieve_context(
            query=str(arguments.get("query") or query),
            user_id=user_id,
            max_facts=int(arguments.get("limit") or 8),
        )

    async def search_graph(arguments: dict[str, Any], _deps: dict[str, ToolObservation]) -> Any:
        hard = {
            "artist_entities": arguments.get("artist_entities") or [],
            "song_entities": arguments.get("song_entities") or [],
            "language": arguments.get("language"),
            "region": arguments.get("region"),
            "instrumental": bool(arguments.get("instrumental")),
        }
        hints = {
            "genres": arguments.get("genres") or [],
            "mood": (arguments.get("moods") or [None])[0],
            "scenario": (arguments.get("scenarios") or [None])[0],
        }
        raw = await asyncio.to_thread(
            graph_candidate_recall,
            hard,
            hints,
            limit=int(arguments.get("limit") or 30),
        )
        songs = _decode_songs(raw)
        year_from = arguments.get("release_year_from")
        year_to = arguments.get("release_year_to")
        if year_from is not None or year_to is not None:
            filtered = []
            for song in songs:
                try:
                    year = int(song.get("release_year") or song.get("year"))
                except (TypeError, ValueError):
                    continue
                if year_from is not None and year < int(year_from):
                    continue
                if year_to is not None and year > int(year_to):
                    continue
                filtered.append(song)
            songs = filtered
        return {"songs": songs, "source": "graph"}

    async def search_audio(arguments: dict[str, Any], _deps: dict[str, ToolObservation]) -> Any:
        variants = list(arguments.get("acoustic_queries") or [])
        payload = {
            "query": variants[0],
            "query_variants": variants,
            "limit": int(arguments.get("limit") or 30),
        }
        raw = await asyncio.to_thread(semantic_search.invoke, payload)
        return {"songs": _decode_songs(raw), "source": "audio"}

    async def inspect_gap(arguments: dict[str, Any], dependencies: dict[str, ToolObservation]) -> Any:
        plan = dict(trusted_plan)
        if arguments.get("requirements"):
            plan["metadata_constraints"] = dict(arguments["requirements"])
        decision = analyze_catalog_gap(
            _dependency_songs(dependencies),
            plan,
            query,
            web_enabled=web_enabled,
        )
        data = decision.model_dump()
        data["metadata"] = {
            "needs_replan": bool(decision.needs_online),
            "target_web_count": decision.target_web_count,
        }
        return data

    async def search_external(arguments: dict[str, Any], _deps: dict[str, ToolObservation]) -> Any:
        result = await execute_search_online_music(str(arguments.get("requirements") or query))
        return {
            "songs": list(result.data or []) if result.success else [],
            "source": "external",
            "error": result.error_message or "",
        }

    async def resolve_playable(arguments: dict[str, Any], dependencies: dict[str, ToolObservation]) -> Any:
        requested = {str(value) for value in arguments.get("candidate_source_ids") or []}
        songs = _dependency_songs(dependencies)
        if requested:
            songs = [
                song for song in songs
                if str(song.get("music_id") or song.get("id") or song.get("source_id")) in requested
            ]
        playable = [song for song in songs if song.get("audio_url") or song.get("play_url") or song.get("preview_url")]
        return {"songs": playable[: int(arguments.get("limit") or 10)], "source": "resolver"}

    async def commit_memory(arguments: dict[str, Any], _deps: dict[str, ToolObservation]) -> Any:
        result = get_memory_gateway().remember_preference(
            user_id=user_id,
            preferences=dict(arguments.get("values") or {}),
        )
        return {
            "success": result.success,
            "evidence_id": arguments.get("evidence_id"),
            "preference_update": result.preference_update,
            "error": result.error,
        }

    registry.register(ToolName.RETRIEVE_MEMORY, retrieve_memory)
    registry.register(ToolName.SEARCH_GRAPH, search_graph)
    registry.register(ToolName.SEARCH_AUDIO, search_audio)
    registry.register(ToolName.INSPECT_CATALOG_GAP, inspect_gap)
    registry.register(ToolName.SEARCH_EXTERNAL_MUSIC, search_external)
    registry.register(ToolName.RESOLVE_PLAYABLE_TRACKS, resolve_playable)
    registry.register(ToolName.COMMIT_MEMORY_DELTA, commit_memory)
    return registry
