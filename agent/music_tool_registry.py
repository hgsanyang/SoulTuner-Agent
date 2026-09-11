"""Production adapters for SoulTuner's bounded ToolPlan registry.

The registry is intentionally built per request so user identity and current
retrieval context are injected by trusted application code, never by the LLM.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Mapping, Callable, Awaitable

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
    return [dict(item) for item in value if isinstance(item, Mapping)
            and not item.get("error") and (item.get("title") or item.get("music_id"))] if isinstance(value, list) else []


def _audio_observation(raw: Any) -> dict[str, Any]:
    """Legacy semantic search returns JSON error records, not raised errors."""
    try:
        value = json.loads(raw) if isinstance(raw, str) else raw
    except (ValueError, TypeError):
        return {"songs": [], "source": "audio", "success": False, "error": "invalid audio search response"}
    records = value if isinstance(value, list) else [value]
    failed = any(isinstance(item, Mapping) and (item.get("error") or item.get("success") is False)
                 for item in records)
    if not isinstance(value, (list, dict)):
        failed = True
    result = {"songs": _decode_songs(value), "source": "audio", "success": not failed}
    if failed:
        # Do not expose database details from the legacy error string.
        result["error"] = "audio search failed or returned incomplete results"
    return result


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


def _song_source_id(song: Mapping[str, Any]) -> str:
    return str(
        song.get("music_id")
        or song.get("id")
        or song.get("source_id")
        or f"{song.get('title')}::{song.get('artist')}"
    ).strip()


def build_music_tool_registry(
    *,
    user_id: str,
    query: str,
    retrieval_plan: Mapping[str, Any] | None = None,
    web_enabled: bool = True,
    authorized_memory_writer: Callable[[str, dict[str, Any]], Awaitable[Any]] | None = None,
) -> ToolRegistry:
    """Build request-scoped executors without exposing identity or credentials."""

    registry = ToolRegistry()
    trusted_plan = dict(retrieval_plan or {})

    async def retrieve_memory(arguments: dict[str, Any], _deps: dict[str, ToolObservation]) -> Any:
        gateway = get_memory_gateway()
        result = await gateway.retrieve_context(
            query=str(arguments.get("query") or query),
            user_id=user_id,
            max_facts=int(arguments.get("limit") or 8),
        )
        if (result.get("memory_trace") or {}).get("status") == "degraded":
            return {**result, "success": False, "error": "memory retrieval degraded",
                    "metadata": {"partial": bool(result.get("profile") or result.get("retrieved_records"))}}
        return result

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
            "negative_targets": list(arguments.get("negative_targets") or []),
            "limit": int(arguments.get("limit") or 30),
        }
        raw = await asyncio.to_thread(semantic_search.invoke, payload)
        return _audio_observation(raw)

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
        if not web_enabled:
            return {"songs": [], "source": "external", "success": False,
                    "error": "web search is disabled for this request"}
        result = await execute_search_online_music(str(arguments.get("requirements") or query))
        return {
            "success": result.success,
            "songs": list(result.data or []),
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
        # A plan's self-reported evidence/confidence is never authorization.
        # The application writer must verify ownership/source and dispatch the
        # declared memory type; absent that integration, fail closed.
        if authorized_memory_writer is None:
            return {"success": False, "error": "memory write requires server-side evidence authorization"}
        return await authorized_memory_writer(user_id, dict(arguments))

    async def read_library(arguments: dict[str, Any], _deps: dict[str, ToolObservation]) -> Any:
        """Read a bounded user collection without accepting identity from the plan."""

        from retrieval.neo4j_client import get_neo4j_client

        relation_by_collection = {
            "liked": "LIKES",
            "saved": "SAVES",
            "disliked": "DISLIKES",
            "recent": "LISTENED_TO",
        }
        collection = str(arguments.get("collection") or "liked")
        relation = relation_by_collection[collection]
        text_query = str(arguments.get("query") or "").strip()
        cypher = f"""
        MATCH (u:User {{id: $user_id}})-[r:{relation}]->(s:Song)
        OPTIONAL MATCH (s)-[:PERFORMED_BY]->(a:Artist)
        WITH s, r, a
        WHERE $query = ''
           OR toLower(s.title) CONTAINS toLower($query)
           OR toLower(coalesce(a.name, s.artist, '')) CONTAINS toLower($query)
        RETURN s.music_id AS music_id,
               s.title AS title,
               coalesce(a.name, s.artist, '') AS artist,
               s.album AS album,
               s.audio_url AS audio_url,
               s.cover_url AS cover_url,
               coalesce(r.updated_at, r.created_at, 0) AS interaction_at
        ORDER BY interaction_at DESC
        LIMIT $limit
        """
        client = get_neo4j_client()
        records = await asyncio.to_thread(
            getattr(client, "execute_read_query", client.execute_query),
            cypher,
            {
                "user_id": user_id,
                "query": text_query,
                "limit": int(arguments.get("limit") or 30),
            },
        )
        songs = [
            {
                "music_id": record.get("music_id"),
                "title": record.get("title") or "",
                "artist": record.get("artist") or "",
                "album": record.get("album") or "",
                "audio_url": record.get("audio_url") or "",
                "cover_url": record.get("cover_url") or "",
                "interaction_at": record.get("interaction_at") or 0,
            }
            for record in records
        ]
        return {
            "songs": songs,
            "source": "library",
            "collection": collection,
            "metadata": {"read_only": True, "user_bound_by_server": True},
        }

    async def stage_ingest(
        arguments: dict[str, Any],
        dependencies: dict[str, ToolObservation],
    ) -> Any:
        """Validate an ingest proposal in shadow mode; never enqueue or persist it."""

        requested = {str(value).strip() for value in arguments.get("candidate_source_ids") or []}
        candidates = _dependency_songs(dependencies)
        if requested:
            candidates = [song for song in candidates if _song_source_id(song) in requested]

        accepted: list[dict[str, Any]] = []
        rejected: list[dict[str, str]] = []
        for song in candidates:
            source_id = _song_source_id(song)
            title = str(song.get("title") or song.get("name") or "").strip()
            artist = str(song.get("artist") or "").strip()
            audio_pointer = str(
                song.get("audio_url")
                or song.get("play_url")
                or song.get("preview_url")
                or song.get("file_basename")
                or ""
            ).strip()
            missing = [
                field
                for field, value in (("title", title), ("artist", artist), ("audio_pointer", audio_pointer))
                if not value
            ]
            if missing:
                rejected.append(
                    {
                        "source_id": source_id,
                        "reason": f"missing {', '.join(missing)}",
                    }
                )
                continue
            accepted.append(
                {
                    "source_id": source_id,
                    "title": title,
                    "artist": artist,
                    "audio_pointer": audio_pointer,
                    "preserve_audio": bool(arguments.get("preserve_audio")),
                }
            )

        return {
            "shadow": True,
            "side_effects_applied": False,
            "would_stage": accepted,
            "rejected": rejected,
            "reason": str(arguments.get("reason") or ""),
            "metadata": {
                "shadow": True,
                "side_effects_applied": False,
                "needs_confirmation": bool(accepted),
                "needs_replan": not bool(accepted),
            },
        }

    registry.register(ToolName.RETRIEVE_MEMORY, retrieve_memory)
    registry.register(ToolName.SEARCH_GRAPH, search_graph)
    registry.register(ToolName.SEARCH_AUDIO, search_audio)
    registry.register(ToolName.INSPECT_CATALOG_GAP, inspect_gap)
    registry.register(ToolName.SEARCH_EXTERNAL_MUSIC, search_external)
    registry.register(ToolName.RESOLVE_PLAYABLE_TRACKS, resolve_playable)
    registry.register(ToolName.COMMIT_MEMORY_DELTA, commit_memory)
    registry.register(ToolName.READ_LIBRARY, read_library)
    registry.register(ToolName.STAGE_INGEST, stage_ingest)
    return registry
