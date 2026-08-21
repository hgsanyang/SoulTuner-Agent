"""Small, reconnecting Aura overlay for the public ModelScope experience."""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Iterable, Mapping


_CACHE_LOCK = threading.Lock()
_CACHE_EXPIRES_AT = 0.0
_CACHE_ROWS: dict[str, dict[str, Any]] = {}
_CACHE_STATUS: dict[str, Any] = {"state": "not-checked", "tracks": 0}


def _connection_settings() -> tuple[str, str, str, str | None]:
    uri = os.getenv("NEO4J_URI", "").strip()
    user = (os.getenv("NEO4J_USER", "").strip() or os.getenv("NEO4J_USERNAME", "").strip())
    password = os.getenv("NEO4J_PASSWORD", "").strip()
    database = os.getenv("NEO4J_DATABASE", "").strip() or None
    return uri, user, password, database


def _query_overlay() -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    uri, user, password, database = _connection_settings()
    if not all((uri, user, password)):
        return {}, {"state": "not-configured", "tracks": 0}

    try:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(
            uri,
            auth=(user, password),
            connection_timeout=float(os.getenv("NEO4J_CONNECTION_TIMEOUT_SECONDS", "5")),
        )
        try:
            driver.verify_connectivity()
            with driver.session(database=database) as session:
                records = session.run(
                    """
                    MATCH (s:Song {dataset: 'song_describer_full'})
                    RETURN toString(s.music_id) AS song_id,
                           s.title AS title,
                           s.artist AS artist,
                           s.audio_url AS audio_url,
                           s.cover_url AS cover_url,
                           s.description AS description,
                           [(s)-[:BELONGS_TO_GENRE]->(g:Genre) | g.name] AS genres,
                           [(s)-[:HAS_MOOD]->(m:Mood) | m.name] AS moods_themes,
                           [(s)-[:HAS_INSTRUMENT]->(i:Instrument) | i.name] AS instruments,
                           coalesce(s.enrichment_status, 'pending') AS enrichment_status
                    """
                )
                rows = {str(record["song_id"]): record.data() for record in records}
        finally:
            driver.close()
    except Exception as exc:
        return {}, {
            "state": "unavailable",
            "tracks": 0,
            "detail": type(exc).__name__,
        }

    enriched = sum(row.get("enrichment_status") == "ready" for row in rows.values())
    return rows, {"state": "ready", "tracks": len(rows), "enriched": enriched}


def graph_overlay(*, force: bool = False) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Return a bounded-TTL graph snapshot so a dropped Aura session can recover."""

    global _CACHE_EXPIRES_AT, _CACHE_ROWS, _CACHE_STATUS
    now = time.monotonic()
    with _CACHE_LOCK:
        if not force and now < _CACHE_EXPIRES_AT:
            return _CACHE_ROWS, dict(_CACHE_STATUS)
        rows, status = _query_overlay()
        ttl = 300.0 if status.get("state") == "ready" else 15.0
        _CACHE_ROWS = rows
        _CACHE_STATUS = status
        _CACHE_EXPIRES_AT = now + ttl
        return _CACHE_ROWS, dict(_CACHE_STATUS)


def merge_graph_overlay(rows: Iterable[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    """Overlay graph-owned metadata while retaining local playable file paths."""

    overlay, status = graph_overlay()
    merged: list[dict[str, Any]] = []
    for original in rows:
        row = dict(original)
        graph = overlay.get(str(row.get("song_id") or ""))
        if graph:
            for key in ("title", "artist", "cover_url", "description"):
                if graph.get(key):
                    row[key] = graph[key]
            for key in ("genres", "moods_themes", "instruments"):
                if graph.get(key):
                    row[key] = list(graph[key])
            row["graph_backend"] = "neo4j_aura"
            row["enrichment_status"] = graph.get("enrichment_status", "pending")
        else:
            row["graph_backend"] = "local_catalog"
        merged.append(row)
    if status.get("state") != "ready":
        for row in merged:
            row["graph_backend"] = "local_catalog"
    return tuple(merged)


def status_markdown() -> str:
    _, status = graph_overlay()
    state = status.get("state")
    if state == "ready":
        return (
            f"图谱：`Aura Neo4j 已连接` · **{status.get('tracks', 0)}** 首曲目 · "
            f"**{status.get('enriched', 0)}** 首三向量就绪"
        )
    if state == "not-configured":
        return "图谱：`未配置 Aura` · 当前使用本地公开目录。"
    return "图谱：`Aura 暂时不可用` · 已自动回退本地目录，并会继续重连。"
