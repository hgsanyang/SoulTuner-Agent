"""SQLite music knowledge store with FTS5 search.

This is the lightweight local RAG store for P15/P11 follow-up work.  Neo4j
keeps graph relationships and short summaries; SQLite keeps the auditable
knowledge cards, source URLs, facts, and full-text indexes.
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping

from config.settings import settings
from services.catalog_enrichment import (
    build_artist_knowledge_query,
    build_song_knowledge_query,
    clamp_confidence,
    normalize_knowledge_card,
)
from services.music_knowledge_cache import knowledge_key

CardKind = Literal["artist", "song"]
FTS_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "song",
    "songs",
    "music",
    "artist",
    "style",
    "background",
    "info",
    "介绍",
    "讲讲",
    "一下",
    "歌曲",
    "歌手",
    "音乐",
    "背景",
    "风格",
    "资料",
    "代表作",
    "和",
    "的",
}


def resolve_store_path(path: str | Path | None = None) -> Path:
    raw = str(path or settings.knowledge_store_path or "../data/knowledge_cache/music_knowledge.sqlite")
    target = Path(raw)
    if not target.is_absolute():
        target = Path(__file__).resolve().parents[1] / target
    return target.resolve()


def _now_ms() -> int:
    return int(time.time() * 1000)


def _dumps(value: Any) -> str:
    return json.dumps(value or [], ensure_ascii=False, sort_keys=True)


def _loads(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    try:
        loaded = json.loads(str(value or "[]"))
        return loaded if isinstance(loaded, list) else []
    except Exception:
        return []


def _clean_query(query: str) -> str:
    tokens = re.findall(r"[\w\u4e00-\u9fff]+", str(query or "").casefold())
    tokens = [token for token in tokens if token and token not in FTS_STOPWORDS]
    # Keep FTS expression conservative; exact CJK spans are still useful, while
    # punctuation-heavy user text falls back to LIKE search below.
    return " OR ".join(dict.fromkeys(tokens[:8]))


def _card_key(card: Mapping[str, Any]) -> str:
    return knowledge_key(str(card.get("kind") or "song"), str(card.get("title") or ""), str(card.get("artist") or ""))


def _dedupe_cards(cards: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for card in cards:
        key = str(card.get("key") or _card_key(card))
        if key in seen:
            continue
        seen.add(key)
        rows.append(card)
    return rows


def _row_to_card(row: sqlite3.Row, kind: CardKind) -> dict[str, Any]:
    base = dict(row)
    base["kind"] = kind
    base["facts"] = _loads(base.pop("facts_json", "[]"))
    base["style_tags"] = _loads(base.pop("style_tags_json", "[]"))
    return base


def _refresh_fts_row(
    conn: sqlite3.Connection,
    *,
    table: str,
    row_id: int,
    values: tuple[Any, ...],
) -> None:
    """Replace one FTS5 row without depending on external triggers."""

    try:
        conn.execute(f"DELETE FROM {table} WHERE rowid = ?", (row_id,))
    except sqlite3.OperationalError:
        # Some SQLite builds are picky about direct FTS row deletes.  The table
        # is small and offline-maintained, so a failed delete should not break
        # the whole enrichment path; the following INSERT OR REPLACE keeps the
        # latest row visible for normal queries.
        pass
    placeholders = ", ".join("?" for _ in range(len(values) + 1))
    columns = {
        "artist_cards_fts": "rowid, artist, summary, style_tags, facts",
        "song_cards_fts": "rowid, title, artist, summary, style_tags, facts",
    }[table]
    conn.execute(
        f"INSERT OR REPLACE INTO {table}({columns}) VALUES ({placeholders})",
        (row_id, *values),
    )


class MusicKnowledgeStore:
    """Small SQLite store for song/artist knowledge cards."""

    def __init__(self, path: str | Path | None = None):
        self.path = resolve_store_path(path)

    def connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def initialize(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sources (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    url TEXT NOT NULL UNIQUE,
                    title TEXT DEFAULT '',
                    provider TEXT DEFAULT 'web',
                    retrieved_at INTEGER NOT NULL,
                    confidence REAL NOT NULL DEFAULT 0.5
                );

                CREATE TABLE IF NOT EXISTS artist_cards (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    artist TEXT NOT NULL UNIQUE,
                    summary TEXT DEFAULT '',
                    style_tags_json TEXT DEFAULT '[]',
                    facts_json TEXT DEFAULT '[]',
                    source_id INTEGER,
                    confidence REAL NOT NULL DEFAULT 0.5,
                    updated_at INTEGER NOT NULL,
                    FOREIGN KEY(source_id) REFERENCES sources(id) ON DELETE SET NULL
                );

                CREATE TABLE IF NOT EXISTS song_cards (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    artist TEXT DEFAULT '',
                    summary TEXT DEFAULT '',
                    release_year INTEGER,
                    style_tags_json TEXT DEFAULT '[]',
                    facts_json TEXT DEFAULT '[]',
                    source_id INTEGER,
                    confidence REAL NOT NULL DEFAULT 0.5,
                    updated_at INTEGER NOT NULL,
                    UNIQUE(title, artist),
                    FOREIGN KEY(source_id) REFERENCES sources(id) ON DELETE SET NULL
                );

                CREATE VIRTUAL TABLE IF NOT EXISTS artist_cards_fts USING fts5(
                    artist, summary, style_tags, facts
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS song_cards_fts USING fts5(
                    title, artist, summary, style_tags, facts
                );
                """
            )

    def upsert_source(
        self,
        *,
        url: str,
        title: str = "",
        provider: str = "web",
        confidence: float = 0.6,
        retrieved_at: int | None = None,
    ) -> int | None:
        url = str(url or "").strip()
        if not url:
            return None
        self.initialize()
        with self.connect() as conn:
            row = conn.execute("SELECT id FROM sources WHERE url = ?", (url,)).fetchone()
            if row:
                conn.execute(
                    """
                    UPDATE sources
                    SET title = COALESCE(NULLIF(?, ''), title),
                        provider = COALESCE(NULLIF(?, ''), provider),
                        confidence = max(confidence, ?),
                        retrieved_at = ?
                    WHERE id = ?
                    """,
                    (title, provider, clamp_confidence(confidence), int(retrieved_at or _now_ms()), row["id"]),
                )
                return int(row["id"])
            cur = conn.execute(
                """
                INSERT INTO sources(url, title, provider, retrieved_at, confidence)
                VALUES (?, ?, ?, ?, ?)
                """,
                (url, title, provider, int(retrieved_at or _now_ms()), clamp_confidence(confidence)),
            )
            return int(cur.lastrowid)

    def upsert_artist_card(
        self,
        *,
        artist: str,
        summary: str = "",
        style_tags: Iterable[str] | None = None,
        facts: Iterable[str] | None = None,
        source_url: str = "",
        source_title: str = "",
        source_provider: str = "web",
        confidence: float = 0.6,
    ) -> dict[str, Any]:
        self.initialize()
        source_id = self.upsert_source(
            url=source_url,
            title=source_title,
            provider=source_provider,
            confidence=confidence,
        )
        confidence = clamp_confidence(confidence)
        facts_json = _dumps(list(facts or [])[:8])
        style_json = _dumps(list(style_tags or [])[:12])
        updated_at = _now_ms()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO artist_cards(artist, summary, style_tags_json, facts_json, source_id, confidence, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(artist) DO UPDATE SET
                    summary = excluded.summary,
                    style_tags_json = excluded.style_tags_json,
                    facts_json = excluded.facts_json,
                    source_id = excluded.source_id,
                    confidence = excluded.confidence,
                    updated_at = excluded.updated_at
                """,
                (artist, summary, style_json, facts_json, source_id, confidence, updated_at),
            )
            row = conn.execute(
                """
                SELECT a.*, so.url AS source_url, so.provider AS source_provider
                FROM artist_cards a
                LEFT JOIN sources so ON so.id = a.source_id
                WHERE a.artist = ?
                """,
                (artist,),
            ).fetchone()
            _refresh_fts_row(
                conn,
                table="artist_cards_fts",
                row_id=int(row["id"]),
                values=(artist, summary, " ".join(style_tags or []), " ".join(facts or [])),
            )
            return _row_to_card(row, "artist")

    def upsert_song_card(
        self,
        *,
        title: str,
        artist: str = "",
        summary: str = "",
        release_year: int | None = None,
        style_tags: Iterable[str] | None = None,
        facts: Iterable[str] | None = None,
        source_url: str = "",
        source_title: str = "",
        source_provider: str = "web",
        confidence: float = 0.6,
    ) -> dict[str, Any]:
        self.initialize()
        source_id = self.upsert_source(
            url=source_url,
            title=source_title,
            provider=source_provider,
            confidence=confidence,
        )
        confidence = clamp_confidence(confidence)
        facts_json = _dumps(list(facts or [])[:8])
        style_json = _dumps(list(style_tags or [])[:12])
        updated_at = _now_ms()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO song_cards(title, artist, summary, release_year, style_tags_json, facts_json, source_id, confidence, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(title, artist) DO UPDATE SET
                    summary = excluded.summary,
                    release_year = excluded.release_year,
                    style_tags_json = excluded.style_tags_json,
                    facts_json = excluded.facts_json,
                    source_id = excluded.source_id,
                    confidence = excluded.confidence,
                    updated_at = excluded.updated_at
                """,
                (title, artist, summary, release_year, style_json, facts_json, source_id, confidence, updated_at),
            )
            row = conn.execute(
                """
                SELECT s.*, so.url AS source_url, so.provider AS source_provider
                FROM song_cards s
                LEFT JOIN sources so ON so.id = s.source_id
                WHERE s.title = ? AND s.artist = ?
                """,
                (title, artist),
            ).fetchone()
            _refresh_fts_row(
                conn,
                table="song_cards_fts",
                row_id=int(row["id"]),
                values=(title, artist, summary, " ".join(style_tags or []), " ".join(facts or [])),
            )
            return _row_to_card(row, "song")

    def get_artist_card(self, artist: str) -> dict[str, Any] | None:
        self.initialize()
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT a.*, so.url AS source_url, so.provider AS source_provider
                FROM artist_cards a
                LEFT JOIN sources so ON so.id = a.source_id
                WHERE a.artist = ?
                """,
                (artist,),
            ).fetchone()
            return _row_to_card(row, "artist") if row else None

    def get_song_card(self, title: str, artist: str = "") -> dict[str, Any] | None:
        self.initialize()
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT s.*, so.url AS source_url, so.provider AS source_provider
                FROM song_cards s
                LEFT JOIN sources so ON so.id = s.source_id
                WHERE s.title = ? AND s.artist = ?
                """,
                (title, artist),
            ).fetchone()
            if not row and artist:
                row = conn.execute(
                    """
                    SELECT s.*, so.url AS source_url, so.provider AS source_provider
                    FROM song_cards s
                    LEFT JOIN sources so ON so.id = s.source_id
                    WHERE s.title = ?
                    ORDER BY s.confidence DESC
                    LIMIT 1
                    """,
                    (title,),
                ).fetchone()
            return _row_to_card(row, "song") if row else None

    def search(self, query: str, *, kind: CardKind | None = None, limit: int = 5, min_confidence: float = 0.0) -> list[dict[str, Any]]:
        self.initialize()
        query = str(query or "").strip()
        if not query:
            return []
        fts_query = _clean_query(query)
        sqlite_rows: list[dict[str, Any]] = []
        with self.connect() as conn:
            if kind in (None, "artist"):
                sqlite_rows.extend(self._search_artist(conn, query, fts_query, limit, min_confidence))
            if kind in (None, "song"):
                sqlite_rows.extend(self._search_song(conn, query, fts_query, limit, min_confidence))
        vector_rows: list[dict[str, Any]] = []
        if str(getattr(settings, "knowledge_vector_backend", "")).casefold() == "qdrant":
            try:
                from services.knowledge_vector_index import search_qdrant_knowledge

                vector_rows = search_qdrant_knowledge(
                    query,
                    kind=kind,
                    limit=limit,
                    min_confidence=min_confidence,
                )
            except Exception:
                vector_rows = []
        sqlite_rows.sort(key=lambda row: (-float(row.get("confidence") or 0), row.get("kind", ""), row.get("title") or row.get("artist") or ""))
        vector_rows.sort(key=lambda row: (-float(row.get("_vector_score") or 0), -float(row.get("confidence") or 0)))
        return _dedupe_cards([*sqlite_rows, *vector_rows])[: max(1, int(limit))]

    def _search_artist(self, conn: sqlite3.Connection, query: str, fts_query: str, limit: int, min_confidence: float) -> list[dict[str, Any]]:
        try:
            if fts_query:
                found = conn.execute(
                    """
                    SELECT a.*, so.url AS source_url, so.provider AS source_provider
                    FROM artist_cards_fts f
                    JOIN artist_cards a ON a.id = f.rowid
                    LEFT JOIN sources so ON so.id = a.source_id
                    WHERE artist_cards_fts MATCH ? AND a.confidence >= ?
                    LIMIT ?
                    """,
                    (fts_query, min_confidence, limit),
                ).fetchall()
                if found:
                    return [_row_to_card(row, "artist") for row in found]
        except sqlite3.OperationalError:
            pass
        like = f"%{query}%"
        return [
            _row_to_card(row, "artist")
            for row in conn.execute(
                """
                SELECT a.*, so.url AS source_url, so.provider AS source_provider
                FROM artist_cards a
                LEFT JOIN sources so ON so.id = a.source_id
                WHERE a.confidence >= ? AND (a.artist LIKE ? OR a.summary LIKE ? OR a.facts_json LIKE ? OR a.style_tags_json LIKE ?)
                LIMIT ?
                """,
                (min_confidence, like, like, like, like, limit),
            ).fetchall()
        ]

    def _search_song(self, conn: sqlite3.Connection, query: str, fts_query: str, limit: int, min_confidence: float) -> list[dict[str, Any]]:
        try:
            if fts_query:
                found = conn.execute(
                    """
                    SELECT s.*, so.url AS source_url, so.provider AS source_provider
                    FROM song_cards_fts f
                    JOIN song_cards s ON s.id = f.rowid
                    LEFT JOIN sources so ON so.id = s.source_id
                    WHERE song_cards_fts MATCH ? AND s.confidence >= ?
                    LIMIT ?
                    """,
                    (fts_query, min_confidence, limit),
                ).fetchall()
                if found:
                    return [_row_to_card(row, "song") for row in found]
        except sqlite3.OperationalError:
            pass
        like = f"%{query}%"
        return [
            _row_to_card(row, "song")
            for row in conn.execute(
                """
                SELECT s.*, so.url AS source_url, so.provider AS source_provider
                FROM song_cards s
                LEFT JOIN sources so ON so.id = s.source_id
                WHERE s.confidence >= ? AND (s.title LIKE ? OR s.artist LIKE ? OR s.summary LIKE ? OR s.facts_json LIKE ? OR s.style_tags_json LIKE ?)
                LIMIT ?
                """,
                (min_confidence, like, like, like, like, like, limit),
            ).fetchall()
        ]

    def upsert_normalized_card(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        card = normalize_knowledge_card(payload)
        if card["kind"] == "artist":
            return self.upsert_artist_card(
                artist=card.get("artist") or card.get("title") or "",
                summary=card.get("summary", ""),
                style_tags=card.get("style_tags") or [],
                facts=card.get("facts", []),
                source_url=card.get("source_url", ""),
                source_provider=card.get("source", "web"),
                confidence=float(card.get("confidence") or 0.5),
            )
        return self.upsert_song_card(
            title=card.get("title", ""),
            artist=card.get("artist", ""),
            summary=card.get("summary", ""),
            release_year=card.get("release_year"),
            style_tags=card.get("style_tags") or [],
            facts=card.get("facts", []),
            source_url=card.get("source_url", ""),
            source_provider=card.get("source", "web"),
            confidence=float(card.get("confidence") or 0.5),
        )

    def candidate_queries_for_card(self, *, kind: CardKind, title: str = "", artist: str = "") -> list[str]:
        if kind == "artist":
            return [build_artist_knowledge_query(artist or title)]
        return [build_song_knowledge_query(title, artist)]
