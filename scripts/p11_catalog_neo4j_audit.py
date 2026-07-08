"""Audit Neo4j catalog coverage after P11 ingestion."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.neo4j_client import get_neo4j_client  # noqa: E402


def run_audit() -> dict:
    client = get_neo4j_client()
    summary = client.execute_query(
        """
        MATCH (s:Song)
        RETURN count(s) AS total,
               sum(CASE WHEN coalesce(properties(s)['unplayable_stub'], false) <> true
                          AND coalesce(s.audio_url, '') <> '' THEN 1 ELSE 0 END) AS playable,
               sum(CASE WHEN coalesce(properties(s)['unplayable_stub'], false) = true THEN 1 ELSE 0 END) AS unplayable_stub,
               sum(CASE WHEN size(coalesce(s.muq_embedding, [])) = 512 THEN 1 ELSE 0 END) AS muq,
               sum(CASE WHEN size(coalesce(s.m2d2_embedding, [])) = 768 THEN 1 ELSE 0 END) AS m2d,
               sum(CASE WHEN size(coalesce(s.omar_embedding, [])) = 1024 THEN 1 ELSE 0 END) AS omar,
               sum(CASE WHEN coalesce(s.tag_source, '') <> '' THEN 1 ELSE 0 END) AS tag_source,
               sum(CASE WHEN properties(s)['release_year'] IS NOT NULL THEN 1 ELSE 0 END) AS release_year,
               sum(CASE WHEN coalesce(s.artist, '') <> '' THEN 1 ELSE 0 END) AS artist
        """,
        {},
    )[0]
    tags = client.execute_query(
        """
        MATCH (s:Song)
        RETURN sum(CASE WHEN EXISTS { MATCH (s)-[:BELONGS_TO_GENRE]->(:Genre) } THEN 1 ELSE 0 END) AS genres,
               sum(CASE WHEN EXISTS { MATCH (s)-[:HAS_MOOD]->(:Mood) } THEN 1 ELSE 0 END) AS moods,
               sum(CASE WHEN EXISTS { MATCH (s)-[:HAS_THEME]->(:Theme) } THEN 1 ELSE 0 END) AS themes,
               sum(CASE WHEN EXISTS { MATCH (s)-[:FITS_SCENARIO]->(:Scenario) } THEN 1 ELSE 0 END) AS scenarios,
               sum(CASE WHEN EXISTS { MATCH (s)-[:HAS_LANGUAGE]->(:Language) } THEN 1 ELSE 0 END) AS languages
        """,
        {},
    )[0]
    duplicate_title_artist = client.execute_query(
        """
        MATCH (s:Song)
        WITH s.title AS title, coalesce(s.artist, '') AS artist, count(s) AS n
        WHERE title IS NOT NULL AND n > 1
        RETURN title, artist, n
        ORDER BY n DESC, title
        LIMIT 20
        """,
        {},
    )
    duplicate_titles = client.execute_query(
        """
        MATCH (s:Song)
        WITH s.title AS title, count(s) AS n,
             collect({artist: coalesce(s.artist, ''), music_id: coalesce(s.music_id, '')})[0..5] AS examples
        WHERE title IS NOT NULL AND n > 1
        RETURN title, n, examples
        ORDER BY n DESC, title
        LIMIT 20
        """,
        {},
    )
    sources = client.execute_query(
        """
        MATCH (s:Song)
        WITH coalesce(s.source, 'unknown') AS source, count(s) AS n
        RETURN source, n
        ORDER BY n DESC
        """,
        {},
    )
    knowledge = client.execute_query(
        """
        MATCH (s:Song)
        RETURN sum(CASE WHEN EXISTS { MATCH (s)-[:HAS_KNOWLEDGE]->(:KnowledgeCard) } THEN 1 ELSE 0 END) AS songs_with_knowledge,
               sum(CASE WHEN coalesce(s.release_year_source, '') = 'knowledge_card' THEN 1 ELSE 0 END) AS release_year_from_knowledge,
               count { MATCH (:KnowledgeCard {kind: 'song'}) } AS song_cards,
               count { MATCH (:KnowledgeCard {kind: 'artist'}) } AS artist_cards,
               count { MATCH (:KnowledgeCard) } AS total_cards
        """,
        {},
    )[0]
    unplayable = client.execute_query(
        """
        MATCH (s:Song)
        WHERE coalesce(properties(s)['unplayable_stub'], false) = true
           OR coalesce(s.audio_url, '') = ''
        RETURN coalesce(s.title, '') AS title,
               coalesce(s.artist, '') AS artist,
               coalesce(toString(s.music_id), '') AS music_id,
               coalesce(s.source, 'unknown') AS source,
               coalesce(toString(s.source_id), '') AS source_id,
               coalesce(toString(properties(s)['unplayable_stub']), '') AS unplayable_stub,
               coalesce(properties(s)['audio_status'], '') AS audio_status,
               coalesce(properties(s)['acquire_status'], '') AS acquire_status
        ORDER BY source, title
        LIMIT 50
        """,
        {},
    )
    return {
        "summary": summary,
        "tag_relationship_coverage": tags,
        "knowledge_coverage": knowledge,
        "source_breakdown": sources,
        "unplayable_examples": unplayable,
        "duplicate_title_artist_top20": duplicate_title_artist,
        "duplicate_title_top20": duplicate_titles,
    }


def main() -> None:
    print(json.dumps(run_audit(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
