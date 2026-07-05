"""Sync optional music knowledge cards into Neo4j as lightweight graph nodes."""

from __future__ import annotations

import json
from typing import Any, Mapping

from services.catalog_enrichment import normalize_knowledge_card
from services.music_knowledge_cache import knowledge_key


def knowledge_card_params(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a knowledge card into safe Neo4j parameters."""

    card = normalize_knowledge_card(payload)
    key = knowledge_key(card["kind"], card.get("title", ""), card.get("artist", ""))
    return {
        "key": key,
        "kind": card["kind"],
        "title": card.get("title", ""),
        "artist": card.get("artist", ""),
        "summary": card.get("summary", ""),
        "facts_json": json.dumps(card.get("facts") or [], ensure_ascii=False, sort_keys=True),
        "source": card.get("source", "web"),
        "source_url": card.get("source_url", ""),
        "confidence": card.get("confidence", 0.5),
    }


UPSERT_KNOWLEDGE_CARD_CYPHER = """
MERGE (k:KnowledgeCard {key: $key})
SET k.kind = $kind,
    k.title = $title,
    k.artist = $artist,
    k.summary = $summary,
    k.facts_json = $facts_json,
    k.source = $source,
    k.source_url = $source_url,
    k.confidence = $confidence,
    k.updated_at = timestamp()
WITH k
FOREACH (_ IN CASE WHEN $kind = 'artist' AND $artist <> '' THEN [1] ELSE [] END |
    MERGE (a:Artist {name: $artist})
    MERGE (a)-[:HAS_KNOWLEDGE]->(k)
)
WITH k
OPTIONAL MATCH (s:Song)
WHERE $kind = 'song'
  AND s.title = $title
  AND ($artist = '' OR coalesce(s.artist, '') = $artist OR EXISTS {
      MATCH (s)-[:PERFORMED_BY]->(a:Artist {name: $artist})
  })
WITH k, s
FOREACH (_ IN CASE WHEN s IS NULL THEN [] ELSE [1] END |
    MERGE (s)-[:HAS_KNOWLEDGE]->(k)
)
RETURN k.key AS key
"""


def upsert_knowledge_card_to_neo4j(client: Any, payload: Mapping[str, Any]) -> str:
    """Write one normalized knowledge card to Neo4j and return its key."""

    params = knowledge_card_params(payload)
    client.execute_query(UPSERT_KNOWLEDGE_CARD_CYPHER, params)
    return str(params["key"])
