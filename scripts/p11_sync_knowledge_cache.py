"""Sync offline music knowledge cards into Neo4j summary nodes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.neo4j_client import get_neo4j_client  # noqa: E402
from services.music_knowledge_cache import MusicKnowledgeCache  # noqa: E402
from services.music_knowledge_graph import upsert_knowledge_card_to_neo4j  # noqa: E402
from services.music_knowledge_store import MusicKnowledgeStore  # noqa: E402


def load_sqlite_cards(store: MusicKnowledgeStore) -> list[dict]:
    store.initialize()
    cards: list[dict] = []
    with store.connect() as conn:
        for row in conn.execute(
            """
            SELECT a.*, so.url AS source_url, so.provider AS source_provider
            FROM artist_cards a
            LEFT JOIN sources so ON so.id = a.source_id
            ORDER BY a.updated_at DESC
            """
        ).fetchall():
            cards.append(
                {
                    "kind": "artist",
                    "artist": row["artist"],
                    "title": row["artist"],
                    "summary": row["summary"],
                    "facts": json.loads(row["facts_json"] or "[]"),
                    "style_tags": json.loads(row["style_tags_json"] or "[]"),
                    "source": row["source_provider"] or "sqlite",
                    "source_url": row["source_url"] or "",
                    "confidence": row["confidence"],
                }
            )
        for row in conn.execute(
            """
            SELECT s.*, so.url AS source_url, so.provider AS source_provider
            FROM song_cards s
            LEFT JOIN sources so ON so.id = s.source_id
            ORDER BY s.updated_at DESC
            """
        ).fetchall():
            cards.append(
                {
                    "kind": "song",
                    "title": row["title"],
                    "artist": row["artist"],
                    "summary": row["summary"],
                    "release_year": row["release_year"],
                    "facts": json.loads(row["facts_json"] or "[]"),
                    "style_tags": json.loads(row["style_tags_json"] or "[]"),
                    "source": row["source_provider"] or "sqlite",
                    "source_url": row["source_url"] or "",
                    "confidence": row["confidence"],
                }
            )
    return cards


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync local music knowledge cache into Neo4j.")
    parser.add_argument("--cache-path", default="")
    parser.add_argument("--store-path", default="")
    parser.add_argument("--from", dest="source", choices=["sqlite", "jsonl"], default="sqlite")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.source == "jsonl":
        cache = MusicKnowledgeCache(args.cache_path) if args.cache_path else MusicKnowledgeCache()
        cards = list(cache.load_all().values())
        path = str(cache.path)
    else:
        store = MusicKnowledgeStore(args.store_path) if args.store_path else MusicKnowledgeStore()
        cards = load_sqlite_cards(store)
        path = str(store.path)
    result = {"source": args.source, "path": path, "cards": len(cards), "synced": 0, "dry_run": args.dry_run}
    if not args.dry_run and cards:
        client = get_neo4j_client()
        for card in cards:
            upsert_knowledge_card_to_neo4j(client, card)
            result["synced"] += 1
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
