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


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync local music knowledge cache into Neo4j.")
    parser.add_argument("--cache-path", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cache = MusicKnowledgeCache(args.cache_path) if args.cache_path else MusicKnowledgeCache()
    cards = list(cache.load_all().values())
    result = {"cache_path": str(cache.path), "cards": len(cards), "synced": 0, "dry_run": args.dry_run}
    if not args.dry_run and cards:
        client = get_neo4j_client()
        for card in cards:
            upsert_knowledge_card_to_neo4j(client, card)
            result["synced"] += 1
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
