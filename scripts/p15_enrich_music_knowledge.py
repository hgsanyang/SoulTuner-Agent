"""Offline web enrichment for artist/song knowledge cards.

This script is intentionally offline-batch oriented: it may call web search, but
the recommendation hot path only reads the resulting local SQLite store.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.neo4j_client import get_neo4j_client  # noqa: E402
from services.music_knowledge_enrichment import enrich_artist_card, enrich_song_card  # noqa: E402
from services.music_knowledge_store import MusicKnowledgeStore  # noqa: E402


def load_seed_songs(limit: int = 20) -> list[dict[str, str]]:
    client = get_neo4j_client()
    rows = client.execute_query(
        """
        MATCH (s:Song)
        WHERE coalesce(s.title, '') <> ''
        OPTIONAL MATCH (s)-[:PERFORMED_BY]->(a:Artist)
        WITH s, coalesce(s.artist, a.name, '') AS artist
        RETURN s.title AS title, artist
        ORDER BY coalesce(s.updated_at, 0) DESC
        LIMIT $limit
        """,
        {"limit": int(limit)},
    )
    return [{"title": row.get("title", ""), "artist": row.get("artist", "")} for row in rows]


def load_seed_artists(limit: int = 20) -> list[str]:
    client = get_neo4j_client()
    rows = client.execute_query(
        """
        MATCH (a:Artist)
        WHERE coalesce(a.name, '') <> ''
        RETURN a.name AS artist
        ORDER BY a.name
        LIMIT $limit
        """,
        {"limit": int(limit)},
    )
    return [row["artist"] for row in rows if row.get("artist")]


async def run(args: argparse.Namespace) -> dict:
    store = MusicKnowledgeStore(args.store_path) if args.store_path else MusicKnowledgeStore()
    store.initialize()
    cards: list[dict] = []

    if args.artist:
        card = await enrich_artist_card(args.artist, store=store, dry_run=args.dry_run)
        if card:
            cards.append(card)
    if args.song:
        title, _, artist = args.song.partition("::")
        card = await enrich_song_card(title.strip(), artist.strip(), store=store, dry_run=args.dry_run)
        if card:
            cards.append(card)
    if args.from_neo4j_artists:
        for artist in load_seed_artists(limit=args.limit):
            card = await enrich_artist_card(artist, store=store, dry_run=args.dry_run)
            if card:
                cards.append(card)
    if args.from_neo4j_songs:
        for song in load_seed_songs(limit=args.limit):
            card = await enrich_song_card(song["title"], song.get("artist", ""), store=store, dry_run=args.dry_run)
            if card:
                cards.append(card)

    return {
        "store_path": str(store.path),
        "cards": len(cards),
        "dry_run": args.dry_run,
        "items": [
            {
                "kind": card.get("kind"),
                "title": card.get("title"),
                "artist": card.get("artist"),
                "confidence": card.get("confidence"),
                "source_url": card.get("source_url"),
                "style_tags": card.get("style_tags", []),
                "release_year": card.get("release_year"),
            }
            for card in cards
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build offline music knowledge cards from web search.")
    parser.add_argument("--artist", default="", help="Enrich one artist card")
    parser.add_argument("--song", default="", help="Enrich one song as 'title::artist'")
    parser.add_argument("--from-neo4j-artists", action="store_true", help="Batch enrich artists from Neo4j")
    parser.add_argument("--from-neo4j-songs", action="store_true", help="Batch enrich songs from Neo4j")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--store-path", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(json.dumps(asyncio.run(run(args)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
