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
from typing import Any, Awaitable, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.neo4j_client import get_neo4j_client  # noqa: E402
from services.knowledge_vector_index import upsert_cards_to_qdrant  # noqa: E402
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
        WHERE artist <> ''
        OPTIONAL MATCH (s)-[:HAS_KNOWLEDGE]->(k:KnowledgeCard)
        WITH s.title AS title,
             artist,
             count(DISTINCT s) AS catalog_count,
             max(coalesce(s.updated_at, 0)) AS updated_at,
             min(CASE WHEN s.release_year IS NULL THEN 0 ELSE 1 END) AS has_release_year,
             count(k) AS knowledge_cards
        WHERE knowledge_cards = 0
        RETURN title, artist
        ORDER BY has_release_year ASC, catalog_count DESC, updated_at DESC, title
        LIMIT $limit
        """,
        {"limit": int(limit)},
    )
    return [{"title": row.get("title", ""), "artist": row.get("artist", "")} for row in rows]


def load_seed_artists(limit: int = 20) -> list[str]:
    client = get_neo4j_client()
    rows = client.execute_query(
        """
        MATCH (a:Artist)<-[:PERFORMED_BY]-(s:Song)
        WHERE coalesce(a.name, '') <> ''
        OPTIONAL MATCH (a)-[:HAS_KNOWLEDGE]->(k:KnowledgeCard)
        WITH a.name AS artist, count(DISTINCT s) AS catalog_count, count(k) AS knowledge_cards
        WHERE knowledge_cards = 0
        RETURN artist
        ORDER BY catalog_count DESC, artist
        LIMIT $limit
        """,
        {"limit": int(limit)},
    )
    return [row["artist"] for row in rows if row.get("artist")]


async def run(args: argparse.Namespace) -> dict:
    store = MusicKnowledgeStore(args.store_path) if args.store_path else MusicKnowledgeStore()
    store.initialize()
    cards: list[dict] = []
    errors: list[dict[str, str]] = []

    async def _run_batch(
        labels: list[str],
        builders: list[Callable[[], Awaitable[dict | None]]],
    ) -> None:
        semaphore = asyncio.Semaphore(max(1, int(args.concurrency)))

        async def _one(label: str, builder: Callable[[], Awaitable[dict | None]]) -> dict[str, Any]:
            async with semaphore:
                try:
                    return {"label": label, "card": await builder()}
                except Exception as exc:
                    return {"label": label, "error": str(exc)[:240]}

        for result in await asyncio.gather(*(_one(label, builder) for label, builder in zip(labels, builders))):
            card = result.get("card")
            if card:
                cards.append(card)
            if result.get("error"):
                errors.append({"item": str(result.get("label") or ""), "error": str(result["error"])})

    if args.artist:
        card = await enrich_artist_card(
            args.artist,
            store=store,
            dry_run=args.dry_run,
            use_llm_summary=args.use_llm_summary,
        )
        if card:
            cards.append(card)
    if args.song:
        title, _, artist = args.song.partition("::")
        card = await enrich_song_card(
            title.strip(),
            artist.strip(),
            store=store,
            dry_run=args.dry_run,
            use_llm_summary=args.use_llm_summary,
        )
        if card:
            cards.append(card)
    if args.from_neo4j_artists:
        artists = load_seed_artists(limit=args.limit)
        await _run_batch(
            artists,
            [
                (
                    lambda artist=artist: enrich_artist_card(
                        artist,
                        store=store,
                        dry_run=args.dry_run,
                        use_llm_summary=args.use_llm_summary,
                    )
                )
                for artist in artists
            ],
        )
    if args.from_neo4j_songs:
        songs = load_seed_songs(limit=args.limit)
        await _run_batch(
            [f"{song['title']}::{song.get('artist', '')}" for song in songs],
            [
                (
                    lambda song=song: enrich_song_card(
                        song["title"],
                        song.get("artist", ""),
                        store=store,
                        dry_run=args.dry_run,
                        use_llm_summary=args.use_llm_summary,
                    )
                )
                for song in songs
            ],
        )

    result = {
        "store_path": str(store.path),
        "cards": len(cards),
        "dry_run": args.dry_run,
        "llm_summary": bool(args.use_llm_summary),
        "qdrant_synced": 0,
        "errors": errors,
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
    if cards and args.sync_qdrant and not args.dry_run:
        try:
            qdrant_result = upsert_cards_to_qdrant(cards)
            result["qdrant_synced"] = int(qdrant_result.get("upserted") or 0)
            result["qdrant_collection"] = qdrant_result.get("collection", "")
        except Exception as exc:
            result["qdrant_error"] = str(exc)[:200]
    return result


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    parser = argparse.ArgumentParser(description="Build offline music knowledge cards from web search.")
    parser.add_argument("--artist", default="", help="Enrich one artist card")
    parser.add_argument("--song", default="", help="Enrich one song as 'title::artist'")
    parser.add_argument("--from-neo4j-artists", action="store_true", help="Batch enrich artists from Neo4j")
    parser.add_argument("--from-neo4j-songs", action="store_true", help="Batch enrich songs from Neo4j")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=1, help="Batch concurrency for Neo4j seed enrichment")
    parser.add_argument("--store-path", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.set_defaults(use_llm_summary=True)
    parser.add_argument(
        "--use-llm-summary",
        action="store_true",
        help="Use DashScope/Qwen web search to build sourced knowledge cards. This is the default.",
    )
    parser.add_argument(
        "--no-llm-summary",
        action="store_false",
        dest="use_llm_summary",
        help="Disable DashScope/Qwen and fall back to external search snippets plus deterministic summarization.",
    )
    parser.add_argument("--sync-qdrant", action="store_true", help="Mirror generated cards into Qdrant.")
    args = parser.parse_args()
    print(json.dumps(asyncio.run(run(args)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
