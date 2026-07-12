"""Backfill zero-shot MuQ acoustic probe fields into Neo4j Song nodes."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.acoustic_probe import PROBE_VERSION, build_probe_text_embeddings, score_catalog_embeddings
from retrieval.neo4j_client import get_neo4j_client


def _fetch_song_embeddings(limit: int = 0) -> list[dict]:
    client = get_neo4j_client()
    query = """
    MATCH (s:Song)
    WHERE s.muq_embedding IS NOT NULL
      AND size(s.muq_embedding) = 512
      AND coalesce(properties(s)['unplayable_stub'], false) <> true
    RETURN elementId(s) AS eid, s.title AS title, s.artist AS artist, s.muq_embedding AS muq_embedding
    ORDER BY coalesce(s.updated_at, 0) DESC, s.title
    """ + ("\nLIMIT $limit" if limit and limit > 0 else "")
    return client.execute_query(query, {"limit": int(limit)}) or []


def backfill_acoustic_probe(*, limit: int = 0, dry_run: bool = False) -> dict:
    rows = _fetch_song_embeddings(limit)
    song_embeddings = {
        str(row["eid"]): row["muq_embedding"]
        for row in rows
        if row.get("eid") and row.get("muq_embedding")
    }
    if not song_embeddings:
        return {"songs": 0, "updated": 0, "dry_run": dry_run}

    from retrieval.muq_embedder import encode_text_to_muq

    probe_embeddings = build_probe_text_embeddings(encode_text_to_muq)
    scores = score_catalog_embeddings(song_embeddings, probe_embeddings)

    updates = []
    row_by_eid = {str(row["eid"]): row for row in rows}
    now_ms = int(time.time() * 1000)
    for eid, fields in scores.items():
        if not fields:
            continue
        row = row_by_eid[eid]
        updates.append(
            {
                "eid": eid,
                "title": row.get("title"),
                "artist": row.get("artist"),
                "fields": fields,
                "version": PROBE_VERSION,
                "updated_at": now_ms,
            }
        )

    if not dry_run and updates:
        client = get_neo4j_client()
        client.execute_query(
            """
            UNWIND $updates AS item
            MATCH (s:Song)
            WHERE elementId(s) = item.eid
            SET s.acoustic_vocalness = item.fields.acoustic_vocalness,
                s.acoustic_instrumentalness = item.fields.acoustic_instrumentalness,
                s.acoustic_drumness = item.fields.acoustic_drumness,
                s.acoustic_energy = item.fields.acoustic_energy,
                s.acoustic_low_energy = item.fields.acoustic_low_energy,
                s.acoustic_probe_version = item.version,
                s.acoustic_probe_updated_at = item.updated_at
            """,
            {"updates": updates},
        )

    sample = updates[:5]
    return {
        "songs": len(song_embeddings),
        "updated": len(updates),
        "dry_run": dry_run,
        "version": PROBE_VERSION,
        "sample": sample,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0, help="Limit number of songs for smoke tests")
    parser.add_argument("--dry-run", action="store_true", help="Compute scores without writing Neo4j")
    args = parser.parse_args()
    result = backfill_acoustic_probe(limit=args.limit, dry_run=args.dry_run)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
