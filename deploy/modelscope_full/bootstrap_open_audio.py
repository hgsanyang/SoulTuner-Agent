"""Idempotently attach the licensed five-track demo to the full Space graph.

The ModelScope dataset is materialised by ``start_full_space.sh``.  This helper
reuses SoulTuner's normal quick-ingest and enrichment worker instead of creating
a deployment-only indexing path.  A fully enriched external graph is left
untouched on later starts.
"""

from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path

from data.pipeline.neo4j_schema_v2 import create_vector_indexes
from scripts.ingest_worker import process_one
from tools.data.queue_song_describer import commit_to_existing_flywheel, load_ingest_songs

EXPECTED_DIMS = {"muq_dim": 512, "m2d_dim": 768, "omar_dim": 1024}


def _enrichment_status(song_ids: list[str]) -> dict[str, dict[str, object]]:
    from retrieval.neo4j_client import get_neo4j_client

    rows = get_neo4j_client().execute_query(
        """
        MATCH (s:Song)
        WHERE toString(s.music_id) IN $song_ids
        RETURN toString(s.music_id) AS song_id,
               coalesce(size(s.muq_embedding), 0) AS muq_dim,
               coalesce(size(s.m2d2_embedding), 0) AS m2d_dim,
               coalesce(size(s.omar_embedding), 0) AS omar_dim,
               coalesce(s.enrichment_status, '') AS enrichment_status
        """,
        {"song_ids": song_ids},
    )
    return {str(row["song_id"]): dict(row) for row in rows}


def _is_ready(row: dict[str, object] | None) -> bool:
    if not row or str(row.get("enrichment_status") or "") != "ready":
        return False
    return all(int(row.get(name) or 0) == expected for name, expected in EXPECTED_DIMS.items())


def _ensure_vector_indexes(timeout_seconds: float = 120.0) -> None:
    """Create and wait for the three indexes used by the retrieval runtime."""

    from retrieval.neo4j_client import get_neo4j_client

    required = {"song_muq_index", "song_m2d2_index", "song_omar_index"}
    create_vector_indexes()
    deadline = time.monotonic() + timeout_seconds
    last_states: dict[str, str] = {}
    while time.monotonic() < deadline:
        rows = get_neo4j_client().execute_query(
            """
            SHOW INDEXES YIELD name, type, state
            WHERE type = 'VECTOR' AND name IN $names
            RETURN name, state
            """,
            {"names": sorted(required)},
        )
        last_states = {str(row["name"]): str(row["state"]) for row in rows}
        if all(last_states.get(name) == "ONLINE" for name in required):
            return
        time.sleep(1)
    raise RuntimeError(f"SoulTuner vector indexes did not become ONLINE: {last_states}")


async def bootstrap(cache_dir: Path, manifest: Path, batch_size: int) -> None:
    songs = load_ingest_songs(manifest, cache_dir)
    song_ids = [str(song["song_id"]) for song in songs]
    current = _enrichment_status(song_ids)
    missing = [song for song in songs if not _is_ready(current.get(str(song["song_id"])))]
    if not missing:
        _ensure_vector_indexes()
        print(f"SoulTuner open-audio graph already ready: {len(songs)} tracks", flush=True)
        return

    await commit_to_existing_flywheel(missing, batch_size=batch_size)
    while await process_one():
        pass

    final = _enrichment_status(song_ids)
    failed = [song_id for song_id in song_ids if not _is_ready(final.get(song_id))]
    if failed:
        raise RuntimeError(f"open-audio enrichment incomplete: {failed}")
    _ensure_vector_indexes()
    print(f"SoulTuner open-audio graph bootstrapped: {len(songs)} tracks", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap the licensed SoulTuner demo catalog")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=5)
    args = parser.parse_args()
    asyncio.run(
        bootstrap(
            args.cache_dir.expanduser().resolve(),
            args.manifest.expanduser().resolve(),
            max(1, args.batch_size),
        )
    )


if __name__ == "__main__":
    main()
