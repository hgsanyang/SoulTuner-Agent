#!/usr/bin/env python
"""Stamp ``catalog_tier`` on Song nodes written before the property existed.

Reads stay correct without this — :func:`services.catalog_tier.candidate_predicate`
infers the tier from ``source`` + ``audio_retention``. The backfill just makes the
data say plainly what the query was inferring, so the next person reading the
graph is not left guessing which of two rules applies.

Only writes a property; deletes nothing. Idempotent.

    python scripts/backfill_catalog_tier.py            # report only
    python scripts/backfill_catalog_tier.py --apply    # write
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.catalog_tier import CANDIDATE, LIBRARY, candidate_predicate  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="write instead of reporting")
    args = parser.parse_args()

    from retrieval.neo4j_client import get_neo4j_client

    client = get_neo4j_client()
    predicate = candidate_predicate()

    counts = client.execute_query(
        f"""
        MATCH (s:Song)
        WHERE coalesce(s.catalog_tier, '') = ''
        RETURN sum(CASE WHEN {predicate} THEN 1 ELSE 0 END) AS candidate,
               sum(CASE WHEN {predicate} THEN 0 ELSE 1 END) AS library
        """,
        {},
    )
    row = counts[0] if counts else {}
    to_candidate = int(row.get("candidate") or 0)
    to_library = int(row.get("library") or 0)
    print(f"unstamped: {to_candidate} -> {CANDIDATE}, {to_library} -> {LIBRARY}")

    if not args.apply:
        print("dry run; pass --apply to write")
        return 0

    # Order matters: stamp the candidates first. Once catalog_tier is set the
    # predicate short-circuits on it, so a second pass cannot reclassify them.
    written = client.execute_query(
        f"""
        MATCH (s:Song) WHERE coalesce(s.catalog_tier, '') = '' AND {predicate}
        SET s.catalog_tier = '{CANDIDATE}'
        RETURN count(s) AS n
        """,
        {},
    )
    print(f"{CANDIDATE}: {int((written[0] if written else {}).get('n') or 0)}")

    written = client.execute_query(
        f"""
        MATCH (s:Song) WHERE coalesce(s.catalog_tier, '') = ''
        SET s.catalog_tier = '{LIBRARY}'
        RETURN count(s) AS n
        """,
        {},
    )
    print(f"{LIBRARY}: {int((written[0] if written else {}).get('n') or 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
