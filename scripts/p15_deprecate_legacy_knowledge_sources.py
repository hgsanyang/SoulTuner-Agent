"""Deprecate legacy knowledge-card sources without deleting audit history."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.music_knowledge_store import MusicKnowledgeStore  # noqa: E402


def run(provider: str, *, dry_run: bool = False) -> dict[str, int | bool | str]:
    store = MusicKnowledgeStore()
    if dry_run:
        store.initialize()
        with store.connect() as conn:
            row = conn.execute(
                """
                SELECT count(*) AS n
                FROM sources so
                LEFT JOIN song_cards sc ON sc.source_id = so.id
                LEFT JOIN artist_cards ac ON ac.source_id = so.id
                WHERE so.provider = ?
                """,
                (provider,),
            ).fetchone()
        return {"provider": provider, "dry_run": True, "matches": int(row["n"] or 0)}
    result = store.deprecate_source_provider(provider)
    return {"provider": provider, "dry_run": False, **result}


def main() -> None:
    parser = argparse.ArgumentParser(description="Deprecate legacy music knowledge source providers.")
    parser.add_argument("--provider", default="Tavily_AI_Answer")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.provider, dry_run=args.dry_run), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
