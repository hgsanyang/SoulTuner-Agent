"""Backfill the canonical SQLite store from the legacy JSONL feedback logs.

SQLite became the system of record after the JSONL logs already existed, so an
install that has been collecting feedback needs its history imported once. This
is idempotent: exposures upsert by id (the provisional/final pair collapses to
one row), and events INSERT OR REPLACE by their own id.

Run a dry run first — it reports what WOULD be written without touching the db:

    python -m scripts.migrate_feedback_jsonl_to_sqlite --dry-run
    python -m scripts.migrate_feedback_jsonl_to_sqlite            # actually write

Honours MUSIC_FEEDBACK_DIR, same as the rest of the feedback layer.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services import feedback_store as fs  # noqa: E402
from services.feedback_logger import (  # noqa: E402
    SLATE_FEEDBACK_FILE,
    _dedupe_by,
    _feedback_dir,
    load_jsonl,
)

# (jsonl file, dedupe key or None, store insert fn) — order matters: exposures
# first so anything keyed to them exists before the events land.
SOURCES = [
    ("exposures.jsonl", "exposure_id", "upsert_exposure"),
    ("song_feedback.jsonl", "song_feedback_id", "insert_song_feedback"),
    (SLATE_FEEDBACK_FILE, "slate_feedback_id", "insert_slate_feedback"),
    ("events.jsonl", "event_id", "insert_user_event"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="report counts without writing to SQLite")
    args = parser.parse_args()

    root = _feedback_dir()
    print(f"feedback dir: {root}")
    print(f"sqlite store: {fs.store_path()}")
    print(f"mode: {'DRY RUN (no writes)' if args.dry_run else 'WRITE'}\n")

    before = {} if args.dry_run else fs.counts()
    total_seen = 0
    for filename, key, fn_name in SOURCES:
        path = root / filename
        rows = load_jsonl(path)
        deduped = _dedupe_by(rows, key) if key else rows
        total_seen += len(deduped)
        collapsed = len(rows) - len(deduped)
        note = f" ({collapsed} provisional/dup rows collapse)" if collapsed else ""
        print(f"{filename:<24} {len(rows):>6} rows -> {len(deduped):>6} unique{note}")
        if args.dry_run:
            continue
        insert = getattr(fs, fn_name)
        for row in deduped:
            insert(row)

    if args.dry_run:
        print(f"\nwould write {total_seen} unique records. Re-run without --dry-run.")
        return 0

    after = fs.counts()
    print("\nSQLite counts (before -> after):")
    for table in ("exposures", "song_feedback", "slate_feedback", "user_events"):
        print(f"  {table:<16} {before.get(table, 0):>6} -> {after.get(table, 0):>6}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
