"""Backfill the canonical SQLite store from the legacy JSONL feedback logs.

SQLite became the system of record after the JSONL logs already existed, so an
install that has been collecting feedback needs its history imported once. This
is idempotent: exposures upsert by id (the provisional/final pair collapses to
one row), and events INSERT OR REPLACE by their own id.

Run a dry run first - it reports what WOULD be written without touching the db:

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
    ID_KEYS,
    SLATE_FEEDBACK_FILE,
    _dedupe_rows,
    _feedback_dir,
    effective_id,
    load_jsonl,
)

# (jsonl file, table, store insert fn) - exposures first so anything keyed to
# them exists before the events land. `table` selects the effective-id keys, so
# legacy slate rows (feedback_id only) are keyed correctly, not dropped.
SOURCES = [
    ("exposures.jsonl", "exposures", "upsert_exposure"),
    ("song_feedback.jsonl", "song_feedback", "insert_song_feedback"),
    (SLATE_FEEDBACK_FILE, "slate_feedback", "insert_slate_feedback"),
    ("events.jsonl", "events", "insert_user_event"),
]

TABLE_TO_COUNT = {
    "exposures": "exposures",
    "song_feedback": "song_feedback",
    "slate_feedback": "slate_feedback",
    "events": "user_events",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="report counts without writing to SQLite")
    args = parser.parse_args()

    root = _feedback_dir()
    print(f"feedback dir: {root}")
    print(f"sqlite store: {fs.store_path()}")
    print(f"mode: {'DRY RUN - no writes' if args.dry_run else 'WRITE'}\n")

    before = fs.counts()
    plan: list[tuple[str, str, list[dict], int]] = []
    for filename, table, fn_name in SOURCES:
        rows = load_jsonl(root / filename)
        id_keys = ID_KEYS[table]
        deduped = _dedupe_rows(rows, id_keys)
        # Unique records that MUST exist after migrating = (already in SQLite) UNION
        # (in the log). The gate below asserts the store reaches exactly this.
        existing_ids = {effective_id(r, id_keys) for r in _load_existing(table)}
        log_ids = {effective_id(r, id_keys) for r in deduped}
        expected_unique = len((existing_ids | log_ids) - {""})
        collapsed = len(rows) - len(deduped)
        note = f" ({collapsed} provisional/dup collapse)" if collapsed else ""
        print(f"{filename:<22} log {len(rows):>5} -> {len(deduped):>5} unique{note}; "
              f"store has {len(existing_ids - {''}):>5}; expect == {expected_unique}")
        plan.append((table, fn_name, deduped, expected_unique))

    if args.dry_run:
        print("\nDRY RUN - re-run without --dry-run to write. Migration FAILS unless "
              "each table reaches exactly its expected unique count.")
        return 0

    # Idempotent, recoverable (NOT one transaction): every insert is an upsert
    # keyed by effective id, so re-running converges and a crash mid-run can be
    # re-run safely. The post-write gate is what proves completeness.
    for table, fn_name, deduped, _expected in plan:
        insert = getattr(fs, fn_name)
        for row in deduped:
            insert(row)

    after = fs.counts()
    ok = True
    print("\nSQLite counts (before -> after / expected):")
    for table, _fn, _rows, expected in [(p[0], p[1], p[2], p[3]) for p in plan]:
        col = TABLE_TO_COUNT[table]
        b, a = before.get(col, 0), after.get(col, 0)
        match = (a == expected)
        flag = "" if match else "  [MISMATCH]"
        print(f"  {col:<16} {b:>6} -> {a:>6} / {expected:>6}{flag}")
        if not match:
            ok = False
    if not ok:
        print("\nFAILED: a table did not reach its expected unique count. "
              "The store is incomplete - do not trust it. Re-run after investigating.")
        return 1
    print("\nOK - every table reached its expected unique count.")
    return 0


def _load_existing(table: str) -> list[dict]:
    loader = {
        "exposures": fs.load_exposures,
        "slate_feedback": fs.load_slate_feedback,
        "song_feedback": fs.load_song_feedback,
        "events": fs.load_events,
    }.get(table)
    return loader() if loader else []


if __name__ == "__main__":
    raise SystemExit(main())
