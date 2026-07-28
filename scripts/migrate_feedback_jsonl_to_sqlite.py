"""Backfill the canonical SQLite store from the legacy JSONL feedback logs.

SQLite became the system of record after the JSONL logs already existed, so an
install that has been collecting feedback needs its history imported once. This
is idempotent: exposures upsert by id (the provisional/final pair collapses to
one row), and events INSERT OR REPLACE by their own id.

Rows that predate the runtime-context contract are quarantined as
``legacy_unclassified`` and ``training_eligible=false``. Migration preserves
them for audit; it never guesses whether an old interaction was real use or a
test.

STOP THE BACKEND FIRST. Migration reads the whole JSONL history and asserts the
store reaches an exact row count; a live backend appending new feedback mid-run
moves that target and the gate fails on a store that is actually fine. This
script refuses to run while the API answers on its port (API_PORT, default 8501;
BACKEND_PORT is accepted as an alias) - override at your own risk
with --allow-live-writes).

    docker compose stop backend
    python -m scripts.migrate_feedback_jsonl_to_sqlite --dry-run   # report only
    python -m scripts.migrate_feedback_jsonl_to_sqlite             # actually write
    docker compose start backend

Honours MUSIC_FEEDBACK_DIR, same as the rest of the feedback layer.
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.error
import urllib.request
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
from services.runtime_context import normalize_provenance  # noqa: E402

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


def _backend_port() -> int:
    """The port the API actually listens on.

    Hardcoding 8501 meant that moving the API to another port silently disabled
    the "is anything still writing?" guard - it would probe a dead port, see
    nothing, and happily migrate underneath a live backend. Read the same env var
    the server itself reads (api/server.py), with BACKEND_PORT as an alias.
    """
    for name in ("API_PORT", "BACKEND_PORT"):
        raw = str(os.getenv(name, "")).strip()
        if raw.isdigit():
            return int(raw)
    try:
        from config.settings import settings

        return int(getattr(settings, "api_port", 8501))
    except Exception:
        return 8501


def _backend_is_live() -> bool:
    """True if the API answers - i.e. something may still be appending events."""
    url = f"http://127.0.0.1:{_backend_port()}/health"
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            return 200 <= response.status < 500
    except (urllib.error.URLError, OSError, ValueError):
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="report counts without writing to SQLite")
    parser.add_argument("--allow-live-writes", action="store_true",
                        help="run even though the backend is up (it may append rows mid-run)")
    args = parser.parse_args()

    root = _feedback_dir()
    print(f"feedback dir: {root}")
    print(f"sqlite store: {fs.store_path()}")
    print(f"mode: {'DRY RUN - no writes' if args.dry_run else 'WRITE'}\n")

    if not args.dry_run and not args.allow_live_writes and _backend_is_live():
        print(f"REFUSING TO RUN: the backend is answering on :{_backend_port()}, so new feedback\n"
              "can land while this migrates. The completeness gate compares against a\n"
              "count taken at the start, so a concurrent write makes a correct store\n"
              "look broken - and a torn read look fine.\n"
              "  docker compose stop backend\n"
              "  <re-run this script>\n"
              "  docker compose start backend\n"
              "Override with --allow-live-writes only if you know nothing is writing.")
        return 1

    before = fs.counts()
    plan: list[tuple[str, str, list[dict], int]] = []
    unkeyed_report: list[str] = []
    for filename, table, fn_name in SOURCES:
        rows = load_jsonl(root / filename)
        id_keys = ID_KEYS[table]
        deduped = [
            normalize_provenance(row)
            for row in _dedupe_rows(rows, id_keys)
        ]
        # Rows with no id at all cannot be migrated safely: the store would mint a
        # synthetic uuid for each, so re-running would insert duplicates and the
        # count gate would fail AFTER the store was already polluted. Catch them
        # here, before a single write.
        unkeyed = [i for i, r in enumerate(rows, 1) if not effective_id(r, id_keys)]
        if unkeyed:
            preview = ", ".join(str(i) for i in unkeyed[:10])
            more = f" (+{len(unkeyed) - 10} more)" if len(unkeyed) > 10 else ""
            # "record #N" not "line N": load_jsonl skips blank lines, so the two
            # only coincide in a file with no blanks. Say what is actually true.
            unkeyed_report.append(
                f"  {filename}: {len(unkeyed)} row(s) with no {'/'.join(id_keys)}"
                f" - record #{preview}{more} (counting non-empty lines)")
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

    if unkeyed_report:
        print("\nABORTED BEFORE WRITING: some log rows carry no id.")
        for entry in unkeyed_report:
            print(entry)
        print("Nothing was written. Migrating these would mint a fresh uuid per row,\n"
              "so the store could never be reconciled with the log and a re-run would\n"
              "duplicate them. Give each row a stable id (or remove it, if it is a\n"
              "truncated tail line) and run again.")
        return 1

    if args.dry_run:
        print("\nLegacy rows without explicit runtime provenance remain quarantined "
              "(training_eligible=false).")
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
