#!/usr/bin/env python
"""Decode selected local cache entries and feed them into the catalog flywheel.

The default mode is a real dry-run: it decodes a bounded sample into a temporary
staging directory, verifies each audio stream, reads embedded metadata, and
reports duplicate decisions. ``--apply`` publishes only rows classified as
ready, writes a reversible run manifest, creates the graph shell, and queues the
existing tag/vector enrichment worker.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.import_netease_cache_manifest import (  # noqa: E402
    DUPLICATE_EXACT,
    DUPLICATE_SUSPECTED,
    METADATA_READY,
    classify,
    fetch_lyrics,
    fetch_metadata,
    load_library_index,
    scan_cache,
)
from services.cache_audio_import import (  # noqa: E402
    build_existing_digest_index,
    choose_preferred_cache_entries,
    plan_as_dict,
    plan_cache_audio,
    publish_cache_audio,
    remove_published_files,
    sha256_file,
)


def _default_data_root() -> Path:
    configured = str(os.getenv("MUSIC_DATA_PATH") or "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return (PROJECT_ROOT.parent / "data").resolve()


async def run_import(args: argparse.Namespace) -> dict[str, Any]:
    run_id = args.run_id or f"cache-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    data_root = Path(args.data_root).expanduser().resolve()
    processed_root = Path(args.processed_root).expanduser().resolve()
    run_root = Path(args.run_root).expanduser().resolve()
    run_root.mkdir(parents=True, exist_ok=True)

    entries, selection_skips = choose_preferred_cache_entries(scan_cache(Path(args.cache_dir)))
    if args.scan_limit:
        entries = entries[: args.scan_limit]
    ids = sorted({entry.song_id for entry in entries})
    metadata = {} if args.no_metadata else await fetch_metadata(ids)
    lyrics = {} if args.no_lyrics or args.no_metadata else await fetch_lyrics(sorted(metadata))
    by_id, by_key = load_library_index()
    classified = classify(entries, metadata, by_id, by_key, lyrics)

    candidates = [entry for entry in classified if entry.state == METADATA_READY]
    if args.limit:
        candidates = candidates[: args.limit]
    source_before = _source_fingerprint(candidates)
    skipped_classification = [
        {
            "path": entry.path,
            "song_id": entry.song_id,
            "state": entry.state,
            "reason": "; ".join(entry.reasons),
        }
        for entry in classified
        if entry.state in {DUPLICATE_EXACT, DUPLICATE_SUSPECTED} or entry.state != METADATA_READY
    ]
    candidate_sizes = {int(entry.bytes) for entry in candidates}
    existing_digests = build_existing_digest_index(
        [processed_root / "audio", data_root / "online_acquired" / "audio"],
        candidate_sizes=candidate_sizes,
    )

    staging_parent = processed_root / ".cache_import_staging"
    staging_parent.mkdir(parents=True, exist_ok=True)
    plans = []
    published = []
    seen_digests: set[str] = set()
    job_id: str | None = None

    with tempfile.TemporaryDirectory(prefix=f"{run_id}-", dir=staging_parent) as temporary:
        for entry in candidates:
            try:
                plan = plan_cache_audio(
                    entry,
                    temporary,
                    existing_digests=existing_digests,
                    seen_digests=seen_digests,
                )
            except Exception as exc:
                plans.append(
                    {
                        "source_path": entry.path,
                        "song_id": entry.song_id,
                        "state": "decode_failed",
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            plans.append(plan_as_dict(plan))
            if not args.apply or plan.state != "ready":
                continue
            result = publish_cache_audio(
                plan,
                processed_root=processed_root,
                lyrics=entry.lyrics,
                run_id=run_id,
            )
            published.append(result)

        if args.apply and published:
            records = [item.record for item in published]
            for record in records:
                record["tagging_mode"] = args.tagging_mode
            from tools.acquire_music import _quick_ingest_to_neo4j
            from services.ingest_queue import enqueue_songs

            await _quick_ingest_to_neo4j(records)
            job_id = enqueue_songs(records)

    state_counts = Counter(str(row.get("state") or "unknown") for row in plans)
    source_after = _source_fingerprint(candidates)
    source_unchanged = source_before == source_after
    if not source_unchanged:
        raise RuntimeError("cache source fingerprint changed during import")
    report = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(),
        "mode": "apply" if args.apply else "dry_run",
        "cache_dir": str(Path(args.cache_dir).resolve()),
        "processed_root": str(processed_root),
        "inventory_rows": len(entries),
        "selected": len(candidates),
        "selection_skips": selection_skips,
        "classification_skips": skipped_classification,
        "plans": plans,
        "state_counts": dict(state_counts),
        "published": [
            {"record": item.record, "created_files": item.created_files, "plan": asdict(item.plan)}
            for item in published
        ],
        "job_id": job_id,
        "source_files_modified": not source_unchanged,
        "source_fingerprint": source_after,
    }
    report_path = run_root / f"{run_id}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report


def _source_fingerprint(entries: list[Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for entry in entries:
        path = Path(str(entry.path))
        stat = path.stat()
        result[str(path)] = {
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": sha256_file(path),
        }
    return result


def rollback_run(report_path: str | Path, *, dry_run: bool = False) -> dict[str, Any]:
    """Plan or execute removal of assets stamped by one cache import run."""
    path = Path(report_path)
    report = json.loads(path.read_text(encoding="utf-8"))
    run_id = str(report.get("run_id") or "").strip()
    if not run_id:
        raise ValueError("run report has no run_id")

    job_id = str(report.get("job_id") or "").strip()
    if job_id:
        from services import ingest_queue

        processing = ingest_queue.PROCESSING_DIR / f"{job_id}.json"
        if processing.exists():
            raise RuntimeError(f"ingest job {job_id} is processing; rollback refused")

    graph_deleted = 0
    graph_rows: list[dict[str, Any]] = []
    try:
        from retrieval.neo4j_client import get_neo4j_client

        client = get_neo4j_client()
        graph_rows = client.execute_query(
            """
            MATCH (s:Song {cache_import_run_id: $run_id})
            OPTIONAL MATCH (:User)-[interaction]->(s)
            WITH s, count(interaction) AS interactions
            RETURN count(s) AS nodes, sum(interactions) AS interactions
            """,
            {"run_id": run_id},
        )
        interactions = int((graph_rows[0] if graph_rows else {}).get("interactions") or 0)
        if interactions:
            raise RuntimeError(
                f"run {run_id} has {interactions} user interactions; rollback refused"
            )
        if not dry_run:
            deleted = client.execute_query(
                """
                MATCH (s:Song {cache_import_run_id: $run_id})
                WITH collect(s) AS songs
                WITH songs, size(songs) AS count
                UNWIND songs AS song
                DETACH DELETE song
                RETURN count
                """,
                {"run_id": run_id},
            )
            graph_deleted = int((deleted[0] if deleted else {}).get("count") or 0)
    except ImportError:
        graph_rows = []

    if job_id and not dry_run:
        from services import ingest_queue

        for directory in (
            ingest_queue.PENDING_DIR,
            ingest_queue.FAILED_DIR,
            ingest_queue.DONE_DIR,
        ):
            (directory / f"{job_id}.json").unlink(missing_ok=True)

    files = [
        value
        for item in report.get("published") or []
        for value in item.get("created_files") or []
    ]
    existing_files = sum(1 for value in files if Path(value).exists())
    removed = 0 if dry_run else remove_published_files(files)
    graph_nodes = int((graph_rows[0] if graph_rows else {}).get("nodes") or 0)
    result = {
        "run_id": run_id,
        "dry_run": dry_run,
        "would_remove_files": existing_files,
        "graph_would_delete": graph_nodes,
        "removed_files": removed,
        "graph_deleted": graph_deleted,
        "graph_checked": bool(graph_rows),
    }
    suffix = ".rollback-plan.json" if dry_run else ".rollback.json"
    rollback_path = path.with_name(path.stem + suffix)
    rollback_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def build_parser() -> argparse.ArgumentParser:
    data_root = _default_data_root()
    from scripts.import_netease_cache_manifest import DEFAULT_CACHE_DIR

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--data-root", default=str(data_root))
    parser.add_argument("--processed-root", default=str(data_root / "processed_audio"))
    parser.add_argument("--run-root", default=str(PROJECT_ROOT / "data" / "cache_import_runs"))
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument(
        "--scan-limit", type=int, default=100,
        help="maximum inventory rows to resolve before selecting --limit import candidates",
    )
    parser.add_argument("--run-id", default="")
    parser.add_argument("--no-metadata", action="store_true")
    parser.add_argument("--no-lyrics", action="store_true")
    parser.add_argument(
        "--tagging-mode",
        choices=("api", "deferred"),
        default="api",
        help=(
            "api calls the configured LLM for lyric tags during enrichment; "
            "deferred extracts vectors now and leaves tags for an offline review bundle"
        ),
    )
    parser.add_argument("--apply", action="store_true", help="publish ready rows and enqueue enrichment")
    rollback = parser.add_mutually_exclusive_group()
    rollback.add_argument("--rollback", metavar="RUN_REPORT", help="rollback one applied run")
    rollback.add_argument(
        "--rollback-dry-run",
        metavar="RUN_REPORT",
        help="verify and report rollback scope without changing graph, queue, or files",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.rollback:
        print(json.dumps(rollback_run(args.rollback), ensure_ascii=False, indent=2))
        return 0
    if args.rollback_dry_run:
        print(json.dumps(
            rollback_run(args.rollback_dry_run, dry_run=True),
            ensure_ascii=False,
            indent=2,
        ))
        return 0
    report = asyncio.run(run_import(args))
    print(json.dumps({
        "run_id": report["run_id"],
        "mode": report["mode"],
        "selected": report["selected"],
        "state_counts": report["state_counts"],
        "published": len(report["published"]),
        "job_id": report["job_id"],
        "report_path": report["report_path"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
