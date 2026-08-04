#!/usr/bin/env python
"""Validate and optionally apply JSONL returned by an external web LLM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from services.offline_tag_review import apply_validated_tags, read_jsonl, validate_result


def run_import(
    task_manifest: str | Path,
    results_path: str | Path,
    *,
    apply: bool = False,
    replace_existing: bool = False,
) -> dict:
    task_rows = read_jsonl(task_manifest)
    tasks = {str(row.get("task_id") or ""): row for row in task_rows}
    results = read_jsonl(results_path)
    validated = [validate_result(row, tasks) for row in results]
    if len({row["task_id"] for row in validated}) != len(validated):
        raise ValueError("duplicate task_id in results")
    applied = 0
    skipped = 0
    if apply:
        from retrieval.neo4j_client import get_neo4j_client

        client = get_neo4j_client()
        for row in validated:
            if apply_validated_tags(client, row, replace_existing=replace_existing):
                applied += 1
            else:
                skipped += 1
    taxonomy_feedback = [
        {"task_id": row["task_id"], **row["taxonomy_feedback"]}
        for row in validated
        if any(row["taxonomy_feedback"].values())
    ]
    return {
        "mode": "apply" if apply else "dry_run",
        "task_count": len(tasks),
        "result_count": len(results),
        "validated": len(validated),
        "applied": applied,
        "skipped": skipped,
        "taxonomy_feedback": taxonomy_feedback,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-manifest", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--replace-existing", action="store_true")
    parser.add_argument("--report")
    args = parser.parse_args()
    result = run_import(
        args.task_manifest,
        args.results,
        apply=args.apply,
        replace_existing=args.replace_existing,
    )
    if args.report:
        Path(args.report).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
