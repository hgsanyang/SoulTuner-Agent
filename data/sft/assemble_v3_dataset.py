"""Assemble the frozen Planner V3 corpus from audited, disjoint sources.

The command fails closed when a source is missing, a sample key is duplicated,
or a decision does not validate against PlannerDecisionV3. Generated corpora
live under ``data/teacher/private`` and remain gitignored.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

from schemas.planner_decision_v3 import PlannerDecisionV3


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sample_key(row: dict[str, Any]) -> str:
    episode_id = str(row.get("episode_id") or "").strip()
    turn_id = row.get("turn_id")
    if not episode_id or turn_id is None:
        raise ValueError("every V3 sample requires episode_id and turn_id")
    return f"{episode_id}#{turn_id}"


def _normalized_row(
    row: dict[str, Any],
    *,
    source_name: str,
) -> dict[str, Any]:
    decision = PlannerDecisionV3.model_validate(row.get("teacher_decision_v3"))
    governance = dict(row.get("training_governance") or {})
    if governance.get("training_eligible") is False:
        raise ValueError(f"ineligible row entered V3 assembly: {_sample_key(row)}")
    governance.update(
        {
            "training_eligible": True,
            "data_purpose": "planner_v3_sft",
            "assembly_source": source_name,
        }
    )
    return {
        **row,
        "teacher_decision_v3": decision.model_dump(mode="json", exclude_none=True),
        "training_governance": governance,
    }


def assemble(
    sources: list[tuple[str, Path]],
    output: Path,
    manifest_output: Path,
    *,
    expected_recollected: int = 7,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    source_counts: dict[str, int] = {}

    for source_name, path in sources:
        rows = _read_jsonl(path)
        if source_name == "strong_teacher_recollection" and len(rows) != expected_recollected:
            raise ValueError(
                f"expected {expected_recollected} strong-teacher recollections, got {len(rows)}"
            )
        source_counts[source_name] = len(rows)
        for raw in rows:
            row = _normalized_row(raw, source_name=source_name)
            key = _sample_key(row)
            if key in seen:
                raise ValueError(f"duplicate V3 sample key: {key}")
            seen.add(key)
            records.append(row)

    records.sort(key=lambda row: (_sample_key(row), str(row.get("current_query") or "")))
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(row, ensure_ascii=False) + "\n"
        for row in records
    )
    output.write_text(payload, encoding="utf-8")

    kind_counts = Counter(
        row["teacher_decision_v3"]["request_kind"] for row in records
    )
    mode_counts = Counter(
        row["teacher_decision_v3"]["response_mode"] for row in records
    )
    manifest = {
        "schema": "planner_decision_v3",
        "records": len(records),
        "sources": source_counts,
        "request_kind": dict(sorted(kind_counts.items())),
        "response_mode": dict(sorted(mode_counts.items())),
        "sample_keys_unique": len(seen) == len(records),
        "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "output": str(output),
    }
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        type=Path,
        default=Path("data/teacher/private/pilot_full_v3.jsonl"),
    )
    parser.add_argument(
        "--reviewed",
        type=Path,
        default=Path("data/sft/reviews/v3_resolved_trainable.jsonl"),
    )
    parser.add_argument(
        "--recollected",
        type=Path,
        default=Path("data/teacher/private/v3_false_clarification_recollected.jsonl"),
    )
    parser.add_argument(
        "--gap-seeds",
        type=Path,
        default=Path("data/sft/curated_v3_gap_seeds.jsonl"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/teacher/private/planner_v3_frozen.jsonl"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/teacher/private/planner_v3_frozen_manifest.json"),
    )
    parser.add_argument("--expected-recollected", type=int, default=7)
    args = parser.parse_args()

    manifest = assemble(
        [
            ("safe_v2_migration", args.base),
            ("manual_ambiguity_review", args.reviewed),
            ("strong_teacher_recollection", args.recollected),
            ("curated_v3_gap_seed", args.gap_seeds),
        ],
        args.out,
        args.manifest,
        expected_recollected=args.expected_recollected,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
