"""Reproducible V2 -> V3 dataset migration with audit (Phase B, GPT 4th review).

Structural migration is a pure function, but structure is not meaning: the legacy
``web_search`` intent covers BOTH "Billboard 本周冠军是谁" (information) and "推荐
几首本周新歌" (recommendation + web). Blanket-mapping would mint wrong labels, so
those samples are quarantined into ``ambiguous_samples.jsonl`` and excluded from
the migrated training set until they are re-judged from the original query.

Also verifies, per record:
  - V3 schema validity
  - V2 -> V3 -> V2 round-trip identity   (only meaningful where V2 was self-consistent)
  - compile equivalence (V2 compiler vs V3 compiler)

Usage:
    python -m data.sft.migrate_v2_to_v3_dataset \
        --in data/teacher/private/pilot_full_clean.jsonl \
        --out data/teacher/private/pilot_full_v3.jsonl \
        --ambiguous data/teacher/private/ambiguous_samples.jsonl \
        --audit data/teacher/private/v3_migration_audit.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# lanes a legacy intent implies, used only to tell "V2 contradicted itself" from
# "V2 legitimately named several tools" (graph primary + web supplement).
_IMPLIED_LANES = {
    "graph_search": {"graph"},
    "vector_search": {"dense"},
    "hybrid_search": {"graph", "dense"},
    "web_search": {"web"},
}


def migrate(in_path: Path, out_path: Path, ambiguous_path: Path, audit_path: Path) -> dict:
    from schemas.planner_decision import PlannerDecisionV2, compile_to_query_plan
    from schemas.planner_decision_v3 import (
        compile_v3_to_query_plan,
        migrate_v2_to_v3,
        migrate_v3_to_v2,
        migration_note,
    )

    rows = [json.loads(line) for line in in_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    kinds: Counter = Counter()
    modes: Counter = Counter()
    ambiguous_reasons: Counter = Counter()
    invalid = rt_fail = compile_fail = 0
    self_conflict = composite_tools = 0
    migrated: list[dict] = []
    ambiguous: list[dict] = []
    failures: list[str] = []

    for row in rows:
        v2 = PlannerDecisionV2.model_validate(row["teacher_decision"])
        tag = f"{row.get('episode_id')}#{row.get('turn_id')}"
        try:
            v3 = migrate_v2_to_v3(v2)
        except Exception as exc:
            invalid += 1
            failures.append(f"{tag}: invalid V3: {exc}")
            continue

        named = set(v2.tool_names)
        implied = _IMPLIED_LANES.get(v2.intent)
        self_consistent = implied is None or not named or named == implied
        if not self_consistent:
            # graph+web with external knowledge required is a legitimate composite
            # call, not a contradiction; only a MISSING required lane is a conflict.
            if implied and implied <= named:
                composite_tools += 1
                self_consistent = True
            else:
                self_conflict += 1

        if self_consistent:
            back = migrate_v3_to_v2(v3)
            if back.intent != v2.intent or sorted(back.tool_names) != sorted(v2.tool_names):
                rt_fail += 1
                failures.append(f"{tag}: round-trip {v2.intent}{v2.tool_names} -> {back.intent}{back.tool_names}")
            p2, p3 = compile_to_query_plan(v2), compile_v3_to_query_plan(v3)
            if (p2.intent_type != p3.intent_type
                    or p2.retrieval_plan.use_graph != p3.retrieval_plan.use_graph
                    or p2.retrieval_plan.use_vector != p3.retrieval_plan.use_vector
                    or p2.retrieval_plan.vector_acoustic_queries != p3.retrieval_plan.vector_acoustic_queries):
                compile_fail += 1
                failures.append(f"{tag}: compile {p2.intent_type} vs {p3.intent_type}")

        note = migration_note(v2)
        record = {**{k: v for k, v in row.items() if k != "teacher_decision"},
                  "teacher_decision_v3": v3.model_dump(mode="json", exclude_none=True),
                  "migration": {"ambiguous": note.ambiguous, "reason": note.reason,
                                "legacy_intent": v2.intent}}
        if note.ambiguous:
            ambiguous_reasons[note.reason] += 1
            ambiguous.append(record)
        else:
            kinds[v3.request_kind] += 1
            modes[v3.response_mode] += 1
            migrated.append(record)

    for path, items in ((out_path, migrated), (ambiguous_path, ambiguous)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as sink:
            for item in items:
                sink.write(json.dumps(item, ensure_ascii=False) + "\n")

    audit = {
        "input": str(in_path),
        "total": len(rows),
        "migrated_trainable": len(migrated),
        "quarantined_ambiguous": len(ambiguous),
        "ambiguous_reasons": dict(ambiguous_reasons),
        "v3_invalid": invalid,
        "roundtrip_failures": rt_fail,
        "compile_mismatches": compile_fail,
        "v2_self_conflict": self_conflict,
        "v2_composite_tools_legit": composite_tools,
        "request_kind": dict(kinds),
        "response_mode": dict(modes),
        "failures": failures[:20],
        "green": invalid == 0 and rt_fail == 0 and compile_fail == 0,
        "out": str(out_path),
        "ambiguous_out": str(ambiguous_path),
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="inp", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ambiguous", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()

    audit = migrate(args.inp, args.out, args.ambiguous, args.audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0 if audit["green"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
