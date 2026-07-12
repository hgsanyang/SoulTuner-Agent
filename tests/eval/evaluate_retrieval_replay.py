"""Replay frozen ToolPlans to isolate retrieval/ranking determinism."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
import time
from typing import Any

from tests.eval.runtime_fingerprint import (
    capture_runtime_fingerprint,
    fingerprint_changes,
)


def _song_signature(item: dict[str, Any]) -> tuple[str, str, str]:
    song = item.get("song") if isinstance(item.get("song"), dict) else item
    return (
        str(song.get("music_id") or song.get("id") or "").strip(),
        str(song.get("title") or "").strip(),
        str(song.get("artist") or "").strip(),
    )


def _frozen_cases(report: dict[str, Any], selected_ids: set[str]) -> list[dict[str, Any]]:
    cases = []
    for case in report.get("cases") or []:
        case_id = str(case.get("id") or "")
        if selected_ids and case_id not in selected_ids:
            continue
        complete_plan = (case.get("dialog_state") or {}).get("last_complete_plan") or {}
        retrieval_plan = complete_plan.get("retrieval_plan") or {}
        if not retrieval_plan:
            continue
        cases.append(
            {
                "id": case_id,
                "query": str(case.get("query") or ""),
                "intent_type": str(complete_plan.get("intent_type") or case.get("intent_type") or ""),
                "retrieval_plan": retrieval_plan,
            }
        )
    missing = selected_ids - {case["id"] for case in cases}
    if missing:
        raise ValueError(f"Cases missing frozen retrieval plans: {', '.join(sorted(missing))}")
    if not cases:
        raise ValueError("No replayable frozen plans found")
    return cases


async def run(report_path: Path, case_ids: list[str], repeats: int, limit: int) -> dict[str, Any]:
    from config.settings import settings
    from retrieval.hybrid_retrieval import MusicHybridRetrieval

    settings.eval_disable_side_effects = True
    settings.explanation_fast_mode = True
    report = json.loads(report_path.read_text(encoding="utf-8"))
    frozen = _frozen_cases(report, {case_id for case_id in case_ids if case_id})
    before = capture_runtime_fingerprint()
    runs: list[dict[str, list[tuple[str, str, str]]]] = []

    for _ in range(max(2, repeats)):
        retriever = MusicHybridRetrieval()
        current: dict[str, list[tuple[str, str, str]]] = {}
        for case in frozen:
            plan = dict(case["retrieval_plan"])
            plan["_intent_type"] = case["intent_type"]
            plan["_user_id"] = "__eval_readonly__"
            result = await retriever.retrieve(
                case["query"],
                limit=limit,
                precomputed_plan=plan,
            )
            current[case["id"]] = [_song_signature(item) for item in (result.data or [])]
        runs.append(current)

    after = capture_runtime_fingerprint()
    reference = runs[0]
    mismatches = {
        case["id"]: [run.get(case["id"], []) for run in runs]
        for case in frozen
        if any(run.get(case["id"], []) != reference.get(case["id"], []) for run in runs[1:])
    }
    state_changes = fingerprint_changes(before, after)
    return {
        "source_report": str(report_path.resolve()),
        "repeats": len(runs),
        "case_ids": [case["id"] for case in frozen],
        "deterministic": not mismatches,
        "mismatches": mismatches,
        "readonly": not state_changes,
        "state_changes": state_changes,
        "runs": runs,
    }


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Replay frozen retrieval plans and verify exact ranking")
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--limit", type=int, default=15)
    args = parser.parse_args()
    result = asyncio.run(run(args.report, args.case_id, args.repeats, args.limit))
    output = Path(__file__).parent / "results" / f"retrieval_replay_{time.strftime('%Y%m%d_%H%M%S')}.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "runs"}, ensure_ascii=False, indent=2))
    print(f"report={output.resolve()}")
    if not result["deterministic"] or not result["readonly"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

