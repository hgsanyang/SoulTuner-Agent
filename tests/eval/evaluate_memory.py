"""Deterministic memory-behavior evaluation.

This ruler isolates the memory update semantics from end-to-end retrieval.  It
does not call LLMs, Neo4j, GraphZep, or Mem0.  Use it before changing
MemoryGateway adapters or feedback-to-preference rules.

Run:
    python -m tests.eval.evaluate_memory
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any

from services.memory_gateway import derive_preferences_from_slate_feedback


@dataclass(frozen=True)
class MemoryCase:
    case_id: str
    rating: str
    reasons: list[str]
    note: str
    must_contain: dict[str, list[str]]
    must_not_contain: dict[str, list[str]]


CASES = [
    MemoryCase(
        case_id="avoid_noisy_music",
        rating="too_noisy",
        reasons=["太吵了"],
        note="少一点 EDM 和土嗨",
        must_contain={
            "avoid_genres": ["EDM"],
            "avoid_moods": ["Energetic", "Aggressive"],
        },
        must_not_contain={},
    ),
    MemoryCase(
        case_id="avoid_over_sad_music",
        rating="too_sad",
        reasons=["太丧了"],
        note="想更温暖一点",
        must_contain={
            "avoid_moods": ["Sad", "Melancholy"],
            "add_moods": ["Warm"],
        },
        must_not_contain={},
    ),
    MemoryCase(
        case_id="ask_for_niche_discovery",
        rating="more_niche",
        reasons=["想更小众"],
        note="不要总是旧歌单",
        must_contain={"activity_contexts": ["longtail", "less_familiar"]},
        must_not_contain={"avoid_moods": ["Energetic"]},
    ),
    MemoryCase(
        case_id="closer_to_seed_is_directional",
        rating="closer_to_seed",
        reasons=["更贴近刚才那首"],
        note="",
        must_contain={"activity_contexts": ["closer_to_seed_song"]},
        must_not_contain={"avoid_genres": ["EDM"]},
    ),
    MemoryCase(
        case_id="ask_for_more_energy",
        rating="too_quiet",
        reasons=["太安静了"],
        note="下次更有劲一点",
        must_contain={
            "add_moods": ["Energetic", "Upbeat"],
            "add_scenarios": ["Driving"],
        },
        must_not_contain={"avoid_moods": ["Energetic"]},
    ),
    MemoryCase(
        case_id="avoid_over_familiar_songs",
        rating="too_familiar",
        reasons=["太像我旧歌单"],
        note="想发现更多没听过的",
        must_contain={"activity_contexts": ["less_familiar", "avoid_overexposed"]},
        must_not_contain={"avoid_genres": ["EDM"]},
    ),
]


def _as_set(values: Any) -> set[str]:
    return {str(v).casefold() for v in (values or [])}


def evaluate_case(case: MemoryCase) -> dict[str, Any]:
    update = derive_preferences_from_slate_feedback(
        rating=case.rating,
        reasons=case.reasons,
        note=case.note,
    )
    failures: list[str] = []
    for field, expected_values in case.must_contain.items():
        actual = _as_set(update.get(field))
        for expected in expected_values:
            if expected.casefold() not in actual:
                failures.append(f"missing {field}:{expected}")
    for field, forbidden_values in case.must_not_contain.items():
        actual = _as_set(update.get(field))
        for forbidden in forbidden_values:
            if forbidden.casefold() in actual:
                failures.append(f"unexpected {field}:{forbidden}")
    return {
        "case_id": case.case_id,
        "passed": not failures,
        "failures": failures,
        "update": update,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args()

    rows = [evaluate_case(case) for case in CASES]
    passed = sum(1 for row in rows if row["passed"])
    report = {
        "total": len(rows),
        "passed": passed,
        "pass_rate": round(passed / len(rows), 4),
        "cases": rows,
    }
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"Memory eval: {passed}/{len(rows)} passed ({report['pass_rate']:.2%})")
        for row in rows:
            status = "PASS" if row["passed"] else "FAIL"
            print(f"- {status} {row['case_id']}")
            for failure in row["failures"]:
                print(f"  - {failure}")
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
