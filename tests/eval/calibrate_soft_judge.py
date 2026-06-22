"""Calibrate objective soft-intent judge against a human gold set.

This script is pure logic: it does not start the Agent, does not call an LLM,
and does not read system-generated explanations.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from tests.eval.soft_judge import judge_objective_soft_intent


DEFAULT_GOLD = Path(__file__).parent / "judge_gold" / "objective_soft_judge_gold.json"


def _unwrap_song(item: dict[str, Any]) -> dict[str, Any]:
    song = item.get("song", item)
    return song if isinstance(song, dict) else {}


def load_gold(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Gold file must be a list: {path}")
    return data


def calibrate(gold_cases: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    confusion: Counter[tuple[str, str]] = Counter()
    by_category = defaultdict(lambda: {"total": 0, "correct": 0, "abstain": 0})

    for case in gold_cases:
        songs = [_unwrap_song(song) for song in case.get("songs", [])]
        expected = case.get("expected_status")
        decision = judge_objective_soft_intent(songs, case.get("judge", {}))
        actual = decision.status
        correct = actual == expected
        category = case.get("category", "uncategorized")

        confusion[(expected, actual)] += 1
        by_category[category]["total"] += 1
        if correct:
            by_category[category]["correct"] += 1
        if actual == "skip":
            by_category[category]["abstain"] += 1

        rows.append({
            "id": case.get("id"),
            "category": category,
            "expected": expected,
            "actual": actual,
            "correct": correct,
            "confidence": decision.confidence,
            "detail": decision.detail,
        })

    total = len(rows)
    correct = sum(1 for row in rows if row["correct"])
    abstain = sum(1 for row in rows if row["actual"] == "skip")
    decided = total - abstain
    decided_correct = sum(1 for row in rows if row["correct"] and row["actual"] != "skip")

    return {
        "total": total,
        "correct": correct,
        "exact_accuracy": correct / total if total else None,
        "abstain": abstain,
        "coverage": decided / total if total else None,
        "decided_accuracy": decided_correct / decided if decided else None,
        "confusion": {f"{k[0]}->{k[1]}": v for k, v in sorted(confusion.items())},
        "by_category": dict(by_category),
        "rows": rows,
    }


def print_report(report: dict[str, Any]) -> None:
    print("=" * 64)
    print("Objective Soft Judge Calibration")
    print("=" * 64)
    print(
        f"total={report['total']} correct={report['correct']} "
        f"exact_accuracy={report['exact_accuracy']:.1%} "
        f"coverage={report['coverage']:.1%} "
        f"decided_accuracy={report['decided_accuracy']:.1%}"
    )

    print("\nConfusion:")
    for key, value in report["confusion"].items():
        print(f"  {key}: {value}")

    print("\nBy category:")
    for category, data in sorted(report["by_category"].items()):
        accuracy = data["correct"] / data["total"] if data["total"] else 0.0
        print(
            f"  {category:24s} total={data['total']:2d} "
            f"correct={data['correct']:2d} abstain={data['abstain']:2d} acc={accuracy:.1%}"
        )

    failures = [row for row in report["rows"] if not row["correct"]]
    if failures:
        print("\nMismatches:")
        for row in failures:
            print(
                f"  [{row['id']}] expected={row['expected']} actual={row['actual']} "
                f"confidence={row['confidence']:.2f} :: {row['detail']}"
            )


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Calibrate objective soft-intent judge")
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--min-accuracy", type=float, default=0.85)
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    args = parser.parse_args()

    report = calibrate(load_gold(args.gold))
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_report(report)

    accuracy = report["exact_accuracy"] or 0.0
    if accuracy < args.min_accuracy:
        print(
            f"\nCalibration below threshold: {accuracy:.1%} < {args.min_accuracy:.1%}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
