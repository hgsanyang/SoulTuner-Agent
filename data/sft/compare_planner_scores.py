"""Compare two planner runs without silently promoting the larger model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


CORE_METRICS = (
    "schema_valid",
    "compilable",
    "request_kind_acc",
    "lane_f1",
    "clarification_precision",
    "clarification_recall",
    "hyde_present_when_dense",
)


def compare(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    tolerance: float = 0.03,
) -> dict[str, Any]:
    findings = []
    deltas = {}
    for metric in CORE_METRICS:
        before = baseline.get(metric)
        after = candidate.get(metric)
        if before is None or after is None:
            findings.append(f"missing metric: {metric}")
            continue
        delta = float(after) - float(before)
        deltas[metric] = round(delta, 4)
        if delta < -tolerance:
            findings.append(f"{metric} regressed by {delta:.4f} (> {tolerance:.4f})")

    if float(candidate.get("schema_valid") or 0.0) < 1.0:
        findings.append("candidate schema_valid must be 1.0")
    if int(candidate.get("lane_authority_violations") or 0) != 0:
        findings.append("candidate lane_authority_violations must be 0")
    if not bool((candidate.get("coverage") or {}).get("complete")):
        findings.append("candidate prediction coverage is incomplete")

    category_deltas = {}
    baseline_categories = baseline.get("by_request_kind") or {}
    candidate_categories = candidate.get("by_request_kind") or {}
    for kind, before_values in sorted(baseline_categories.items()):
        after_values = candidate_categories.get(kind)
        if not after_values:
            findings.append(f"candidate is missing request category: {kind}")
            continue
        category_deltas[kind] = {}
        for metric in ("schema_valid", "request_kind_acc", "lane_f1"):
            delta = float(after_values.get(metric, 0.0)) - float(before_values.get(metric, 0.0))
            category_deltas[kind][metric] = round(delta, 4)
            if delta < -tolerance:
                findings.append(f"{kind}.{metric} regressed by {delta:.4f}")

    return {
        "passed": not findings,
        "tolerance": tolerance,
        "metric_deltas": deltas,
        "category_deltas": category_deltas,
        "findings": findings,
        "note": "Quality-only gate. Runtime p50 must be measured separately before deployment.",
    }


def compare_split_gap(
    regression: dict[str, Any],
    sealed: dict[str, Any],
    *,
    tolerance: float = 0.08,
) -> dict[str, Any]:
    """Detect collapse on unseen entities without treating split mix as identical."""
    findings: list[str] = []
    deltas: dict[str, float] = {}
    for metric in CORE_METRICS:
        familiar = regression.get(metric)
        unseen = sealed.get(metric)
        if familiar is None or unseen is None:
            findings.append(f"missing split metric: {metric}")
            continue
        delta = float(unseen) - float(familiar)
        deltas[metric] = round(delta, 4)
        if delta < -tolerance:
            findings.append(f"sealed {metric} trails regression by {-delta:.4f}")
    return {
        "passed": not findings,
        "tolerance": tolerance,
        "sealed_minus_regression": deltas,
        "findings": findings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--candidate-regression", type=Path)
    parser.add_argument("--split-gap-tolerance", type=float, default=0.08)
    parser.add_argument("--tolerance", type=float, default=0.03)
    parser.add_argument("--json", dest="json_out", type=Path)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    report = compare(baseline, candidate, tolerance=args.tolerance)
    if args.candidate_regression:
        regression = json.loads(args.candidate_regression.read_text(encoding="utf-8"))
        split_gap = compare_split_gap(
            regression, candidate, tolerance=args.split_gap_tolerance
        )
        report["sealed_vs_regression"] = split_gap
        if not split_gap["passed"]:
            report["passed"] = False
            report["findings"].extend(
                f"sealed_vs_regression: {finding}" for finding in split_gap["findings"]
            )
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 12


if __name__ == "__main__":
    raise SystemExit(main())
