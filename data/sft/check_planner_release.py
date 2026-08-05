"""Apply the frozen V4 model-quality gates to one trained planner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


OVERALL_METRICS = ("request_kind_acc", "lane_f1", "hyde_present_when_dense")
CATEGORY_METRICS = ("request_kind_acc", "lane_f1")


def _minimum(value_pp: float) -> float:
    return 1.0 + value_pp / 100.0


def check_release(
    manifest: dict[str, Any],
    regression: dict[str, Any],
    sealed: dict[str, Any],
) -> dict[str, Any]:
    gates = manifest["sealed_policy"]["release_gates"]
    overall_minimum = _minimum(float(gates["overall_vs_teacher_pp"]))
    category_minimum = 1.0 - float(gates["per_kind_max_regression_pp"]) / 100.0
    max_split_gap = float(gates["sealed_vs_regression_max_gap_pp"]) / 100.0
    findings: list[str] = []

    for split_name, score in (("regression", regression), ("sealed", sealed)):
        if not bool((score.get("coverage") or {}).get("complete")):
            findings.append(f"{split_name}: prediction coverage is incomplete")
        if float(score.get("schema_valid") or 0.0) != float(gates["schema_validity"]):
            findings.append(f"{split_name}: schema validity is not 1.0")
        if float(score.get("compilable") or 0.0) != 1.0:
            findings.append(f"{split_name}: one or more predictions do not compile")
        if int(score.get("lane_authority_violations") or 0) != int(
            gates["lane_authority_violations"]
        ):
            findings.append(f"{split_name}: lane authority violations are non-zero")
        for metric in OVERALL_METRICS:
            value = score.get(metric)
            if value is None or float(value) < overall_minimum:
                findings.append(
                    f"{split_name}.{metric}={value!r} is below {overall_minimum:.4f}"
                )
        if int(score.get("clarification_gold_cases") or 0) > 0:
            for metric in ("clarification_precision", "clarification_recall"):
                value = score.get(metric)
                if value is None or float(value) < overall_minimum:
                    findings.append(
                        f"{split_name}.{metric}={value!r} is below "
                        f"{overall_minimum:.4f}"
                    )
        for kind, values in sorted((score.get("by_request_kind") or {}).items()):
            for metric in CATEGORY_METRICS:
                value = values.get(metric)
                if value is None or float(value) < category_minimum:
                    findings.append(
                        f"{split_name}.{kind}.{metric}={value!r} is below "
                        f"{category_minimum:.4f}"
                    )

    split_deltas: dict[str, float] = {}
    for metric in OVERALL_METRICS:
        delta = float(sealed.get(metric) or 0.0) - float(regression.get(metric) or 0.0)
        split_deltas[metric] = round(delta, 4)
        if delta < -max_split_gap:
            findings.append(
                f"sealed.{metric} trails regression by {-delta:.4f} (> {max_split_gap:.4f})"
            )

    return {
        "passed": not findings,
        "dataset_version": manifest.get("dataset_version"),
        "thresholds": {
            "overall_minimum": overall_minimum,
            "category_minimum": category_minimum,
            "max_sealed_vs_regression_gap": max_split_gap,
        },
        "sealed_minus_regression": split_deltas,
        "findings": findings,
        "note": (
            "Planner contract gate only. Outcome holdout, MuQ attribute P@10 and "
            "served latency remain separate deployment gates."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--regression", type=Path, required=True)
    parser.add_argument("--sealed", type=Path, required=True)
    parser.add_argument("--json", dest="json_out", type=Path, required=True)
    args = parser.parse_args()
    report = check_release(
        json.loads(args.manifest.read_text(encoding="utf-8")),
        json.loads(args.regression.read_text(encoding="utf-8")),
        json.loads(args.sealed.read_text(encoding="utf-8")),
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 14


if __name__ == "__main__":
    raise SystemExit(main())
