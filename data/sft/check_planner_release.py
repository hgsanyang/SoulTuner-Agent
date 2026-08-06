"""Apply the frozen V4 model-quality gates to one trained planner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


OVERALL_METRICS = ("request_kind_acc", "lane_f1", "hyde_present_when_dense")
CATEGORY_METRICS = ("request_kind_acc", "lane_f1")

#: Below this many supporting rows a metric is reported, not enforced.
#:
#: This is a **minimum operational support**, not a statistical one. It is the
#: point below which a percentage stops being worth failing a release over —
#: regression holds one row each for acquisition, library and conversation, so
#: those metrics can only be 0.0 or 1.0 and a single miss would block a release
#: while a pass would mean nothing. It does **not** license any claim that 20
#: rows resolve a 3pp difference with confidence; the 3pp thresholds come from
#: the frozen manifest and are a policy choice, not a measurement guarantee.
#:
#: sealed is unaffected: five kinds, ~100 rows each, always enforced.
MIN_OPERATIONAL_SUPPORT = 20

#: A per-kind metric may be genuinely undefined for the rows it was asked about.
#: `lane_f1` is undefined where no gold row asks for any lane at all: every
#: correct row then contributes nothing to tp/fp/fn, and the category's score is
#: decided entirely by whichever row is wrong. score_student reports such a
#: metric as null with this status, and a null carrying it must never be read as
#: a zero — that is the difference between "does not apply" and "scored badly".
NOT_APPLICABLE = "not_applicable"

#: Metrics whose support is counted by their own denominator rather than by the
#: split's gold count. Precision rests on how many cases were predicted; recall
#: rests on how many were in the gold. Judging both by the gold count enforced a
#: precision built on 5 predictions as though it rested on 3 gold rows.
#: (metric, support key, fallback). The fallback keeps score reports written
#: before the supports existed enforceable: they already carried the same two
#: counts under different names, and reading a missing key as support=0 would
#: quietly stop gating clarification on every historical report.
CLARIFICATION_SUPPORTS = (
    ("clarification_precision", "clarification_precision_support",
     "clarification_predicted_cases"),
    ("clarification_recall", "clarification_recall_support",
     "clarification_gold_cases"),
)


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
    # 支持数低于操作下限就只报告不判死：n=1 的指标只能取 0.0 或 1.0，
    # 拿它当硬门既会误杀也证明不了什么。这是操作下限，不声称统计置信度。
    canaries: list[str] = []
    not_applicable: list[str] = []
    support_by_kind: dict[str, dict[str, int]] = {}
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
        for metric, support_key, fallback_key in CLARIFICATION_SUPPORTS:
            raw_support = score.get(support_key)
            if raw_support is None:
                raw_support = score.get(fallback_key)
            support = int(raw_support or 0)
            value = score.get(metric)
            if support == 0:
                # No cases on this side at all: the metric has no denominator.
                not_applicable.append(f"{split_name}.{metric} (support=0, undefined)")
                continue
            if value is not None and float(value) >= overall_minimum:
                continue
            if support >= MIN_OPERATIONAL_SUPPORT:
                findings.append(
                    f"{split_name}.{metric}={value!r} is below "
                    f"{overall_minimum:.4f} (support={support})"
                )
            else:
                canaries.append(
                    f"{split_name}.{metric}={value!r} (support={support}, below "
                    f"{overall_minimum:.4f} but under the {MIN_OPERATIONAL_SUPPORT}-row "
                    "minimum operational support)"
                )
        for kind, values in sorted((score.get("by_request_kind") or {}).items()):
            support = int(values.get("support") or values.get("rows") or 0)
            support_by_kind.setdefault(split_name, {})[kind] = support
            # regression 里 acquisition/library/conversation 各只有 1 条。n=1 时
            # 指标只能取 0.0 或 1.0，判错一次就是 0.0 直接卡门，判对也证明不了
            # 3pp 的测量能力。那种类别是"已知行为必须逐条通过"的 canary，不是
            # 统计结论。sealed 五类各 100 条，继续按统计硬门执行。
            enforced = split_name == "sealed" or support >= MIN_OPERATIONAL_SUPPORT
            for metric in CATEGORY_METRICS:
                # A metric the scorer declared undefined for these rows is not a
                # zero. conversation's lane_f1 is undefined because no gold row
                # asks for a lane; treating that null as 0.0 failed a category
                # whose rows were 100/101 exactly right.
                if values.get(f"{metric}_status") == NOT_APPLICABLE:
                    not_applicable.append(
                        f"{split_name}.{kind}.{metric} (no gold-positive rows; "
                        f"tool_set_exact_match="
                        f"{values.get('tool_set_exact_match_numerator')}/"
                        f"{values.get('tool_set_exact_match_denominator')})"
                    )
                    continue
                value = values.get(metric)
                below = value is None or float(value) < category_minimum
                if not below:
                    continue
                if enforced:
                    findings.append(
                        f"{split_name}.{kind}.{metric}={value!r} is below "
                        f"{category_minimum:.4f} (support={support})"
                    )
                else:
                    canaries.append(
                        f"{split_name}.{kind}.{metric}={value!r} (support={support}, "
                        f"below {category_minimum:.4f} but under the "
                        f"{MIN_OPERATIONAL_SUPPORT}-row minimum operational support)"
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
            "min_operational_support": MIN_OPERATIONAL_SUPPORT,
        },
        "sealed_minus_regression": split_deltas,
        "support_by_request_kind": support_by_kind,
        "findings": findings,
        # 不达标但支持数低于操作下限：照常报出数值，但不判死，也不声称统计置信度。
        "low_support_canaries": canaries,
        # 对这批行根本没有定义的指标。与"算出来是 0"必须分开看。
        "not_applicable": not_applicable,
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
