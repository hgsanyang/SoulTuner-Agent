"""Implementation-independent metrics for the sealed Memory v2 benchmark."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Iterable


def _ids(value: Any) -> set[str]:
    if not isinstance(value, (list, tuple, set)):
        return set()
    return {str(item) for item in value if str(item).strip()}


def _ratio(numerator: int | float, denominator: int | float, *, empty: float = 0.0) -> float:
    return float(numerator) / float(denominator) if denominator else float(empty)


def percentile(values: Iterable[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    quantile = min(1.0, max(0.0, float(quantile)))
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def score_observations(observations: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(observations)
    true_positive = 0
    retrieved_total = 0
    expected_total = 0
    stale_total = 0
    cross_user_total = 0
    cross_user_incidents = 0
    deletion_total = 0
    deletion_incidents = 0
    hard_constraint_passes = 0
    hard_constraint_evaluable = 0
    over_personalized = 0
    abstain_cases = 0
    token_total = 0
    latencies: list[float] = []
    category_values: dict[str, list[float]] = defaultdict(list)

    for row in rows:
        expected = _ids(row.get("expected_memory_ids"))
        retrieved = _ids(row.get("retrieved_memory_ids"))
        stale = _ids(row.get("stale_memory_ids")) & retrieved
        cross_user = _ids(row.get("cross_user_memory_ids")) & retrieved
        deleted = _ids(row.get("deleted_memory_ids")) & retrieved
        matched = expected & retrieved
        hard_constraint_known = row.get("hard_constraint_ok") is not None
        hard_constraint_ok = bool(row.get("hard_constraint_ok")) if hard_constraint_known else True
        should_abstain = bool(row.get("should_abstain", not expected))
        is_over_personalized = should_abstain and bool(retrieved)

        true_positive += len(matched)
        retrieved_total += len(retrieved)
        expected_total += len(expected)
        stale_total += len(stale)
        cross_user_total += len(cross_user)
        cross_user_incidents += int(bool(cross_user))
        deletion_total += len(deleted)
        deletion_incidents += int(bool(deleted))
        hard_constraint_evaluable += int(hard_constraint_known)
        hard_constraint_passes += int(hard_constraint_known and hard_constraint_ok)
        over_personalized += int(is_over_personalized)
        abstain_cases += int(should_abstain)
        token_total += max(0, int(row.get("memory_token_count") or 0))
        latencies.append(max(0.0, float(row.get("memory_latency_ms") or 0.0)))

        if expected:
            memory_success = _ratio(len(matched), len(expected))
        else:
            memory_success = 1.0 if not retrieved else 0.0
        passed = (
            memory_success == 1.0
            and hard_constraint_ok
            and not stale
            and not cross_user
            and not deleted
            and not is_over_personalized
        )
        category = str(row.get("category") or "uncategorized")
        category_values[category].append(float(passed))

    return {
        "case_count": len(rows),
        "memory_precision": _ratio(true_positive, retrieved_total, empty=1.0),
        "memory_recall": _ratio(true_positive, expected_total, empty=1.0),
        "stale_memory_rate": _ratio(stale_total, retrieved_total),
        "cross_user_leak_count": cross_user_total,
        "cross_user_leak_incidence": _ratio(cross_user_incidents, len(rows)),
        "deletion_residue_count": deletion_total,
        "deletion_residue_incidence": _ratio(deletion_incidents, len(rows)),
        "constraint_at_10": _ratio(hard_constraint_passes, hard_constraint_evaluable),
        "hard_constraint_coverage": _ratio(hard_constraint_evaluable, len(rows)),
        "over_personalization_rate": _ratio(over_personalized, abstain_cases),
        "hard_constraint_violation_rate": _ratio(
            hard_constraint_evaluable - hard_constraint_passes,
            hard_constraint_evaluable,
        ),
        "memory_latency_p50_ms": percentile(latencies, 0.50),
        "memory_latency_p95_ms": percentile(latencies, 0.95),
        "memory_token_count": token_total,
        "category_pass_rate": {
            category: sum(values) / len(values)
            for category, values in sorted(category_values.items())
        },
    }


def paired_bootstrap_delta(
    baseline: dict[str, float],
    candidate: dict[str, float],
    *,
    samples: int = 5000,
    seed: int = 20260713,
) -> dict[str, float | int]:
    case_ids = sorted(set(baseline) & set(candidate))
    if not case_ids:
        return {"n": 0, "delta": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    deltas = [float(candidate[key]) - float(baseline[key]) for key in case_ids]
    point = sum(deltas) / len(deltas)
    rng = random.Random(seed)
    estimates = [
        sum(deltas[rng.randrange(len(deltas))] for _ in deltas) / len(deltas)
        for _ in range(max(1, int(samples)))
    ]
    return {
        "n": len(case_ids),
        "delta": point,
        "ci_low": percentile(estimates, 0.025),
        "ci_high": percentile(estimates, 0.975),
    }


def paired_cluster_bootstrap_delta(
    baseline: dict[str, float],
    candidate: dict[str, float],
    case_to_bundle: dict[str, str],
    *,
    samples: int = 10000,
    seed: int = 271828,
) -> dict[str, float | int]:
    common = sorted(set(baseline) & set(candidate) & set(case_to_bundle))
    by_bundle: dict[str, list[float]] = defaultdict(list)
    for case_id in common:
        by_bundle[str(case_to_bundle[case_id])].append(
            float(candidate[case_id]) - float(baseline[case_id])
        )
    bundles = sorted(by_bundle)
    if not bundles:
        return {"n_cases": 0, "n_bundles": 0, "delta": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    bundle_means = [sum(by_bundle[key]) / len(by_bundle[key]) for key in bundles]
    point = sum(bundle_means) / len(bundle_means)
    rng = random.Random(seed)
    estimates = [
        sum(bundle_means[rng.randrange(len(bundle_means))] for _ in bundles) / len(bundles)
        for _ in range(max(1, int(samples)))
    ]
    return {
        "n_cases": len(common),
        "n_bundles": len(bundles),
        "delta": point,
        "ci_low": percentile(estimates, 0.025),
        "ci_high": percentile(estimates, 0.975),
    }


def evaluate_preregistered_gates(
    arms: dict[str, dict[str, Any]],
    *,
    semantic_constraint_delta: dict[str, float] | None = None,
    sidecar_constraint_delta: dict[str, float] | None = None,
) -> dict[str, bool]:
    structured = arms.get("structured", {})
    semantic = arms.get("semantic", {})
    sidecar = arms.get("sidecar", {})
    structured_violation = float(structured.get("hard_constraint_violation_rate") or 0.0)
    semantic_delta = semantic_constraint_delta or {}
    sidecar_delta = sidecar_constraint_delta or {}
    return {
        "cross_user_leak_zero": all(
            int(metrics.get("cross_user_leak_count") or 0) == 0
            for metrics in arms.values()
        ),
        "deletion_residue_zero": all(
            int(metrics.get("deletion_residue_count") or 0) == 0
            for name, metrics in arms.items()
            if name != "off"
        ),
        "semantic_gain_gate": float(semantic_delta.get("ci_low") or 0.0) >= 0.05,
        "sidecar_gain_gate": float(sidecar_delta.get("ci_low") or 0.0) >= 0.05,
        "semantic_violation_gate": (
            float(semantic.get("hard_constraint_violation_rate") or 0.0)
            - structured_violation
        ) <= 0.01,
        "sidecar_violation_gate": (
            float(sidecar.get("hard_constraint_violation_rate") or 0.0)
            - structured_violation
        ) <= 0.01,
    }
