from tests.eval.memory_blind_metrics import (
    evaluate_preregistered_gates,
    paired_cluster_bootstrap_delta,
    paired_bootstrap_delta,
    percentile,
    score_observations,
)


def test_score_observations_counts_lifecycle_and_abstention() -> None:
    metrics = score_observations(
        [
            {
                "case_id": "positive",
                "category": "explicit",
                "expected_memory_ids": ["m1"],
                "retrieved_memory_ids": ["m1"],
                "hard_constraint_ok": True,
                "memory_latency_ms": 2,
                "memory_token_count": 8,
            },
            {
                "case_id": "negative",
                "category": "abstain",
                "expected_memory_ids": [],
                "retrieved_memory_ids": ["stale"],
                "stale_memory_ids": ["stale"],
                "deleted_memory_ids": ["stale"],
                "should_abstain": True,
                "memory_latency_ms": 6,
                "memory_token_count": 4,
            },
        ]
    )

    assert metrics["memory_precision"] == 0.5
    assert metrics["memory_recall"] == 1.0
    assert metrics["stale_memory_rate"] == 0.5
    assert metrics["deletion_residue_count"] == 1
    assert metrics["over_personalization_rate"] == 1.0
    assert metrics["memory_latency_p50_ms"] == 4.0
    assert metrics["memory_token_count"] == 12
    assert metrics["hard_constraint_coverage"] == 0.5


def test_paired_bootstrap_delta_is_deterministic() -> None:
    baseline = {f"c{i}": 0.0 for i in range(20)}
    candidate = {f"c{i}": 1.0 for i in range(20)}
    result = paired_bootstrap_delta(baseline, candidate, samples=200)

    assert result == paired_bootstrap_delta(baseline, candidate, samples=200)
    assert result["delta"] == 1.0
    assert result["ci_low"] == 1.0


def test_cluster_bootstrap_resamples_bundles_not_cases() -> None:
    baseline = {"a1": 0.0, "a2": 0.0, "b1": 1.0, "b2": 1.0}
    candidate = {"a1": 1.0, "a2": 1.0, "b1": 1.0, "b2": 1.0}
    result = paired_cluster_bootstrap_delta(
        baseline,
        candidate,
        {"a1": "a", "a2": "a", "b1": "b", "b2": "b"},
        samples=200,
    )

    assert result["n_cases"] == 4
    assert result["n_bundles"] == 2
    assert result["delta"] == 0.5


def test_preregistered_gates_require_gain_and_zero_leakage() -> None:
    arms = {
        "off": {"cross_user_leak_count": 0},
        "structured": {
            "cross_user_leak_count": 0,
            "deletion_residue_count": 0,
            "hard_constraint_violation_rate": 0.01,
        },
        "semantic": {
            "cross_user_leak_count": 0,
            "deletion_residue_count": 0,
            "hard_constraint_violation_rate": 0.02,
        },
        "sidecar": {
            "cross_user_leak_count": 0,
            "deletion_residue_count": 0,
            "hard_constraint_violation_rate": 0.03,
        },
    }
    gates = evaluate_preregistered_gates(
        arms,
        semantic_constraint_delta={"ci_low": 0.06},
        sidecar_constraint_delta={"ci_low": 0.04},
    )

    assert gates["cross_user_leak_zero"] is True
    assert gates["deletion_residue_zero"] is True
    assert gates["semantic_gain_gate"] is True
    assert gates["sidecar_gain_gate"] is False
    assert gates["semantic_violation_gate"] is True
    assert gates["sidecar_violation_gate"] is False


def test_percentile_interpolates() -> None:
    assert percentile([0, 10], 0.5) == 5.0


def test_missing_hard_constraint_is_unknown_not_automatic_pass() -> None:
    metrics = score_observations(
        [{"case_id": "memory-only", "expected_memory_ids": [], "retrieved_memory_ids": []}]
    )

    assert metrics["hard_constraint_coverage"] == 0.0
    assert metrics["constraint_at_10"] == 0.0
