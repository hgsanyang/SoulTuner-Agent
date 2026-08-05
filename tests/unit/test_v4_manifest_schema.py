"""A schema nobody has validated a document against is a schema-shaped comment.

data/sft/v4/MANIFEST.schema.json exists to make the two things the V3 audit could
not answer impossible to leave unanswered in a V4 build: whether a row's plan was
ever really executed, and whether the sealed split shares entities with train.

These tests drive it with one good manifest and the specific bad ones — each bad
case is a situation V3 is actually in today.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

jsonschema = pytest.importorskip("jsonschema")

V4 = Path(__file__).resolve().parents[2] / "data" / "sft" / "v4"


@pytest.fixture(scope="module")
def schema() -> dict:
    return json.loads((V4 / "MANIFEST.schema.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def validator(schema):
    from jsonschema import Draft202012Validator

    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _split(rows: int = 100) -> dict:
    return {
        "path": "data/sft/v4/train_v4.jsonl",
        "sha256": "a" * 64,
        "rows": rows,
        "counts_by_request_kind": {"recommendation": rows},
        "counts_by_trajectory": {"single_turn": rows},
    }


@pytest.fixture
def manifest() -> dict:
    return {
        "manifest_version": "1.0",
        "dataset_version": "v4.0.0",
        "created_at": "2026-07-29T12:00:00Z",
        "generator_commit": "626ee9c",
        "splits": {
            "train": _split(8000),
            "regression": _split(226),
            "sealed": _split(400),
        },
        "sealed_policy": {
            "entity_disjoint": True,
            "template_disjoint": True,
            "episode_namespace": "sealed_v4",
            "seed_pool": "user_logs_2026Q3_batch_b",
            "measured": {
                "shared_episodes": 0,
                "shared_artists": 0,
                "shared_songs": 0,
                "shared_templates": 0,
                "max_near_dupe_jaccard": 0.41,
            },
            "release_gates": {
                "overall_vs_teacher_pp": -3.0,
                "per_kind_max_regression_pp": 3.0,
                "schema_validity": 1.0,
                "lane_authority_violations": 0,
                "sealed_vs_regression_max_gap_pp": 8.0,
            },
        },
        "validator": {
            "tool": "scripts/validate_sft_dataset.py",
            "commit": "626ee9c",
            "hard_findings": 0,
        },
    }


def test_the_config_file_parses():
    json.loads((V4 / "validator_config.json").read_text(encoding="utf-8"))


def test_a_complete_manifest_is_accepted(validator, manifest):
    assert not list(validator.iter_errors(manifest))


def test_a_sealed_split_reusing_train_artists_is_rejected(validator, manifest):
    """Exactly the V3 eval's situation: 64 shared artists, episodes disjoint.
    That split cannot separate learning from recognising, so it must not be
    describable as sealed."""
    manifest["sealed_policy"]["measured"]["shared_artists"] = 64
    assert list(validator.iter_errors(manifest))


def test_a_near_duplicate_above_the_ceiling_is_rejected(validator, manifest):
    """V3's eval peaked at 0.84 Jaccard against train."""
    manifest["sealed_policy"]["measured"]["max_near_dupe_jaccard"] = 0.84
    assert list(validator.iter_errors(manifest))


def test_the_sealed_episode_namespace_must_be_prefixed(validator, manifest):
    manifest["sealed_policy"]["episode_namespace"] = "eval_"
    assert list(validator.iter_errors(manifest))


@pytest.mark.parametrize(
    "field", ["generator_commit", "dataset_version", "sealed_policy", "validator"]
)
def test_provenance_fields_are_not_optional(validator, manifest, field):
    """A dataset that cannot say which commit built it cannot be regenerated."""
    del manifest[field]
    assert list(validator.iter_errors(manifest))


def test_all_three_splits_are_required(validator, manifest):
    del manifest["splits"]["sealed"]
    assert list(validator.iter_errors(manifest))


def test_release_gates_are_frozen_before_scoring(validator, manifest):
    del manifest["sealed_policy"]["release_gates"]
    assert list(validator.iter_errors(manifest))


# ---- per-row provenance conditionals ----------------------------------------

@pytest.fixture
def provenance(schema):
    from jsonschema import Draft202012Validator

    return Draft202012Validator({**schema["$defs"]["sampleProvenance"], "$defs": schema["$defs"]})


@pytest.fixture
def row() -> dict:
    return {
        "seed_source": "real_user_query",
        "episode_id": "sealed_0001",
        "turn_id": 0,
        "request_kind": "recommendation",
        "trajectory_kind": "single_turn",
        "observation_origin": "none",
        "teacher": {"model": "qwen3.7-plus", "version": "2026-07"},
        "reviewer": {"model": "independent-reviewer", "version": "1"},
        "reviewer_verdict": "accept",
    }


def test_a_single_shot_planner_row_needs_no_trace(provenance, row):
    """All 1515 V3 rows are this shape: no observation exists to be traced, and
    demanding one would make the whole format invalid rather than honest."""
    assert not list(provenance.iter_errors(row))


def test_reviewer_verdict_is_required(provenance, row):
    del row["reviewer_verdict"]
    assert list(provenance.iter_errors(row))


def test_a_teacher_invented_failure_is_rejected(provenance, row):
    """An imagined failure teaches recovery from a failure mode the system does
    not have."""
    row["trajectory_kind"] = "failure_recovery"
    row["observation_origin"] = "teacher_authored"
    assert list(provenance.iter_errors(row))


def test_claiming_real_execution_without_a_trace_is_rejected(provenance, row):
    row["observation_origin"] = "real_execution"
    assert list(provenance.iter_errors(row))


def test_a_real_traced_failure_is_accepted(provenance, row):
    row["trajectory_kind"] = "failure_recovery"
    row["observation_origin"] = "real_execution"
    row["trace_id"] = "run-7f3a"
    row["execution_environment"] = "production"
    assert not list(provenance.iter_errors(row))


def test_a_controlled_harness_failure_is_accepted_and_cannot_claim_production(
    provenance, row
):
    row["trajectory_kind"] = "failure_recovery"
    row["observation_origin"] = "harness_execution"
    row["trace_id"] = "harness-timeout-001"
    row["execution_environment"] = "controlled_harness"
    assert not list(provenance.iter_errors(row))

    row["execution_environment"] = "production"
    assert list(provenance.iter_errors(row))


def test_real_execution_cannot_be_mislabeled_as_controlled_harness(provenance, row):
    row["observation_origin"] = "real_execution"
    row["trace_id"] = "run-7f3a"
    row["execution_environment"] = "controlled_harness"
    assert list(provenance.iter_errors(row))


def test_the_v3_baseline_in_the_config_matches_what_the_gate_reports():
    """The recorded baseline is what a future build gets compared against. If it
    drifts from the gate's own output it is worse than no baseline."""
    config = json.loads((V4 / "validator_config.json").read_text(encoding="utf-8"))
    baseline = config["v3_baseline_2026_07_29"]
    assert baseline["hard_findings"] == sum(baseline["hard_findings_breakdown"].values())
    assert baseline["rows_with_execution_trace"] == 0
    # the gate's hard tier must not silently start including a report-only check
    hard = set(config["hard"]["checks"])
    assert set(baseline["hard_findings_breakdown"]) <= hard
