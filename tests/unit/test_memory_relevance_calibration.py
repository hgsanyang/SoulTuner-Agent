import json
from pathlib import Path

import pytest

from tests.eval.evaluate_memory_relevance_calibration import (
    DEFAULT_FIXTURE,
    attach_scores,
    calibrate,
    calibrate_thresholds,
    choose_rank_policy,
    choose_threshold,
    load_examples,
    print_report,
    score_examples,
)


def _example(
    example_id: str,
    *,
    group_id: str,
    level: str,
    query: str,
    memory: str,
    label: int,
) -> dict:
    return {
        "id": example_id,
        "group_id": group_id,
        "level": level,
        "language": "en",
        "category": "test",
        "query": query,
        "memory": memory,
        "label": label,
        "tags": ["synthetic_test"],
    }


def test_open_fixture_has_required_multilingual_coverage() -> None:
    payload = json.loads(Path(DEFAULT_FIXTURE).read_text(encoding="utf-8"))
    rows = load_examples()

    assert payload["sealed"] is False
    assert "synthetic" in payload["provenance"].lower()
    assert len(rows) >= 60
    assert {row["level"] for row in rows} == {"L1", "L2"}
    assert {row["language"] for row in rows} == {"zh", "en"}
    assert len({row["group_id"] for row in rows}) >= 20
    assert {row["category"] for row in rows}.issuperset(
        {
            "mood_add",
            "mood_avoid",
            "genre_add",
            "genre_avoid",
            "scenario_add",
            "scenario_avoid",
            "artist_add",
            "artist_avoid",
            "language_add",
            "language_avoid",
            "current_override",
            "unrelated",
            "non_recommendation",
            "expired_context",
        }
    )


def test_fixture_contains_positive_override_and_abstention_labels() -> None:
    rows = load_examples()

    assert any(row["label"] == 1 and "genuine_positive" in row["tags"] for row in rows)
    assert any(row["label"] == 0 and "explicit_override" in row["tags"] for row in rows)
    assert any(row["label"] == 0 and "stale_like_mismatch" in row["tags"] for row in rows)
    assert all(
        row["label"] == 0
        for row in rows
        if row["category"] in {"unrelated", "non_recommendation"}
    )


def test_level_thresholds_maximize_recall_at_precision_target() -> None:
    rows = [
        {"id": "l1-a", "level": "L1", "label": 1, "score": 0.95},
        {"id": "l1-b", "level": "L1", "label": 1, "score": 0.90},
        {"id": "l1-c", "level": "L1", "label": 0, "score": 0.85},
        {"id": "l1-d", "level": "L1", "label": 1, "score": 0.80},
        {"id": "l2-a", "level": "L2", "label": 1, "score": 0.78},
        {"id": "l2-b", "level": "L2", "label": 1, "score": 0.70},
        {"id": "l2-c", "level": "L2", "label": 0, "score": 0.65},
        {"id": "l2-d", "level": "L2", "label": 0, "score": 0.20},
    ]

    thresholds = calibrate_thresholds(rows, target_precision=0.85)

    assert thresholds["L1"]["threshold"] == 0.90
    assert thresholds["L1"]["metrics"]["precision"] == 1.0
    assert thresholds["L1"]["metrics"]["recall"] == pytest.approx(2 / 3)
    assert thresholds["L2"]["threshold"] == 0.70
    assert thresholds["L2"]["metrics"]["recall"] == 1.0


def test_threshold_failure_reports_false_positives_and_recall() -> None:
    result = choose_threshold(
        [
            {"id": "negative", "label": 0, "score": 0.99},
            {"id": "positive", "label": 1, "score": 0.90},
        ],
        target_precision=0.85,
    )

    assert result["target_met"] is False
    assert result["metrics"]["recall"] == 1.0
    assert result["metrics"]["false_positives"] == 1
    assert result["metrics"]["false_positive_rate"] == 1.0


def test_rank_policy_uses_top_k_and_smallest_useful_margin() -> None:
    rows = [
        {"id": "g1-positive", "group_id": "g1", "level": "L1", "label": 1, "score": 0.95},
        {"id": "g1-negative", "group_id": "g1", "level": "L1", "label": 0, "score": 0.85},
        {"id": "g2-positive-a", "group_id": "g2", "level": "L2", "label": 1, "score": 0.90},
        {"id": "g2-positive-b", "group_id": "g2", "level": "L2", "label": 1, "score": 0.88},
        {"id": "g2-negative", "group_id": "g2", "level": "L2", "label": 0, "score": 0.70},
    ]

    policy = choose_rank_policy(
        rows,
        {"L1": 0.50, "L2": 0.50},
        target_precision=0.85,
        top_ks=(1, 2),
        margins=(0.0, 0.03, 0.10),
    )

    assert policy["top_k"] == 2
    assert policy["margin"] == 0.03
    assert policy["metrics"]["precision"] == 1.0
    assert policy["metrics"]["recall"] == 1.0
    assert policy["metrics"]["false_positives"] == 0


def test_compact_score_rows_merge_by_id_and_require_full_coverage() -> None:
    examples = [
        _example("a", group_id="g1", level="L1", query="q1", memory="m1", label=1),
        _example("b", group_id="g2", level="L2", query="q2", memory="m2", label=0),
    ]

    merged = attach_scores(examples, [{"id": "b", "score": 0.2}, {"id": "a", "score": 0.9}])

    assert [(row["id"], row["score"]) for row in merged] == [("a", 0.9), ("b", 0.2)]
    with pytest.raises(ValueError, match="missing example ids"):
        attach_scores(examples, [{"id": "a", "score": 0.9}])


def test_injected_batch_scorer_is_used_without_model_imports() -> None:
    class StubScorer:
        def __init__(self) -> None:
            self.calls = []

        def score(self, query: str, documents: list[str]) -> list[float]:
            self.calls.append((query, documents))
            return [0.9 if "match" in document else 0.1 for document in documents]

    examples = [
        _example("a", group_id="g", level="L1", query="same query", memory="match memory", label=1),
        _example("b", group_id="g", level="L1", query="same query", memory="other memory", label=0),
    ]
    scorer = StubScorer()

    rows = score_examples(examples, scorer)

    assert scorer.calls == [("same query", ["match memory", "other memory"])]
    assert [row["score"] for row in rows] == [0.9, 0.1]


def test_end_to_end_report_is_precision_focused_and_prints_false_positives(capsys) -> None:
    examples = load_examples()
    scored = [{**row, "score": 0.9 if row["label"] else 0.1} for row in examples]

    report = calibrate(scored)
    print_report(report)
    output = capsys.readouterr().out

    assert report["example_count"] == len(examples)
    assert report["thresholds"]["L1"]["threshold"] == 0.9
    assert report["thresholds"]["L2"]["threshold"] == 0.9
    assert report["rank_policy"]["metrics"]["precision"] == 1.0
    assert report["rank_policy"]["metrics"]["recall"] == 1.0
    assert report["precision_goal_met"] is True
    assert "false_positives=0" in output
    assert "L1 final" in output
    assert "L2 final" in output
