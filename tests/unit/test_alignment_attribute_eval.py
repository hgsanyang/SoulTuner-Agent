from __future__ import annotations

import pytest

import json

from tests.eval.evaluate_alignment_attribute import (
    label_matches,
    load_query_variant_file,
    precision_at_k,
    precision_means_by_bucket,
)


def test_label_matches_language_exact():
    assert label_matches({"language": "Chinese"}, {"field": "language", "equals": "chinese"})
    assert not label_matches({"language": "English"}, {"field": "language", "equals": "chinese"})


def test_label_matches_contains_any():
    labels = {"genres": ["Indie Rock", "Alternative"]}

    assert label_matches(labels, {"field": "genres", "contains_any": ["rock", "metal"]})
    assert not label_matches(labels, {"field": "genres", "contains_any": ["classical"]})


def test_precision_at_k_uses_top_k():
    labels = {
        "a": {"moods": ["melancholy"]},
        "b": {"moods": ["happy"]},
        "c": {"moods": ["lonely"]},
    }
    target = {"field": "moods", "contains_any": ["melancholy", "lonely"]}

    assert precision_at_k(["a", "b", "c"], labels, target, 2) == 0.5


def test_precision_at_k_rejects_non_positive_k():
    with pytest.raises(ValueError):
        precision_at_k(["a"], {}, {"field": "language", "equals": "chinese"}, 0)


def test_load_query_variant_file_dedupes_and_caps(tmp_path):
    path = tmp_path / "variants.json"
    path.write_text(
        json.dumps({
            "variants": {
                "q1": ["a", "a", "b", "c", "d", "e"],
                "q2": "not-a-list",
            }
        }),
        encoding="utf-8",
    )

    assert load_query_variant_file(str(path)) == {"q1": ["a", "b", "c", "d"]}


def test_precision_means_by_bucket_groups_nested_fields():
    rows = [
        {"target": {"field": "moods"}, "precision_at_k": {"muq": 0.8, "m2d": 0.2}},
        {"target": {"field": "moods"}, "precision_at_k": {"muq": 0.6, "m2d": 0.4}},
        {"target": {"field": "genres"}, "precision_at_k": {"muq": 0.3, "m2d": None}},
    ]

    metrics = precision_means_by_bucket(rows, ["muq", "m2d"], bucket_key="target.field")

    assert metrics["moods"]["muq"] == pytest.approx(0.7)
    assert metrics["moods"]["m2d"] == pytest.approx(0.3)
    assert metrics["genres"] == {"muq": 0.3, "m2d": 0.0}
