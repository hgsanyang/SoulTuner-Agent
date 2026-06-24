from __future__ import annotations

import pytest

from tests.eval.evaluate_alignment_attribute import label_matches, precision_at_k


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
