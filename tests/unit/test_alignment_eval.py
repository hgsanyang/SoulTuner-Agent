from tests.eval.alignment_metrics import (
    first_relevant_rank,
    mean_reciprocal_rank,
    recall_at_k,
    summarize_alignment_ranks,
)


def test_first_relevant_rank_single_id():
    assert first_relevant_rank(["a", "b", "c"], ["b"]) == 2
    assert first_relevant_rank(["a", "b", "c"], ["x"]) is None


def test_first_relevant_rank_multiple_relevant_ids():
    assert first_relevant_rank(["a", "b", "c"], ["c", "b"]) == 2


def test_recall_at_k_counts_hits():
    ranks = [1, 3, 10, 11, None]
    assert recall_at_k(ranks, 1) == 0.2
    assert recall_at_k(ranks, 5) == 0.4
    assert recall_at_k(ranks, 10) == 0.6


def test_recall_at_k_rejects_invalid_k():
    try:
        recall_at_k([1], 0)
    except ValueError as exc:
        assert "k must be positive" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_mrr_treats_misses_as_zero():
    assert mean_reciprocal_rank([1, 2, None]) == (1.0 + 0.5 + 0.0) / 3


def test_summarize_alignment_ranks():
    summary = summarize_alignment_ranks([1, 2, 7, None], ks=(1, 5, 10))
    assert summary["count"] == 4
    assert summary["recall_at_1"] == 0.25
    assert summary["recall_at_5"] == 0.5
    assert summary["recall_at_10"] == 0.75
    assert summary["misses"] == 1
