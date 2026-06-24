"""Pure metric helpers for text-to-audio alignment evaluation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence


def first_relevant_rank(ranked_ids: Sequence[str], relevant_ids: Iterable[str]) -> int | None:
    """Return the 1-based rank of the first relevant id, or None when absent."""
    relevant = {str(item) for item in relevant_ids if item is not None}
    if not relevant:
        return None
    for index, item_id in enumerate(ranked_ids, start=1):
        if str(item_id) in relevant:
            return index
    return None


def recall_at_k(ranks: Sequence[int | None], k: int) -> float:
    """Compute Recall@K from 1-based ranks."""
    if k <= 0:
        raise ValueError("k must be positive")
    if not ranks:
        return 0.0
    hits = sum(1 for rank in ranks if rank is not None and rank <= k)
    return hits / len(ranks)


def mean_reciprocal_rank(ranks: Sequence[int | None]) -> float:
    """Compute MRR from 1-based ranks."""
    if not ranks:
        return 0.0
    return sum(0.0 if rank is None else 1.0 / rank for rank in ranks) / len(ranks)


def summarize_alignment_ranks(ranks: Sequence[int | None], ks: Sequence[int] = (1, 5, 10)) -> dict:
    """Return Recall@K and MRR for a list of ranks."""
    return {
        "count": len(ranks),
        **{f"recall_at_{k}": recall_at_k(ranks, k) for k in ks},
        "mrr": mean_reciprocal_rank(ranks),
        "misses": sum(1 for rank in ranks if rank is None),
    }
