"""Query-relevant retrieval over the local auditable memory ledger."""

from __future__ import annotations

import math
import re
import time
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from services.memory_models import MemoryLayer, MemoryRecord


@dataclass(frozen=True)
class RetrievedMemory:
    record: MemoryRecord
    score: float
    relevance: float
    why_used: str

    def model_dump(self) -> dict:
        return {
            "record_id": self.record.record_id,
            "layer": self.record.layer.value,
            "field": self.record.payload.get("field"),
            "value": self.record.payload.get("value"),
            "score": round(self.score, 4),
            "relevance": round(self.relevance, 4),
            "why_used": self.why_used,
            "expires_at": self.record.expires_at,
            "source": self.record.source,
        }


class MemoryRelevanceRetriever:
    """Lightweight multilingual relevance ranking without a network dependency.

    Consolidation stores natural-language retrieval cues. Character n-gram cosine
    matches those cues to the current query, while confidence and recency only
    break ties. This is memory selection, not a third song-recall engine.
    """

    def __init__(self, *, min_relevance: float = 0.08):
        self.min_relevance = max(0.0, min(1.0, float(min_relevance)))

    def retrieve(
        self,
        *,
        query: str,
        records: Iterable[MemoryRecord],
        max_facts: int = 8,
        include_episodic: bool = False,
        now_ms: int | None = None,
    ) -> list[RetrievedMemory]:
        now = int(now_ms if now_ms is not None else time.time() * 1000)
        query_vector = _features(query)
        ranked: list[RetrievedMemory] = []
        for record in records:
            if record.layer == MemoryLayer.INFERRED:
                layer_prior = 1.0
            elif include_episodic and record.layer == MemoryLayer.EPISODIC:
                layer_prior = 0.7
            else:
                continue
            text = _record_text(record)
            relevance = _cosine(query_vector, _features(text)) if query_vector else 0.0
            if query_vector and relevance < self.min_relevance:
                continue
            age_days = max(0.0, (now - record.created_at) / 86_400_000)
            recency = math.exp(-age_days / 45.0)
            score = (
                0.68 * relevance
                + 0.17 * max(0.0, min(1.0, record.confidence))
                + 0.10 * recency
                + 0.05 * layer_prior
            )
            reason = (
                f"query relevance={relevance:.2f}; confidence={record.confidence:.2f}; "
                f"age_days={age_days:.1f}"
            )
            ranked.append(RetrievedMemory(record, score, relevance, reason))
        ranked.sort(key=lambda item: (item.score, item.record.created_at), reverse=True)
        return ranked[: max(1, min(int(max_facts), 20))]


def _record_text(record: MemoryRecord) -> str:
    payload = record.payload
    values: list[str] = [
        str(payload.get("user_text") or ""),
        str(payload.get("field") or ""),
        str(payload.get("value") or ""),
        str(payload.get("decision_summary") or ""),
        str(payload.get("description") or ""),
    ]
    cues = payload.get("retrieval_cues") or []
    if isinstance(cues, list):
        values.extend(str(value or "") for value in cues)
    return " ".join(value for value in values if value).strip()


def _features(text: str) -> Counter[str]:
    normalized = re.sub(r"\s+", " ", str(text or "").casefold()).strip()
    if not normalized:
        return Counter()
    compact = re.sub(r"\s+", "", normalized)
    features: Counter[str] = Counter()
    for token in re.findall(r"[\w]+", normalized, flags=re.UNICODE):
        features[f"w:{token}"] += 1
    for size in (2, 3):
        for index in range(max(0, len(compact) - size + 1)):
            features[f"c{size}:{compact[index:index + size]}"] += 1
    return features


def _cosine(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(value * right.get(key, 0) for key, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0
