"""Query-relevant retrieval over the local auditable memory ledger."""

from __future__ import annotations

import math
import re
import time
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Protocol

from services.memory_models import MemoryLayer, MemoryRecord


# Locked on the open multilingual calibration fixture. The sealed blind-v2
# benchmark is only used as a final regression gate, never to tune these values.
RELEVANCE_POLICY_VERSION = "open-calibration-v1"

DEFAULT_LAYER_THRESHOLDS = {
    MemoryLayer.EXPLICIT: 0.063129,
    MemoryLayer.INFERRED: 0.074551,
    MemoryLayer.EPISODIC: 0.08,
}

# Scene-bound preference scopes. A memory scoped to one scene must never
# influence a different scene; when the current scene is unknown, scene-scoped
# memories stay out (fail-closed personalization). Lifecycle scopes
# (global/contextual/temporary) are scene-agnostic and always eligible.
SCENE_SCOPES = frozenset(
    {"driving", "commute", "focus", "sleep", "late_night", "rainy", "romantic", "workout"}
)
LIFECYCLE_SCOPES = frozenset({"global", "contextual", "temporary"})

# Structured planner scenario labels -> scene scope. This is enum
# normalization of typed planner output, not free-text keyword routing.
_SCENARIO_TO_SCENE = {
    "driving": "driving", "drive": "driving", "car": "driving",
    "commute": "commute", "commuting": "commute",
    "study": "focus", "work": "focus", "focus": "focus", "coding": "focus", "reading": "focus",
    "sleep": "sleep", "bedtime": "sleep",
    "late night": "late_night", "late_night": "late_night", "night": "late_night",
    "rainy": "rainy", "rain": "rainy", "rainy day": "rainy",
    "romantic": "romantic", "date": "romantic", "social": "romantic",
    "workout": "workout", "gym": "workout", "running": "workout", "exercise": "workout",
}

# Episodic memories decay faster than stable preferences: an event from last
# week should matter, one from two months ago usually should not.
EPISODIC_RECENCY_HALF_LIFE_DAYS = 14.0


def normalize_scene(scenario: str | None) -> str:
    """Map a structured scenario label to a scene scope; unknown -> ""."""
    return _SCENARIO_TO_SCENE.get(str(scenario or "").strip().casefold(), "")


@dataclass(frozen=True)
class RetrievedMemory:
    record: MemoryRecord
    score: float
    relevance: float
    why_used: str

    def model_dump(self) -> dict:
        canonical_id = str(self.record.payload.get("canonical_memory_id") or self.record.record_id)
        return {
            "memory_id": canonical_id,
            "record_id": self.record.record_id,
            "user_id": self.record.user_id,
            "layer": self.record.layer.value,
            "status": self.record.status.value,
            "memory_key": self.record.memory_key,
            "field": self.record.payload.get("field"),
            "value": self.record.payload.get("value"),
            "score": round(self.score, 4),
            "relevance": round(self.relevance, 4),
            "why_used": self.why_used,
            "expires_at": self.record.expires_at,
            "source": self.record.source,
            "evidence_ids": _evidence_ids(self.record),
            "scope": self.record.payload.get("scope", "global"),
            "confidence": round(self.record.confidence, 4),
            "created_at_ms": self.record.created_at,
            "expires_at_ms": self.record.expires_at,
        }


class SemanticMemoryScorer(Protocol):
    name: str

    def score(self, query: str, documents: list[str]) -> list[float]:
        ...


class MemoryRelevanceRetriever:
    """Lightweight multilingual relevance ranking without a network dependency.

    Consolidation stores natural-language retrieval cues. Character n-gram cosine
    matches those cues to the current query, while confidence and recency only
    break ties. This is memory selection, not a third song-recall engine.
    """

    def __init__(
        self,
        *,
        min_relevance: float = 0.08,
        semantic_scorer: SemanticMemoryScorer | None = None,
        layer_thresholds: dict[MemoryLayer | str, float] | None = None,
        max_per_layer: int | None = None,
    ):
        self.min_relevance = max(0.0, min(1.0, float(min_relevance)))
        self.semantic_scorer = semantic_scorer
        self.layer_thresholds = {
            MemoryLayer(str(layer.value if isinstance(layer, MemoryLayer) else layer)): max(
                0.0, min(1.0, float(threshold))
            )
            for layer, threshold in (layer_thresholds or {}).items()
        }
        self.max_per_layer = None if max_per_layer is None else max(1, int(max_per_layer))

    @property
    def backend_name(self) -> str:
        return self.semantic_scorer.name if self.semantic_scorer is not None else "char-ngram"

    def describe(self) -> dict:
        """Auditable retrieval policy for memory traces and reports."""
        return {
            "backend": self.backend_name,
            "policy_version": RELEVANCE_POLICY_VERSION,
            "min_relevance": self.min_relevance,
            "layer_thresholds": {
                layer.value: threshold for layer, threshold in self.layer_thresholds.items()
            },
            "max_per_layer": self.max_per_layer,
        }

    def retrieve(
        self,
        *,
        query: str,
        records: Iterable[MemoryRecord],
        max_facts: int = 8,
        include_episodic: bool = False,
        now_ms: int | None = None,
        scene: str = "",
    ) -> list[RetrievedMemory]:
        now = int(now_ms if now_ms is not None else time.time() * 1000)
        current_scene = str(scene or "").strip().casefold()
        eligible: list[tuple[MemoryRecord, float]] = []
        for record in records:
            record_scope = str(record.payload.get("scope") or "global").strip().casefold()
            if record_scope in SCENE_SCOPES and record_scope != current_scene:
                # Scene-bound memory outside its scene (or with unknown scene)
                # must not influence the response.
                continue
            if record.layer in {MemoryLayer.EXPLICIT, MemoryLayer.INFERRED}:
                eligible.append((record, 1.0))
            elif include_episodic and record.layer == MemoryLayer.EPISODIC:
                eligible.append((record, 0.7))
        texts = [_record_text(record) for record, _ in eligible]
        query_vector = _features(query)
        semantic_scores: list[float] | None = None
        if self.semantic_scorer is not None and str(query or "").strip() and texts:
            semantic_scores = self.semantic_scorer.score(str(query), texts)
            if len(semantic_scores) != len(texts):
                raise ValueError("semantic memory scorer returned an invalid score count")
        ranked: list[RetrievedMemory] = []
        for index, (record, layer_prior) in enumerate(eligible):
            text = _record_text(record)
            relevance = (
                semantic_scores[index]
                if semantic_scores is not None
                else _cosine(query_vector, _features(text)) if query_vector else 0.0
            )
            threshold = self.layer_thresholds.get(record.layer, self.min_relevance)
            if str(query or "").strip() and relevance < threshold:
                continue
            age_days = max(0.0, (now - record.created_at) / 86_400_000)
            # Episodic events lose applicability much faster than stable
            # preferences; this keeps current constraints dominant over
            # stale episodes without a hard cutoff.
            half_life = (
                EPISODIC_RECENCY_HALF_LIFE_DAYS
                if record.layer == MemoryLayer.EPISODIC
                else 45.0
            )
            recency = math.exp(-age_days / half_life)
            record_scope = str(record.payload.get("scope") or "global").strip().casefold()
            scene_matched = bool(current_scene) and record_scope == current_scene
            score = (
                0.68 * relevance
                + 0.17 * max(0.0, min(1.0, record.confidence))
                + 0.10 * recency
                + 0.05 * layer_prior
            )
            if scene_matched:
                score += 0.05  # bounded boost: scene-fit memory ranks first, never overrides relevance gates
            reason = (
                f"query relevance={relevance:.2f}; confidence={record.confidence:.2f}; "
                f"age_days={age_days:.1f}"
            )
            if scene_matched:
                reason += f"; scene={record_scope}"
            if record.layer == MemoryLayer.EPISODIC:
                reason += f"; episodic_half_life_days={half_life:.0f}"
            ranked.append(RetrievedMemory(record, score, relevance, reason))
        ranked.sort(key=lambda item: (item.score, item.record.created_at), reverse=True)
        if self.max_per_layer is not None:
            per_layer: Counter[MemoryLayer] = Counter()
            bounded: list[RetrievedMemory] = []
            for item in ranked:
                if per_layer[item.record.layer] >= self.max_per_layer:
                    continue
                per_layer[item.record.layer] += 1
                bounded.append(item)
            ranked = bounded
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


def _evidence_ids(record: MemoryRecord) -> list[str]:
    values = record.payload.get("evidence_ids") or []
    if not isinstance(values, list):
        values = []
    result = [str(value) for value in values if str(value).strip()]
    if record.evidence_id and record.evidence_id not in result:
        result.append(record.evidence_id)
    return result


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
