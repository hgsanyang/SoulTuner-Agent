"""Qdrant-backed semantic index for offline music knowledge cards.

SQLite remains the authoritative local store for exact lookup and source
auditing.  This module mirrors card summaries into Qdrant so broad questions
like "gothic post-punk bands" can still find useful local knowledge before the
Agent decides to go online.
"""

from __future__ import annotations

import hashlib
import math
import re
import uuid
from typing import Any, Iterable, Mapping

import requests

from config.settings import settings
from services.music_knowledge_cache import knowledge_key

KNOWLEDGE_VECTOR_DIM = 384
DEFAULT_COLLECTION = "soultuner_music_knowledge"
TOKEN_RE = re.compile(r"[\w\u4e00-\u9fff]+", re.IGNORECASE)


class KnowledgeVectorUnavailable(RuntimeError):
    """Raised when Qdrant is unavailable or rejects a request."""


def _normalise_url(url: str) -> str:
    return str(url or "").rstrip("/")


def knowledge_card_key(card: Mapping[str, Any]) -> str:
    return knowledge_key(
        str(card.get("kind") or "song"),
        str(card.get("title") or ""),
        str(card.get("artist") or ""),
    )


def text_for_card(card: Mapping[str, Any]) -> str:
    """Build the text that is embedded into Qdrant."""

    facts = card.get("facts") or []
    style_tags = card.get("style_tags") or []
    details = card.get("details") or {}
    details_text = ""
    if isinstance(details, Mapping):
        details_text = " ".join(
            str(part or "")
            for part in (
                " ".join(str(key) for key in details.keys()),
                " ".join(str(value) for value in details.values()),
            )
        )
    return " ".join(
        str(part or "")
        for part in (
            card.get("kind"),
            card.get("title"),
            card.get("artist"),
            card.get("summary"),
            " ".join(str(tag) for tag in style_tags),
            " ".join(str(fact) for fact in facts),
            details_text,
            card.get("release_year"),
        )
        if part
    )


def hash_embed_text(text: str, *, dim: int = KNOWLEDGE_VECTOR_DIM) -> list[float]:
    """Deterministic lightweight text vector.

    This is intentionally dependency-free so Docker can start and index cards
    without downloading another embedding model.  It is a real vector index
    bridge, not a final semantic model; the encoder can later be swapped for
    DashScope embedding or BGE without changing the Qdrant contract.
    """

    vector = [0.0] * dim
    tokens = TOKEN_RE.findall(str(text or "").casefold())
    features: list[str] = []
    for token in tokens:
        features.append(token)
        if len(token) > 2:
            features.extend(token[i : i + 3] for i in range(0, max(1, len(token) - 2)))
    if not features:
        return vector
    for feature in features:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] & 1 else -1.0
        vector[bucket] += sign
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 0:
        return vector
    return [round(value / norm, 8) for value in vector]


def qdrant_point_id(key: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"soultuner:{key}"))


def payload_for_card(card: Mapping[str, Any]) -> dict[str, Any]:
    key = knowledge_card_key(card)
    return {
        "key": key,
        "kind": str(card.get("kind") or "song"),
        "title": str(card.get("title") or ""),
        "artist": str(card.get("artist") or ""),
        "summary": str(card.get("summary") or ""),
        "facts": list(card.get("facts") or [])[:8],
        "style_tags": list(card.get("style_tags") or [])[:12],
        "details": dict(card.get("details") or {}) if isinstance(card.get("details"), Mapping) else {},
        "release_year": card.get("release_year"),
        "source": str(card.get("source") or card.get("source_provider") or "web"),
        "source_url": str(card.get("source_url") or ""),
        "confidence": float(card.get("confidence") or 0.5),
    }


def card_from_payload(payload: Mapping[str, Any], *, score: float | None = None) -> dict[str, Any]:
    card = dict(payload)
    card["kind"] = str(card.get("kind") or "song")
    if score is not None:
        card["_vector_score"] = round(float(score), 6)
    return card


class QdrantKnowledgeIndex:
    """Minimal Qdrant HTTP client for knowledge-card vectors."""

    def __init__(
        self,
        *,
        url: str | None = None,
        collection: str | None = None,
        timeout: float | None = None,
        dim: int = KNOWLEDGE_VECTOR_DIM,
    ) -> None:
        self.url = _normalise_url(url or settings.qdrant_url)
        self.collection = collection or getattr(settings, "knowledge_qdrant_collection", DEFAULT_COLLECTION)
        self.timeout = float(timeout if timeout is not None else getattr(settings, "knowledge_qdrant_timeout_seconds", 0.8))
        self.dim = dim

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        try:
            response = requests.request(
                method,
                f"{self.url}{path}",
                timeout=self.timeout,
                **kwargs,
            )
        except requests.RequestException as exc:
            raise KnowledgeVectorUnavailable(str(exc)) from exc
        if response.status_code >= 400:
            raise KnowledgeVectorUnavailable(f"Qdrant HTTP {response.status_code}: {response.text[:160]}")
        if not response.content:
            return {}
        return response.json()

    def ensure_collection(self) -> None:
        try:
            self._request("GET", f"/collections/{self.collection}")
            return
        except KnowledgeVectorUnavailable:
            pass
        try:
            self._request(
                "PUT",
                f"/collections/{self.collection}",
                json={"vectors": {"size": self.dim, "distance": "Cosine"}},
            )
        except KnowledgeVectorUnavailable as exc:
            if "already exists" not in str(exc):
                raise

    def upsert_cards(self, cards: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
        points = []
        for card in cards:
            payload = payload_for_card(card)
            text = text_for_card(payload)
            points.append(
                {
                    "id": qdrant_point_id(payload["key"]),
                    "vector": hash_embed_text(text, dim=self.dim),
                    "payload": payload,
                }
            )
        if not points:
            return {"upserted": 0}
        self.ensure_collection()
        self._request(
            "PUT",
            f"/collections/{self.collection}/points",
            params={"wait": "true"},
            json={"points": points},
        )
        return {"upserted": len(points), "collection": self.collection}

    def search(
        self,
        query: str,
        *,
        kind: str | None = None,
        limit: int = 5,
        min_confidence: float = 0.0,
    ) -> list[dict[str, Any]]:
        vector = hash_embed_text(query, dim=self.dim)
        if not any(vector):
            return []
        body: dict[str, Any] = {
            "vector": vector,
            "limit": max(1, int(limit) * 2),
            "with_payload": True,
        }
        try:
            data = self._request("POST", f"/collections/{self.collection}/points/search", json=body)
            raw = data.get("result") or []
        except KnowledgeVectorUnavailable:
            body = {"query": vector, "limit": max(1, int(limit) * 2), "with_payload": True}
            data = self._request("POST", f"/collections/{self.collection}/points/query", json=body)
            raw = data.get("result", {}).get("points") or data.get("result") or []
        cards = []
        for row in raw:
            payload = row.get("payload") or {}
            if kind and payload.get("kind") != kind:
                continue
            if float(payload.get("confidence") or 0.0) < min_confidence:
                continue
            cards.append(card_from_payload(payload, score=row.get("score")))
            if len(cards) >= limit:
                break
        return cards


def upsert_cards_to_qdrant(cards: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    return QdrantKnowledgeIndex().upsert_cards(cards)


def search_qdrant_knowledge(
    query: str,
    *,
    kind: str | None = None,
    limit: int = 5,
    min_confidence: float = 0.0,
) -> list[dict[str, Any]]:
    return QdrantKnowledgeIndex().search(query, kind=kind, limit=limit, min_confidence=min_confidence)
