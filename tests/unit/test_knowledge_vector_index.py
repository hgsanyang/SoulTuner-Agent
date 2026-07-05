import pytest

from services.knowledge_vector_index import (
    KNOWLEDGE_VECTOR_DIM,
    QdrantKnowledgeIndex,
    hash_embed_text,
    knowledge_card_key,
    payload_for_card,
    qdrant_point_id,
)


def test_hash_embed_text_is_deterministic_and_normalized():
    first = hash_embed_text("The Cure gothic rock post punk")
    second = hash_embed_text("The Cure gothic rock post punk")

    assert first == second
    assert len(first) == KNOWLEDGE_VECTOR_DIM
    assert 0.99 <= sum(value * value for value in first) <= 1.01


def test_payload_for_card_is_qdrant_safe():
    card = {
        "kind": "artist",
        "artist": "The Cure",
        "summary": "English post-punk band.",
        "style_tags": ["Post-Punk"],
        "source_url": "https://example.com/the-cure",
        "confidence": 0.9,
    }

    payload = payload_for_card(card)

    assert payload["key"] == "artist::the cure"
    assert payload["source_url"] == "https://example.com/the-cure"
    assert qdrant_point_id(payload["key"]) == qdrant_point_id(payload["key"])
    assert knowledge_card_key(payload) == payload["key"]


def test_qdrant_index_upsert_and_search_use_http_contract(monkeypatch):
    calls = []

    class Response:
        status_code = 200
        content = b"{}"
        text = "{}"

        def __init__(self, body=None):
            self._body = body or {}

        def json(self):
            return self._body

    def fake_request(method, url, timeout, **kwargs):
        calls.append({"method": method, "url": url, "timeout": timeout, **kwargs})
        if url.endswith("/points/search"):
            return Response(
                {
                    "result": [
                        {
                            "score": 0.8,
                            "payload": {
                                "kind": "artist",
                                "artist": "The Cure",
                                "summary": "Post-punk band.",
                                "confidence": 0.9,
                            },
                        }
                    ]
                }
            )
        return Response({"result": "ok"})

    monkeypatch.setattr("services.knowledge_vector_index.requests.request", fake_request)
    index = QdrantKnowledgeIndex(url="http://qdrant:6333", collection="test_cards", timeout=0.2)

    result = index.upsert_cards([
        {"kind": "artist", "artist": "The Cure", "summary": "Post-punk band.", "confidence": 0.9}
    ])
    hits = index.search("gothic rock", kind="artist", min_confidence=0.5)

    assert result["upserted"] == 1
    assert hits[0]["artist"] == "The Cure"
    assert hits[0]["_vector_score"] == pytest.approx(0.8)
    assert any(call["method"] == "PUT" and call["url"].endswith("/collections/test_cards") for call in calls)
    assert any(call["method"] == "PUT" and call["url"].endswith("/points") for call in calls)
    assert any(call["method"] == "POST" and call["url"].endswith("/points/search") for call in calls)

