from services.knowledge_vector_index import (
    KnowledgeVectorUnavailable,
    QdrantKnowledgeIndex,
    payload_for_card,
    text_for_card,
)


def test_ensure_collection_returns_when_collection_exists(monkeypatch):
    calls = []
    index = QdrantKnowledgeIndex(url="http://localhost:6333", collection="test")

    def fake_request(method, path, **kwargs):
        calls.append((method, path, kwargs))
        return {"result": {"status": "green"}}

    monkeypatch.setattr(index, "_request", fake_request)
    index.ensure_collection()

    assert calls == [("GET", "/collections/test", {})]


def test_ensure_collection_ignores_existing_put_conflict(monkeypatch):
    calls = []
    index = QdrantKnowledgeIndex(url="http://localhost:6333", collection="test")

    def fake_request(method, path, **kwargs):
        calls.append((method, path))
        if method == "GET":
            raise KnowledgeVectorUnavailable("not found")
        raise KnowledgeVectorUnavailable("already exists")

    monkeypatch.setattr(index, "_request", fake_request)
    index.ensure_collection()

    assert calls == [("GET", "/collections/test"), ("PUT", "/collections/test")]


def test_payload_and_text_include_structured_details():
    card = {
        "kind": "song",
        "title": "Running Up That Hill",
        "artist": "Kate Bush",
        "summary": "A synth-pop song.",
        "facts": ["Originally released in 1985."],
        "style_tags": ["Synth-Pop"],
        "details": {"album": "Hounds of Love", "era": "1980s"},
        "release_year": 1985,
        "source_url": "https://example.com/song",
        "confidence": 0.9,
    }

    payload = payload_for_card(card)
    text = text_for_card(card)

    assert payload["details"]["album"] == "Hounds of Love"
    assert "Hounds of Love" in text
    assert "1980s" in text
