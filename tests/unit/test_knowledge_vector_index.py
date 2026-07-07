from services.knowledge_vector_index import KnowledgeVectorUnavailable, QdrantKnowledgeIndex


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
