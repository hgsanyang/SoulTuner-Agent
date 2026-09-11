import logging
import sys
import types

from retrieval.hybrid_retrieval import MusicHybridRetrieval


class _Neo4jClient:
    driver = object()

    def __init__(self):
        self.calls = []

    def execute_query(self, query, params=None):
        self.calls.append({"query": query, "params": params or {}})
        if "RETURN identity.key AS candidate_key" in query:
            return [
                {
                    "title": "near seed",
                    "candidate_key": "1",
                    "muq_emb": [1.0, 0.0],
                    "m2d_emb": None,
                    "omar_emb": [1.0, 0.0],
                },
                {
                    "title": "far seed",
                    "candidate_key": "0",
                    "muq_emb": [1.0, 0.0],
                    "m2d_emb": None,
                    "omar_emb": [-1.0, 0.0],
                },
            ]
        if "RETURN s.omar_embedding AS omar_emb" in query:
            return [{"omar_emb": [1.0, 0.0]}]
        return []


def test_tri_anchor_uses_reference_song_omar_seed(monkeypatch, caplog):
    fake_muq = types.ModuleType("retrieval.muq_embedder")
    fake_muq.encode_text_to_muq = lambda _text: [1.0, 0.0]
    monkeypatch.setitem(sys.modules, "retrieval.muq_embedder", fake_muq)

    neo4j = _Neo4jClient()
    import retrieval.neo4j_client as neo4j_client

    monkeypatch.setattr(neo4j_client, "get_neo4j_client", lambda: neo4j)

    retriever = MusicHybridRetrieval()
    retriever._current_reference_song_entities = ["seed song"]
    candidates = [
        {"song": {"title": "far seed", "artist": "B"}, "similarity_score": 0.9},
        {"song": {"title": "near seed", "artist": "A"}, "similarity_score": 0.8},
    ]

    with caplog.at_level(logging.INFO):
        ranked = retriever._tri_anchor_rerank(candidates, "similar to seed song")

    assert "OMAR 声学锚使用参考种子歌" in caplog.text
    assert [item["song"]["title"] for item in ranked] == ["near seed", "far seed"]
    assert ranked[0]["_acoustic_score"] > ranked[1]["_acoustic_score"]
    assert ranked[0]["_semantic_score"] == ranked[1]["_semantic_score"]


def test_same_title_uses_separate_candidate_identity(monkeypatch):
    fake_muq = types.ModuleType("retrieval.muq_embedder")
    fake_muq.encode_text_to_muq = lambda _: [1., 0.]
    monkeypatch.setitem(sys.modules, "retrieval.muq_embedder", fake_muq)
    class Client:
        driver = object()
        def execute_query(self, query, params):
            assert [i["music_id"] for i in params["identities"]] == ["a", "b"]
            assert "size(matches) = 1" in query
            return [{"candidate_key": "0", "muq_emb": [1., 0.]},
                    {"candidate_key": "1", "muq_emb": [-1., 0.]}]
    monkeypatch.setattr("retrieval.neo4j_client.get_neo4j_client", lambda: Client())
    ranked = MusicHybridRetrieval()._tri_anchor_rerank([
        {"song": {"music_id": "a", "title": "Same", "artist": "A"}},
        {"song": {"music_id": "b", "title": "Same", "artist": "B"}},
    ], "quiet")
    assert ranked[0]["_semantic_score"] > ranked[1]["_semantic_score"]


def test_muq_failure_never_calls_m2d(monkeypatch):
    fake_muq = types.ModuleType("retrieval.muq_embedder")
    def broken(_):
        raise RuntimeError("encoder unavailable")
    fake_muq.encode_text_to_muq = broken
    monkeypatch.setitem(sys.modules, "retrieval.muq_embedder", fake_muq)
    fake_m2d = types.ModuleType("retrieval.m2d_embedder")
    def forbidden(*args, **kwargs):
        raise AssertionError("must not switch semantic backend")
    fake_m2d.encode_text = forbidden
    monkeypatch.setitem(sys.modules, "retrieval.m2d_embedder", fake_m2d)
    monkeypatch.setattr("retrieval.neo4j_client.get_neo4j_client", lambda: _Neo4jClient())
    candidates = [{"song": {"title": "keep", "artist": "A"}, "similarity_score": .8}]
    ranked = MusicHybridRetrieval()._tri_anchor_rerank(candidates, "quiet")
    assert ranked[0]["song"]["title"] == "keep"
    assert "semantic_encoder_unavailable" in ranked[0]["retrieval_warnings"]
