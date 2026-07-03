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
        if "RETURN s.title AS title" in query:
            return [
                {
                    "title": "near seed",
                    "muq_emb": [1.0, 0.0],
                    "m2d_emb": None,
                    "omar_emb": [1.0, 0.0],
                },
                {
                    "title": "far seed",
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
