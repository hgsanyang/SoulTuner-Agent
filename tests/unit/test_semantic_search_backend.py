import pytest

from tools import semantic_search


@pytest.mark.parametrize("backend", ["muq", "m2d", "both"])
def test_dense_backend_accepts_supported_values(monkeypatch, backend):
    monkeypatch.setattr(semantic_search.settings, "dense_text_audio_backend", backend)

    assert semantic_search._dense_backend() == backend


def test_dense_backend_falls_back_to_muq_for_unknown_value(monkeypatch):
    monkeypatch.setattr(semantic_search.settings, "dense_text_audio_backend", "unknown")

    assert semantic_search._dense_backend() == "muq"


def test_backend_specs_use_matching_neo4j_vectors():
    assert semantic_search._backend_spec("muq") == {
        "name": "MuQ-MuLan",
        "index": "song_muq_index",
        "property": "muq_embedding",
        "source": "Neo4j SemanticSearch (MuQ-MuLan)",
    }
    assert semantic_search._backend_spec("m2d")["index"] == "song_m2d2_index"
    assert semantic_search._backend_spec("m2d")["property"] == "m2d2_embedding"
