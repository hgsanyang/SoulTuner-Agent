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


def test_plan_query_variants_default_to_m2d_only(monkeypatch):
    monkeypatch.setattr(semantic_search.settings, "plan_query_variant_mode", "m2d")

    assert semantic_search._should_apply_plan_query_variants("m2d")
    assert not semantic_search._should_apply_plan_query_variants("muq")


def test_plan_query_variants_can_enable_all_or_off(monkeypatch):
    monkeypatch.setattr(semantic_search.settings, "plan_query_variant_mode", "all")
    assert semantic_search._should_apply_plan_query_variants("muq")

    monkeypatch.setattr(semantic_search.settings, "plan_query_variant_mode", "off")
    assert not semantic_search._should_apply_plan_query_variants("m2d")


def test_semantic_search_filters_unplayable_song_nodes():
    where = semantic_search._playable_song_where("song")

    assert "song.audio_url IS NOT NULL" in where
    assert "unplayable_stub" in where


def test_dense_query_variant_auto_mode_does_not_use_fixed_phrases(monkeypatch):
    monkeypatch.setattr(semantic_search.settings, "dense_query_variant_mode", "auto")

    assert not semantic_search._should_use_dense_query_variants("需要安静温柔的雨天歌")
    assert not semantic_search._should_use_dense_query_variants("similar songs with the same vibe")


def test_dense_query_variant_manual_on_still_available(monkeypatch):
    monkeypatch.setattr(semantic_search.settings, "dense_query_variant_mode", "on")

    assert semantic_search._should_use_dense_query_variants("任何查询都用于离线 bake-off")
