from agent.catalog_gap import analyze_catalog_gap
from config.settings import settings
from services.music_knowledge_store import MusicKnowledgeStore


def _song(title, artist="A", **extra):
    song = {"title": title, "artist": artist, "preview_url": "u"}
    song.update(extra)
    return {"song": song}


def test_release_year_gap_uses_local_knowledge_cards(monkeypatch, tmp_path):
    store = MusicKnowledgeStore(tmp_path / "knowledge.sqlite")
    for index in range(5):
        store.upsert_song_card(
            title=f"Old Song {index}",
            artist="A",
            summary="A classic 1980s song.",
            release_year=1985,
            source_url=f"https://example.com/{index}",
            confidence=0.86,
        )
    monkeypatch.setattr(settings, "knowledge_store_path", str(store.path), raising=False)
    monkeypatch.setattr(settings, "knowledge_gap_enabled", True, raising=False)
    monkeypatch.setattr(settings, "knowledge_gap_min_confidence", 0.55, raising=False)

    local = [_song(f"Old Song {index}") for index in range(12)]
    decision = analyze_catalog_gap(
        local,
        {"metadata_constraints": {"release_year_from": 1980, "release_year_to": 1989, "era": "80s"}},
        "推荐80年代的老歌",
        web_enabled=True,
    )

    assert "metadata_release_year_missing" not in decision.reasons
    assert decision.details["knowledge_evidence"]["local_song_release_year_hits"] == 5


def test_external_knowledge_request_uses_local_knowledge_store(monkeypatch, tmp_path):
    store = MusicKnowledgeStore(tmp_path / "knowledge.sqlite")
    store.upsert_artist_card(
        artist="The Cure",
        summary="The Cure are an English post-punk and gothic rock band.",
        style_tags=["Post-Punk", "Gothic Rock"],
        facts=["Known for atmospheric guitar textures."],
        source_url="https://example.com/the-cure",
        confidence=0.9,
    )
    monkeypatch.setattr(settings, "knowledge_store_path", str(store.path), raising=False)
    monkeypatch.setattr(settings, "knowledge_gap_enabled", True, raising=False)
    monkeypatch.setattr(settings, "knowledge_gap_min_confidence", 0.55, raising=False)

    local = [_song(f"Song {index}", artist="The Cure") for index in range(12)]
    decision = analyze_catalog_gap(
        local,
        {"metadata_constraints": {"external_knowledge_required": True}},
        "讲讲 The Cure 的歌手背景和风格",
        web_enabled=True,
    )

    assert "external_knowledge_required" not in decision.reasons
    assert decision.details["knowledge_evidence"]["query_hits"] >= 1


def test_external_knowledge_request_records_qdrant_evidence(monkeypatch, tmp_path):
    store = MusicKnowledgeStore(tmp_path / "knowledge.sqlite")
    store.initialize()
    monkeypatch.setattr(settings, "knowledge_store_path", str(store.path), raising=False)
    monkeypatch.setattr(settings, "knowledge_gap_enabled", True, raising=False)
    monkeypatch.setattr(settings, "knowledge_gap_min_confidence", 0.55, raising=False)
    monkeypatch.setattr(settings, "knowledge_vector_backend", "qdrant", raising=False)
    monkeypatch.setattr(
        "services.knowledge_vector_index.search_qdrant_knowledge",
        lambda *args, **kwargs: [
            {
                "kind": "artist",
                "artist": "Fishmans",
                "summary": "Japanese dub, dream pop and psychedelic pop band.",
                "source_url": "https://example.com/fishmans",
                "confidence": 0.88,
                "_vector_score": 0.73,
                "style_tags": ["Dub", "Dream Pop"],
            }
        ],
    )

    local = [_song(f"Song {index}", artist="Fishmans") for index in range(12)]
    decision = analyze_catalog_gap(
        local,
        {"metadata_constraints": {"external_knowledge_required": True}},
        "讲讲 Fishmans 的迷幻梦幻流行风格",
        web_enabled=True,
    )

    evidence = decision.details["knowledge_evidence"]
    assert "external_knowledge_required" not in decision.reasons
    assert evidence["query_vector_hits"] == 1
    assert evidence["cards"][0]["retrieval_source"] == "qdrant"
