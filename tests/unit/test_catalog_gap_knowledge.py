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
    decision = analyze_catalog_gap(local, {}, "推荐80年代的老歌", web_enabled=True)

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
    decision = analyze_catalog_gap(local, {}, "讲讲 The Cure 的歌手背景和风格", web_enabled=True)

    assert "external_knowledge_required" not in decision.reasons
    assert decision.details["knowledge_evidence"]["query_hits"] >= 1

