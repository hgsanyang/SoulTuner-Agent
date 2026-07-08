from scripts.p15_knowledge_source_audit import build_report
from services.music_knowledge_store import MusicKnowledgeStore


def test_knowledge_source_audit_reports_quality_breakdown(tmp_path):
    store = MusicKnowledgeStore(tmp_path / "knowledge.sqlite")
    store.upsert_song_card(
        title="Song A",
        artist="Artist A",
        summary="A sourced song.",
        release_year=2000,
        source_url="https://music.apple.com/us/song/song-a/1",
        confidence=0.9,
    )
    store.upsert_artist_card(
        artist="Artist B",
        summary="A sourced artist.",
        source_url="https://baike.baidu.com/item/artist-b",
        confidence=0.8,
    )

    report = build_report(tmp_path / "knowledge.sqlite")

    assert report["total_cards"] == 2
    assert report["song_release_year_cards"] == 1
    assert report["quality_breakdown"]["high"] == 1
    assert report["quality_breakdown"]["medium"] == 1
