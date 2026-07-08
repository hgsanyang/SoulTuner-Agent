from services.recommendation_knowledge_backfill import select_missing_knowledge_songs
from services.music_knowledge_store import MusicKnowledgeStore


def test_select_missing_knowledge_songs_skips_complete_cards(tmp_path):
    store = MusicKnowledgeStore(tmp_path / "knowledge.sqlite")
    store.upsert_song_card(
        title="Complete",
        artist="Artist",
        summary="A sourced card.",
        release_year=2001,
        source_url="https://example.com/complete",
        confidence=0.9,
    )

    selected = select_missing_knowledge_songs(
        [
            {"song": {"title": "Complete", "artist": "Artist", "release_year": 2001}},
            {"song": {"title": "Missing", "artist": "Artist"}},
        ],
        store=store,
        limit=5,
    )

    assert selected == [{"title": "Missing", "artist": "Artist"}]


def test_select_missing_knowledge_songs_respects_limit(tmp_path):
    store = MusicKnowledgeStore(tmp_path / "knowledge.sqlite")
    selected = select_missing_knowledge_songs(
        [
            {"title": "A", "artist": "X"},
            {"title": "B", "artist": "X"},
            {"title": "C", "artist": "X"},
        ],
        store=store,
        limit=2,
    )

    assert [row["title"] for row in selected] == ["A", "B"]
