from scripts.p11_sync_knowledge_cache import load_sqlite_cards
from services.music_knowledge_store import MusicKnowledgeStore


def test_load_sqlite_cards_preserves_sources_and_summary_only(tmp_path):
    store = MusicKnowledgeStore(tmp_path / "knowledge.sqlite")
    store.upsert_artist_card(
        artist="Massive Attack",
        summary="Trip-hop group from Bristol.",
        style_tags=["Trip-Hop"],
        facts=["Known for atmospheric electronic music."],
        source_url="https://example.com/massive-attack",
        confidence=0.81,
    )
    store.upsert_song_card(
        title="Teardrop",
        artist="Massive Attack",
        summary="A 1998 trip-hop song.",
        release_year=1998,
        source_url="https://example.com/teardrop",
        confidence=0.84,
    )

    cards = load_sqlite_cards(store)

    assert {card["kind"] for card in cards} == {"artist", "song"}
    assert any(card["source_url"] == "https://example.com/massive-attack" for card in cards)
    assert any(card.get("release_year") == 1998 for card in cards)

