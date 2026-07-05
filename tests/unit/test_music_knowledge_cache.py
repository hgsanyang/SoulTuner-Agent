from services.music_knowledge_cache import MusicKnowledgeCache, knowledge_key


def test_knowledge_key_distinguishes_song_and_artist():
    assert knowledge_key("artist", artist="Teleman") == "artist::teleman"
    assert knowledge_key("song", title="Anchor", artist="Teleman") == "song::anchor::teleman"


def test_music_knowledge_cache_upserts_and_searches(tmp_path):
    cache = MusicKnowledgeCache(tmp_path / "knowledge.jsonl")
    cache.upsert(
        {
            "kind": "song",
            "title": "Anchor",
            "artist": "Teleman",
            "summary": "A warm indie pop song with gentle rhythm.",
            "facts": ["Released by Teleman."],
            "source": "web",
        }
    )

    loaded = cache.get(kind="song", title="Anchor", artist="Teleman")
    assert loaded is not None
    assert loaded["summary"].startswith("A warm")

    hits = cache.search_terms(["gentle", "teleman"])
    assert len(hits) == 1
    assert hits[0]["title"] == "Anchor"
