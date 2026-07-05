from services.music_knowledge_store import MusicKnowledgeStore


def test_music_knowledge_store_upserts_and_searches_artist(tmp_path):
    store = MusicKnowledgeStore(tmp_path / "knowledge.sqlite")

    card = store.upsert_artist_card(
        artist="The Cure",
        summary="English post-punk and gothic rock band with atmospheric guitar textures.",
        style_tags=["Post-Punk", "Gothic Rock"],
        facts=["Formed in Crawley."],
        source_url="https://example.com/the-cure",
        confidence=0.82,
    )

    assert card["artist"] == "The Cure"
    hits = store.search("The Cure gothic rock", kind="artist", min_confidence=0.7)
    assert hits[0]["artist"] == "The Cure"
    assert hits[0]["source_url"] == "https://example.com/the-cure"
    assert "Gothic Rock" in hits[0]["style_tags"]


def test_music_knowledge_store_upserts_song_release_year_and_source(tmp_path):
    store = MusicKnowledgeStore(tmp_path / "knowledge.sqlite")

    store.upsert_song_card(
        title="Running Up That Hill",
        artist="Kate Bush",
        summary="A synth-pop song originally released in 1985.",
        release_year=1985,
        style_tags=["Synth-Pop", "Art Pop"],
        facts=["Originally released in 1985."],
        source_url="https://example.com/running-up-that-hill",
        confidence=0.9,
    )

    card = store.get_song_card("Running Up That Hill", "Kate Bush")
    assert card is not None
    assert card["release_year"] == 1985
    assert card["source_url"] == "https://example.com/running-up-that-hill"

    hits = store.search("1985 synth pop Kate Bush", kind="song", min_confidence=0.8)
    assert hits[0]["title"] == "Running Up That Hill"


def test_music_knowledge_store_normalized_payload_keeps_style_and_year(tmp_path):
    store = MusicKnowledgeStore(tmp_path / "knowledge.sqlite")

    card = store.upsert_normalized_card(
        {
            "kind": "song",
            "title": "恋曲1980",
            "artist": "罗大佑",
            "summary": "华语流行里的经典民谣摇滚作品。",
            "facts": ["常被归入华语经典老歌。"],
            "style_tags": ["Folk", "Chinese Pop"],
            "release_year": 1982,
            "source": "web",
            "source_url": "https://example.com/1980",
            "confidence": 0.76,
        }
    )

    assert card["release_year"] == 1982
    assert "Chinese Pop" in card["style_tags"]

