import json

from services.catalog_enrichment import (
    build_artist_knowledge_query,
    build_song_knowledge_query,
    extract_release_year,
    normalize_acquisition_metadata,
    normalize_knowledge_card,
    prepare_tag_enrichment,
)


def test_prepare_tag_enrichment_cleans_caps_and_tracks_provenance():
    enriched = prepare_tag_enrichment(
        {
            "genres": [" Rock ", "rock", "Folk", "Indie", "Pop", "Dream Pop", "Electronic"],
            "moods": ["unknown", "Healing"],
            "tempo": ["120bpm"],
        },
        source="llm_lyrics",
    )

    assert enriched["genres"] == ["Rock", "Folk", "Indie", "Pop", "Dream Pop"]
    assert enriched["moods"] == ["Healing"]
    assert enriched["themes"] == []
    assert enriched["scenarios"] == []

    confidence = json.loads(enriched["tag_confidence_json"])
    sources = json.loads(enriched["tag_sources_json"])
    assert confidence["genres"]["Rock"] == 0.72
    assert sources["moods"]["Healing"] == "llm_lyrics"


def test_extract_release_year_from_netease_publish_time():
    assert extract_release_year({"publishTime": 915148800000}) == 1999
    assert extract_release_year({"release_date": "1985-07-01"}) == 1985
    assert extract_release_year({"release_year": "2012"}) == 2012
    assert extract_release_year({"release_year": "12"}) is None


def test_normalize_acquisition_metadata_keeps_only_observed_fields():
    normalized = normalize_acquisition_metadata(
        {
            "musicId": 123,
            "musicName": "Song A",
            "artist": [["Artist A", 1], ["Artist B", 2]],
            "album": "Album A",
            "duration": 240000,
            "format": "mp3",
            "source": "online",
            "source_platform": "netease",
            "publishTime": 915148800000,
        }
    )

    assert normalized["music_id"] == "123"
    assert normalized["artist"] == "Artist A、Artist B"
    assert normalized["release_year"] == 1999
    assert normalized["metadata_source"] == "netease"


def test_normalize_knowledge_card_is_bounded_and_source_aware():
    card = normalize_knowledge_card(
        {
            "kind": "artist",
            "artist": "Teleman",
            "summary": "A" * 2000,
            "facts": [f"fact-{i}" for i in range(20)],
            "source": "web",
            "source_url": "https://example.com/teleman",
        }
    )

    assert card["kind"] == "artist"
    assert card["confidence"] == 0.68
    assert len(card["summary"]) == 900
    assert len(card["facts"]) == 8


def test_knowledge_queries_are_music_specific():
    assert "发行" in build_song_knowledge_query("稻香", "周杰伦")
    assert "代表作" in build_artist_knowledge_query("周杰伦")
