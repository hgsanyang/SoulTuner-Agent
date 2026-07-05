from services.music_knowledge_enrichment import (
    WebSnippet,
    build_card_from_snippets,
    extract_release_year,
    infer_style_tags,
    normalize_snippets,
)


def test_normalize_snippets_dedupes_and_keeps_source_url():
    snippets = normalize_snippets(
        [
            {"title": "A", "content": "Rock band biography", "url": "https://example.com/a", "source": "searxng"},
            {"title": "A copy", "content": "Duplicate", "url": "https://example.com/a", "source": "tavily"},
            {"title": "Empty", "content": "", "url": "https://example.com/empty"},
        ]
    )

    assert len(snippets) == 1
    assert snippets[0].url == "https://example.com/a"
    assert snippets[0].source == "searxng"


def test_build_card_from_snippets_infers_style_year_and_confidence():
    card = build_card_from_snippets(
        kind="song",
        title="Running Up That Hill",
        artist="Kate Bush",
        snippets=[
            WebSnippet(
                title="Running Up That Hill",
                content="Running Up That Hill is a 1985 art pop and synth-pop song by Kate Bush.",
                url="https://example.com/song",
                source="searxng",
            ),
            WebSnippet(
                title="Kate Bush song",
                content="The track is known for electronic drums and atmospheric pop production.",
                url="https://example.com/song-2",
                source="tavily",
            ),
        ],
    )

    assert card is not None
    assert card["release_year"] == 1985
    assert card["source_url"] == "https://example.com/song"
    assert card["confidence"] >= 0.72
    assert "Pop" in card["style_tags"] or "Electronic" in card["style_tags"]


def test_style_and_year_helpers_are_conservative():
    tags = infer_style_tags("quiet lo-fi indie folk guitar")
    assert {"Folk", "Indie", "Lo-fi"}.issubset(set(tags))
    assert extract_release_year("released in 1899 then remastered") is None
    assert extract_release_year("originally released in 1994") == 1994
