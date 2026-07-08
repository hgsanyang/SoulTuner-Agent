from services.music_knowledge_enrichment import (
    WebSnippet,
    build_card_from_snippets,
    extract_release_year,
    infer_style_tags,
    normalize_snippets,
    _extract_json_object,
    _llm_web_card,
    _normalise_llm_card,
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


def test_extract_json_object_accepts_wrapped_model_output():
    parsed = _extract_json_object('```json\n{"summary":"ok","confidence":0.8}\n```')

    assert parsed == {"summary": "ok", "confidence": 0.8}


def test_llm_summary_path_can_override_deterministic_card(monkeypatch):
    snippets = [
        WebSnippet(
            title="Band profile",
            content="Slowdive are associated with shoegaze and dream pop.",
            url="https://example.com/slowdive",
            source="web",
        )
    ]

    monkeypatch.setattr(
        "services.music_knowledge_enrichment._llm_card_from_snippets",
        lambda **kwargs: {
            "kind": "artist",
            "title": "Slowdive",
            "artist": "Slowdive",
            "summary": "LLM structured summary.",
            "facts": ["Shoegaze band."],
            "style_tags": ["Shoegaze"],
            "source": "web",
            "source_url": "https://example.com/slowdive",
            "confidence": 0.91,
        },
    )

    card = build_card_from_snippets(kind="artist", title="Slowdive", artist="Slowdive", snippets=snippets, use_llm_summary=True)

    assert card["summary"] == "LLM structured summary."
    assert card["confidence"] == 0.91


def test_qwen_web_search_card_parses_responses_api_json(monkeypatch):
    class FakeResponses:
        def create(self, **kwargs):
            assert kwargs["model"] == "qwen3.7-plus"
            assert kwargs["tools"] == [{"type": "web_search"}]

            class Response:
                output_text = (
                    '{"summary":"The Cure 是英国后朋克/哥特摇滚乐队。",'
                    '"facts":["代表作包括 Boys Don’t Cry。"],'
                    '"style_tags":["Post-Punk","Gothic Rock"],'
                    '"release_year":null,'
                    '"details":{"country_or_region":"United Kingdom","artist_type":"band"},'
                    '"confidence":0.86,'
                    '"sources":["https://example.com/the-cure"]}'
                )

            return Response()

    class FakeClient:
        responses = FakeResponses()

    monkeypatch.setenv("MODEL_NAME", "deepseek-ai/DeepSeek-V3.2")
    monkeypatch.setattr("services.music_knowledge_enrichment._dashscope_openai_client", lambda: FakeClient())
    monkeypatch.setattr("services.music_knowledge_enrichment.settings.llm_default_model", "qwen3.7-plus", raising=False)

    card = _llm_web_card(kind="artist", query="The Cure 音乐人 简介 风格", artist="The Cure", title="The Cure")

    assert card is not None
    assert card["source"] == "dashscope_web_search"
    assert card["source_url"] == "https://example.com/the-cure"
    assert "Gothic Rock" in card["style_tags"]
    assert card["details"]["artist_type"] == "band"


def test_llm_card_rejects_search_aggregator_source_urls():
    card = _normalise_llm_card(
        kind="song",
        title="Some Song",
        artist="Some Artist",
        parsed={
            "summary": "A sourced-looking but unsupported summary.",
            "facts": ["Fact"],
            "style_tags": ["Pop"],
            "release_year": 2000,
            "confidence": 0.9,
            "sources": ["https://tavily.com/search?q=some-song"],
        },
    )

    assert card is None


def test_llm_card_allows_non_search_baike_source_urls():
    card = _normalise_llm_card(
        kind="song",
        title="Some Song",
        artist="Some Artist",
        parsed={
            "summary": "A supported summary.",
            "facts": ["Fact"],
            "style_tags": ["Pop"],
            "release_year": 2000,
            "confidence": 0.9,
            "sources": ["https://baike.baidu.com/item/some-song"],
        },
    )

    assert card is not None
    assert card["source_url"].startswith("https://baike.baidu.com/")
