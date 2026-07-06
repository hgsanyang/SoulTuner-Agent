"""Tests for the R1 multi-recall fusion and hard-filter boundary."""

from retrieval.retrieval_fusion import (
    apply_hard_filters,
    normalize_song_key,
    recall_weights_for_intent,
    weighted_rrf,
)


def _item(title: str, artist: str, rank: int = 0, **song_fields) -> dict:
    song = {
        "title": title,
        "artist": artist,
        "language": song_fields.pop("language", "Chinese"),
        "region": song_fields.pop("region", "Mainland China"),
        **song_fields,
    }
    return {
        "key": normalize_song_key(title, artist),
        "rank": rank,
        "raw_score": 1.0,
        "song": song,
    }


def test_content_recall_sources_remain_enabled_for_every_intent():
    for intent_type in ("graph_search", "hybrid_search", "vector_search", "unknown"):
        weights = recall_weights_for_intent(intent_type)
        assert set(weights) == {"graph", "dense"}
        assert all(weight > 0 for weight in weights.values())


def test_intent_only_changes_recall_weights():
    graph = recall_weights_for_intent("graph_search")
    vector = recall_weights_for_intent("vector_search")

    assert graph["graph"] > graph["dense"]
    assert vector["dense"] > vector["graph"]


def test_entity_query_prioritizes_graph_content_recall():
    weights = recall_weights_for_intent(
        "hybrid_search",
        query="我想听周杰伦的歌",
        hard_constraints={"artist_entities": ["周杰伦"], "song_entities": []},
    )

    assert weights["graph"] > weights["dense"]


def test_scene_query_prioritizes_dense_audio_context_without_extra_sources():
    weights = recall_weights_for_intent(
        "hybrid_search",
        query="需要偏柔软安静的雨天专注歌单",
        soft_intent={"goal": "柔软安静，适合雨天专注"},
        hints={"mood": "Calm", "scenario": "Study"},
    )

    assert weights["dense"] > weights["graph"]
    assert "personal" not in weights
    assert "cold" not in weights
    assert "lexical" not in weights


def test_similarity_seed_keeps_dense_ahead_of_entity_exactness():
    weights = recall_weights_for_intent(
        "hybrid_search",
        query="有没有类似 Running Up That Hill 听感的歌",
        hard_constraints={"song_entities": ["Running Up That Hill"]},
        soft_intent={"vibe": "similar sound"},
    )

    assert weights["dense"] > weights["graph"]


def test_weighted_rrf_rewards_cross_source_hits():
    sources = {
        "graph": [
            _item("青花瓷", "周杰伦", rank=0),
            _item("稻香", "周杰伦", rank=1),
        ],
        "dense": [
            _item("稻香", "周杰伦", rank=0),
            _item("夜曲", "周杰伦", rank=1),
        ],
    }

    fused = weighted_rrf(sources, recall_weights_for_intent("hybrid_search"))

    assert fused[0]["song"]["title"] == "稻香"
    assert fused[0]["_source_ranks"] == {"graph": 2, "dense": 1}
    assert fused[0]["_rrf_score"] > fused[1]["_rrf_score"]


def test_weighted_rrf_merges_richer_metadata():
    sources = {
        "graph": [_item("A", "Artist", genres=["Indie"], preview_url=None)],
        "dense": [
            _item(
                "A",
                "Artist",
                genres=["Rock"],
                moods=["Dreamy"],
                preview_url="http://localhost/audio/a.flac",
            )
        ],
    }

    song = weighted_rrf(sources, recall_weights_for_intent("hybrid_search"))[0]["song"]

    assert song["genres"] == ["Indie", "Rock"]
    assert song["moods"] == ["Dreamy"]
    assert song["preview_url"].endswith("a.flac")


def test_hard_filter_applies_entities_language_region_instrumental_and_dislikes():
    candidates = weighted_rrf(
        {
            "dense": [
                _item(
                    "Focus Piano",
                    "Alice",
                    rank=0,
                    language="Instrumental",
                    region="Japan",
                ),
                _item(
                    "Focus Piano",
                    "Bob",
                    rank=1,
                    language="Instrumental",
                    region="Japan",
                ),
                _item(
                    "Focus Piano",
                    "Alice",
                    rank=2,
                    language="English",
                    region="Western",
                ),
            ]
        },
        recall_weights_for_intent("vector_search"),
    )

    filtered = apply_hard_filters(
        candidates,
        {
            "artist_entities": ["Alice"],
            "song_entities": ["Focus Piano"],
            "language": "Instrumental",
            "region": "Japan",
            "instrumental": True,
        },
    )

    assert [(item["song"]["title"], item["song"]["artist"]) for item in filtered] == [
        ("Focus Piano", "Alice")
    ]
    assert apply_hard_filters(filtered, {}, disliked_titles={"Focus Piano"}) == []


def test_soft_hints_are_not_hard_filters():
    candidates = weighted_rrf(
        {
            "dense": [
                _item("Quiet Song", "A", moods=["Peaceful"], scenarios=["Study"]),
                _item("Loud Song", "B", moods=["Energetic"], scenarios=["Party"]),
            ]
        },
        recall_weights_for_intent("vector_search"),
    )

    # The hard filter API intentionally has no mood/scenario/genre arguments.
    filtered = apply_hard_filters(candidates, {})

    assert len(filtered) == 2


def test_normalize_song_key_handles_width_case_and_punctuation():
    assert normalize_song_key("青花瓷（Live）", "Jay Chou") == normalize_song_key(
        "青花瓷(Live)",
        "JAYCHOU",
    )


def test_weighted_rrf_handles_empty_sources():
    assert weighted_rrf({}, recall_weights_for_intent("hybrid_search")) == []


def test_source_weight_can_change_fused_order_without_disabling_a_source():
    sources = {
        "graph": [_item("Graph First", "A", rank=0)],
        "dense": [_item("Dense First", "B", rank=0)],
    }

    graph_order = weighted_rrf(sources, recall_weights_for_intent("graph_search"))
    vector_order = weighted_rrf(sources, recall_weights_for_intent("vector_search"))

    assert graph_order[0]["song"]["title"] == "Graph First"
    assert vector_order[0]["song"]["title"] == "Dense First"


def test_unknown_language_and_region_do_not_conflict_with_hard_constraints():
    candidates = weighted_rrf(
        {
            "dense": [
                _item("Unknown Labels", "A", language="Unknown", region=None),
                _item("English Song", "B", language="English", region="Western"),
            ]
        },
        recall_weights_for_intent("vector_search"),
    )

    filtered = apply_hard_filters(candidates, {"language": "Japanese", "region": "Japan"})

    assert [(item["song"]["title"], item["song"]["artist"]) for item in filtered] == [
        ("Unknown Labels", "A")
    ]


def test_hard_filter_falls_back_to_safety_filtered_rrf_top_k_when_too_sparse():
    candidates = weighted_rrf(
        {
            "dense": [
                _item("Unknown First", "A", rank=0, language="Unknown", region=None),
                _item("Unknown Second", "B", rank=1, language=None, region=None),
                _item("Blocked", "C", rank=2, language="Unknown", region=None),
                _item("English Conflict", "D", rank=3, language="English", region="Western"),
            ]
        },
        recall_weights_for_intent("vector_search"),
    )

    class Logger:
        def __init__(self):
            self.messages = []

        def warning(self, message, *args):
            self.messages.append(message % args)

    logger = Logger()
    filtered = apply_hard_filters(
        candidates,
        {"language": "Japanese", "region": "Japan"},
        disliked_titles={"Blocked"},
        limit=3,
        logger=logger,
    )

    assert [item["song"]["title"] for item in filtered] == [
        "Unknown First",
        "Unknown Second",
        "English Conflict",
    ]
    assert logger.messages
    assert logger.messages[0].startswith("[HardFilterFallback]")


def test_hard_filter_keeps_strict_language_matches_ahead_of_unknown_labels():
    candidates = weighted_rrf(
        {
            "dense": [
                _item("Unknown First", "A", rank=0, language="Unknown", region=None),
                _item("Cantonese Song", "B", rank=1, language="Cantonese", region="Hong Kong"),
            ]
        },
        recall_weights_for_intent("vector_search"),
    )

    filtered = apply_hard_filters(candidates, {"language": "Cantonese"}, limit=8)

    assert [item["song"]["title"] for item in filtered] == [
        "Cantonese Song",
        "Unknown First",
    ]


def test_entity_alias_matching_is_case_and_spacing_insensitive():
    candidates = weighted_rrf(
        {"graph": [_item("Love Story", "Taylor Swift")]},
        recall_weights_for_intent("graph_search"),
    )

    filtered = apply_hard_filters(
        candidates,
        {
            "artist_entities": ["taylor  swift"],
            "song_entities": ["love story"],
        },
    )

    assert len(filtered) == 1
