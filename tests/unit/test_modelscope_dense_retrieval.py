from __future__ import annotations

import base64

from deploy.modelscope_space import retrieval_demo as retrieval


def _plan() -> dict:
    return {
        "hard": {"language": None},
        "soft": {},
        "hints": {"mood": [], "scenario": ["旅行"], "genre": ["摇滚"]},
        "metadata": {"era": None},
        "acoustic_queries": ["energetic vocal rock music for a road trip"],
    }


def _row(song_id: str, genre: str) -> dict:
    return {
        "song_id": song_id,
        "title": song_id,
        "artist": "Open Artist",
        "genres": [genre],
        "moods_themes": [],
        "scenarios": [],
        "instruments": [],
    }


def test_retrieval_uses_real_aura_m2d2_scores_when_available(monkeypatch) -> None:
    monkeypatch.setattr(
        retrieval,
        "load_catalog",
        lambda: (_row("rock", "alternative rock road trip"), _row("ambient", "ambient")),
    )
    monkeypatch.setattr(retrieval, "resolve_audio_source", lambda _row: None)
    monkeypatch.setattr(retrieval, "encode_text_query", lambda _text: [0.1] * 768)
    monkeypatch.setattr(
        retrieval,
        "vector_query_scores",
        lambda *_args, **_kwargs: ({"rock": 0.94, "ambient": 0.12}, {"state": "ready"}),
    )

    rows = retrieval.retrieve(
        "公路旅行摇滚",
        _plan(),
        {"graph_weight": 0.25, "dense_weight": 0.75},
        top_k=2,
    )

    assert rows[0]["song_id"] == "rock"
    assert rows[0]["dense_source"] == "m2d2_aura"
    assert rows[0]["dense_score"] == 0.94
    assert rows[0]["graph_score"] == 1.0


def test_chinese_genre_and_scene_aliases_match_english_catalog_tags() -> None:
    score = retrieval._graph_score(
        "公路旅行摇滚",
        _row("track", "alternative rock road trip"),
        _plan(),
    )

    assert score == 1.0


def test_packaged_svg_cover_is_resolved_as_safe_data_url(monkeypatch, tmp_path) -> None:
    catalog = tmp_path / "catalog.jsonl"
    catalog.write_text("", encoding="utf-8")
    cover = tmp_path / "covers" / "track.svg"
    cover.parent.mkdir()
    cover.write_text("<svg/>", encoding="utf-8")
    monkeypatch.setenv("SOULTUNER_CATALOG_PATH", str(catalog))
    retrieval._cover_data_url.cache_clear()

    resolved = retrieval.resolve_cover_source({"cover_fallback_path": "covers/track.svg"})

    assert resolved == "data:image/svg+xml;base64," + base64.b64encode(b"<svg/>").decode()
    assert retrieval.resolve_cover_source({"cover_fallback_path": "../outside.svg"}) == ""
