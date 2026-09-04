from __future__ import annotations

import base64
import os
from pathlib import Path

import pytest

from deploy.modelscope_space import dense_runtime
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


def test_retrieval_uses_muq_omar_aura_scores_when_available(monkeypatch) -> None:
    monkeypatch.setattr(
        retrieval,
        "load_catalog",
        lambda: (_row("rock", "alternative rock road trip"), _row("ambient", "ambient")),
    )
    monkeypatch.setattr(retrieval, "resolve_audio_source", lambda _row: None)
    monkeypatch.setattr(
        retrieval,
        "encode_primary_text_query",
        lambda _text, **_kwargs: ([0.1] * 512, "muq"),
    )
    monkeypatch.setattr(
        retrieval,
        "hybrid_vector_query_scores",
        lambda *_args, **_kwargs: ({"rock": 0.94, "ambient": 0.12}, {"state": "ready"}),
    )

    rows = retrieval.retrieve(
        "公路旅行摇滚",
        _plan(),
        {"graph_weight": 0.25, "dense_weight": 0.75},
        top_k=2,
    )

    assert rows[0]["song_id"] == "rock"
    assert rows[0]["dense_source"] == "muq_omar_aura"
    assert rows[0]["dense_score"] == 0.94
    assert rows[0]["graph_score"] == 1.0


def test_chinese_genre_and_scene_aliases_match_english_catalog_tags() -> None:
    score = retrieval._graph_score(
        "公路旅行摇滚",
        _row("track", "alternative rock road trip"),
        _plan(),
    )

    assert score == 1.0


def test_playback_and_cover_io_only_runs_for_ranked_slate(monkeypatch) -> None:
    rows = tuple(_row(f"song-{index}", "ambient") for index in range(100))
    audio_calls: list[str] = []
    cover_calls: list[str] = []
    monkeypatch.setattr(retrieval, "load_catalog", lambda: rows)
    monkeypatch.setattr(
        retrieval,
        "encode_primary_text_query",
        lambda _text, **_kwargs: (None, "catalog"),
    )
    monkeypatch.setattr(
        retrieval,
        "resolve_audio_source",
        lambda row: audio_calls.append(row["song_id"]) or f"/{row['song_id']}.mp3",
    )
    monkeypatch.setattr(
        retrieval,
        "resolve_cover_source",
        lambda row: cover_calls.append(row["song_id"]) or f"https://example/{row['song_id']}.jpg",
    )

    result = retrieval.retrieve(
        "安静氛围",
        _plan(),
        {"graph_weight": 1.0, "dense_weight": 0.0},
        top_k=5,
    )

    assert len(result) == 5
    assert len(audio_calls) == 5
    assert len(cover_calls) == 5
    assert all("_catalog_row" not in row for row in result)


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


def test_m2d_fallback_requires_local_bert_and_forces_offline_loading(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "source"
    audio_embedder = source / "retrieval" / "audio_embedder.py"
    audio_embedder.parent.mkdir(parents=True)
    audio_embedder.write_text("", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint-30.pth"
    checkpoint.write_bytes(b"weights")
    bert = tmp_path / "bert"
    bert.mkdir()
    monkeypatch.setattr(dense_runtime, "_source_root", lambda: source)
    monkeypatch.setattr(dense_runtime, "_checkpoint", lambda: checkpoint)
    monkeypatch.setattr(dense_runtime, "_bert_snapshot", lambda: bert)

    with pytest.raises(FileNotFoundError, match="text encoder cache"):
        dense_runtime._prepare_m2d_import()

    for name in dense_runtime._BERT_FILES:
        (bert / name).write_bytes(b"model")
    dense_runtime._prepare_m2d_import()

    assert Path(os.environ["GOOGLE_BERT_BERT_BASE_UNCASED_PATH"]) == bert
    assert os.environ["HF_HUB_OFFLINE"] == "1"


def test_dense_text_warmup_runs_once_outside_first_request(monkeypatch) -> None:
    calls: list[str] = []

    class ImmediateThread:
        def __init__(self, *, target, **_kwargs):
            self.target = target

        def start(self):
            self.target()

    monkeypatch.setattr(dense_runtime, "_WARMUP_STATE", "not-started")
    monkeypatch.setattr(dense_runtime, "_WARMUP_BACKEND", "")
    monkeypatch.setattr(dense_runtime.threading, "Thread", ImmediateThread)
    monkeypatch.setattr(
        dense_runtime,
        "encode_primary_text_query",
        lambda text: (calls.append(text) or [0.1] * 512, "muq"),
    )

    assert dense_runtime.launch_dense_text_warmup() == "starting"
    assert dense_runtime.dense_warmup_status() == "ready"
    assert dense_runtime.dense_warmup_backend() == "muq"
    assert dense_runtime.launch_dense_text_warmup() == "ready"
    assert calls == ["适合安静放松、温暖而有空间感的音乐"]


def test_dense_runtime_can_opt_into_muq_then_m2d_compatibility(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []
    monkeypatch.delenv("SOULTUNER_DENSE_PRIMARY_BACKEND", raising=False)
    monkeypatch.setenv("SOULTUNER_ENABLE_M2D_FALLBACK", "1")
    monkeypatch.setattr(
        dense_runtime,
        "encode_text_query",
        lambda text, backend: calls.append((text, backend)) or ([0.2] * 768 if backend == "m2d" else None),
    )

    vector, backend = dense_runtime.encode_primary_text_query(
        "梦幻一点的歌曲有没有",
        fallback_text="dreamy ethereal music",
    )

    assert backend == "m2d"
    assert len(vector or []) == 768
    assert calls == [
        ("梦幻一点的歌曲有没有", "muq"),
        ("dreamy ethereal music", "m2d"),
    ]


def test_dense_runtime_gpu_default_never_loads_m2d(monkeypatch) -> None:
    monkeypatch.delenv("SOULTUNER_DENSE_PRIMARY_BACKEND", raising=False)
    monkeypatch.delenv("SOULTUNER_ENABLE_M2D_FALLBACK", raising=False)

    assert dense_runtime._backend_order() == ("muq",)
