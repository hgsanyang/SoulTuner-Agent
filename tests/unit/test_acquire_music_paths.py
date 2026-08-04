import asyncio
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

from tools.acquire_music import (
    EmbeddingExtraction,
    EnrichmentIncompleteError,
    _background_flywheel,
    _quick_ingest_to_neo4j,
    _resolve_enrichment_paths,
    _song_identity_parameters,
    _sync_extract_embeddings,
)


def test_enrichment_resolves_durable_library_static_urls(tmp_path, monkeypatch):
    monkeypatch.setenv("MUSIC_DATA_ROOT", str(tmp_path))
    song = {
        "audio_url": "/static/audio/Track - Artist.flac",
        "lrc_url": "/static/lyrics/Track - Artist.lrc",
    }

    audio, lyric = _resolve_enrichment_paths(song, "Track - Artist", "flac")

    assert Path(audio) == tmp_path / "processed_audio" / "audio" / "Track - Artist.flac"
    assert Path(lyric) == tmp_path / "processed_audio" / "lyrics" / "Track - Artist.lrc"


def test_enrichment_prefers_existing_explicit_paths(tmp_path):
    audio_path = tmp_path / "custom.flac"
    lyric_path = tmp_path / "custom.lrc"
    audio_path.write_bytes(b"fLaC")
    lyric_path.write_text("lyrics", encoding="utf-8")

    audio, lyric = _resolve_enrichment_paths(
        {"audio_path": str(audio_path), "lrc_path": str(lyric_path)},
        "ignored", "flac",
    )

    assert Path(audio) == audio_path
    assert Path(lyric) == lyric_path


def test_quick_ingest_preserves_cache_source_and_run_id(monkeypatch):
    class FakeClient:
        def __init__(self):
            self.calls = []

        def execute_query(self, query, params):
            self.calls.append((query, params))
            return []

    client = FakeClient()
    monkeypatch.setattr("retrieval.neo4j_client.get_neo4j_client", lambda: client)
    monkeypatch.setattr("tools.acquire_music._promote_external_candidate_feedback", lambda *args, **kwargs: None)

    asyncio.run(_quick_ingest_to_neo4j([{
        "title": "Track",
        "artist": "Artist",
        "song_id": "42",
        "source_id": "42",
        "source": "netease_cache",
        "platform": "netease",
        "album": "Album",
        "duration": 123000,
        "ext": "flac",
        "audio_url": "/static/audio/Track - Artist.flac",
        "audio_retention": "saved",
        "catalog_tier": "library",
        "cache_import_run_id": "run-42",
    }]))

    write_query, params = client.calls[-1]
    assert "s.source = $source" in write_query
    assert "s.cache_import_run_id" in write_query
    assert params["source"] == "netease_cache"
    assert params["cache_import_run_id"] == "run-42"


def test_song_identity_collapses_non_breaking_whitespace():
    identity = _song_identity_parameters({
        "song_id": 42,
        "title": "You've\u00a0Changed   But I'm Still a Child",
        "artist": "  a hidden trace  ",
    })

    assert identity == {
        "music_id": "42",
        "title": "You've Changed But I'm Still a Child",
        "artist_name": "a hidden trace",
    }


def test_embedding_models_fail_independently(monkeypatch):
    fake_librosa = SimpleNamespace(
        get_duration=lambda **_kwargs: 1.0,
        load=lambda *_args, **_kwargs: (np.zeros(160, dtype=np.float32), 16000),
        resample=lambda audio, **_kwargs: audio,
    )
    fake_audio_embedder = SimpleNamespace(
        encode_audio_to_embedding=lambda *_args, **_kwargs: [1.0, 2.0],
        extract_audio_representation=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ModuleNotFoundError("omar_rq")
        ),
    )
    fake_muq_embedder = SimpleNamespace(
        encode_audio_to_muq=lambda *_args, **_kwargs: [3.0, 4.0, 5.0],
    )
    monkeypatch.setitem(sys.modules, "librosa", fake_librosa)
    monkeypatch.setitem(sys.modules, "retrieval.audio_embedder", fake_audio_embedder)
    monkeypatch.setitem(sys.modules, "retrieval.muq_embedder", fake_muq_embedder)

    result = _sync_extract_embeddings("song.flac")

    assert result.vectors["muq_embedding"] == [3.0, 4.0, 5.0]
    assert result.vectors["m2d2_embedding"] == [1.0, 2.0]
    assert result.vectors["omar_embedding"] == []
    assert "ModuleNotFoundError" in result.errors["omar_embedding"]


def test_flywheel_matches_catalog_by_stable_music_id(tmp_path, monkeypatch):
    audio = tmp_path / "track.flac"
    lyric = tmp_path / "track.lrc"
    audio.write_bytes(b"fLaC")
    lyric.write_text("[00:00.00]enough lyrics for a tag extraction", encoding="utf-8")

    class FakeClient:
        def __init__(self):
            self.calls = []

        def execute_query(self, query, params):
            self.calls.append((query, params))
            if "embedding_dimensions" in query:
                raise AssertionError("unexpected query")
            if "size(coalesce(s.m2d2_embedding" in query:
                return [{"eid": "1", "m2d2": 768, "omar": 1024, "muq": 512}]
            return [{"eid": "1"}]

    client = FakeClient()
    monkeypatch.setattr("retrieval.neo4j_client.get_neo4j_client", lambda: client)
    monkeypatch.setattr(
        "tools.acquire_music._extract_lyrics_tags",
        lambda *_args, **_kwargs: asyncio.sleep(0, result={"moods": ["Calm"]}),
    )
    monkeypatch.setattr(
        "tools.acquire_music._extract_embeddings",
        lambda *_args, **_kwargs: asyncio.sleep(0, result=EmbeddingExtraction(vectors={
            "muq_embedding": [0.1] * 512,
            "m2d2_embedding": [0.2] * 768,
            "omar_embedding": [0.3] * 1024,
        })),
    )

    result = asyncio.run(_background_flywheel([{
        "song_id": "42",
        "title": "Track",
        "artist": "Artist",
        "file_basename": "Track - Artist",
        "ext": "flac",
        "audio_path": str(audio),
        "lrc_path": str(lyric),
    }]))

    assert result["failure_count"] == 0
    assert result["songs"][0]["embedding_dimensions"] == {
        "m2d2": 768,
        "omar": 1024,
        "muq": 512,
    }
    assert all(params["music_id"] == "42" for _, params in client.calls)


def test_flywheel_deferred_tagging_skips_llm_but_keeps_embeddings(tmp_path, monkeypatch):
    audio = tmp_path / "track.flac"
    lyric = tmp_path / "track.lrc"
    audio.write_bytes(b"fLaC")
    lyric.write_text("lyrics that would normally trigger the tag model", encoding="utf-8")

    class FakeClient:
        def execute_query(self, query, _params):
            if "size(coalesce(s.m2d2_embedding" in query:
                return [{"eid": "1", "m2d2": 768, "omar": 1024, "muq": 512}]
            return [{"eid": "1"}]

    monkeypatch.setattr("retrieval.neo4j_client.get_neo4j_client", FakeClient)
    monkeypatch.setattr(
        "tools.acquire_music._extract_lyrics_tags",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("LLM must not run")),
    )
    monkeypatch.setattr(
        "tools.acquire_music._extract_embeddings",
        lambda *_args, **_kwargs: asyncio.sleep(0, result=EmbeddingExtraction(vectors={
            "muq_embedding": [0.1] * 512,
            "m2d2_embedding": [0.2] * 768,
            "omar_embedding": [0.3] * 1024,
        })),
    )

    result = asyncio.run(_background_flywheel([{
        "song_id": "42",
        "title": "Track",
        "artist": "Artist",
        "file_basename": "Track - Artist",
        "ext": "flac",
        "audio_path": str(audio),
        "lrc_path": str(lyric),
        "tagging_mode": "deferred",
    }]))

    assert result["failure_count"] == 0
    assert result["songs"][0]["tagged"] is False
    assert result["songs"][0]["tagging_status"] == "deferred"
    assert result["songs"][0]["embedding_dimensions"]["muq"] == 512


def test_flywheel_fails_when_required_retrieval_vector_is_missing(tmp_path, monkeypatch):
    audio = tmp_path / "track.flac"
    audio.write_bytes(b"fLaC")

    class FakeClient:
        def execute_query(self, query, _params):
            if "size(coalesce(s.m2d2_embedding" in query:
                return [{"eid": "1", "m2d2": 0, "omar": 1024, "muq": 0}]
            return [{"eid": "1"}]

    monkeypatch.setattr("retrieval.neo4j_client.get_neo4j_client", FakeClient)
    monkeypatch.setattr(
        "tools.acquire_music._extract_embeddings",
        lambda *_args, **_kwargs: asyncio.sleep(0, result=EmbeddingExtraction(
            vectors={"omar_embedding": [0.3] * 1024},
            errors={"muq_embedding": "RuntimeError: unavailable"},
        )),
    )

    with pytest.raises(EnrichmentIncompleteError, match="missing m2d2_embedding,muq_embedding"):
        asyncio.run(_background_flywheel([{
            "song_id": "42",
            "title": "Track",
            "artist": "Artist",
            "file_basename": "Track - Artist",
            "ext": "flac",
            "audio_path": str(audio),
        }]))
