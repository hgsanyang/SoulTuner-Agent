from __future__ import annotations

import sys
from types import SimpleNamespace

from deploy.modelscope_space import graph_runtime


def test_merge_graph_overlay_prefers_aura_metadata_without_losing_audio(monkeypatch) -> None:
    monkeypatch.setattr(
        graph_runtime,
        "graph_overlay",
        lambda: (
            {
                "sdd-1": {
                    "song_id": "sdd-1",
                    "title": "Graph title",
                    "artist": "Graph artist",
                    "cover_url": "https://usercontent.jamendo.com/cover.jpg",
                    "genres": ["ambient"],
                    "moods_themes": ["relaxing"],
                    "instruments": ["piano"],
                    "enrichment_status": "pending",
                }
            },
            {"state": "ready", "tracks": 1, "enriched": 0},
        ),
    )

    merged = graph_runtime.merge_graph_overlay([{"song_id": "sdd-1", "title": "Local", "audio_relpath": "00/1.mp3"}])

    assert merged[0]["title"] == "Graph title"
    assert merged[0]["audio_relpath"] == "00/1.mp3"
    assert merged[0]["graph_backend"] == "neo4j_aura"
    assert merged[0]["genres"] == ["ambient"]


def test_unavailable_aura_falls_back_to_local_catalog(monkeypatch) -> None:
    monkeypatch.setattr(
        graph_runtime,
        "graph_overlay",
        lambda: ({}, {"state": "unavailable", "tracks": 0}),
    )

    merged = graph_runtime.merge_graph_overlay([{"song_id": "sdd-1", "title": "Local"}])

    assert merged[0]["title"] == "Local"
    assert merged[0]["graph_backend"] == "local_catalog"


def test_vector_query_reconnects_and_returns_aura_scores(monkeypatch) -> None:
    captured = {}

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def run(self, query, **params):
            captured["query"] = query
            captured["params"] = params
            return [
                {"song_id": "sdd-1", "score": 0.91},
                {"song_id": "sdd-2", "score": 0.82},
            ]

    class Driver:
        def session(self, **_kwargs):
            return Session()

        def close(self):
            captured["closed"] = True

    database = SimpleNamespace(
        driver=lambda *_args, **_kwargs: Driver(),
    )
    monkeypatch.setitem(
        sys.modules,
        "neo4j",
        SimpleNamespace(GraphDatabase=database),
    )
    monkeypatch.setattr(
        graph_runtime,
        "_connection_settings",
        lambda: ("neo4j+s://example", "user", "secret", "neo4j"),
    )

    scores, status = graph_runtime.vector_query_scores([0.1, 0.2], limit=2)

    assert scores == {"sdd-1": 0.91, "sdd-2": 0.82}
    assert status == {"state": "ready", "matches": 2, "index": "song_m2d2_index"}
    assert captured["params"]["limit"] == 2
    assert "db.index.vector.queryNodes" in captured["query"]
    assert captured["params"]["datasets"] == [
        "song_describer_full",
        "fma_small_balanced",
    ]
    assert captured["closed"] is True
