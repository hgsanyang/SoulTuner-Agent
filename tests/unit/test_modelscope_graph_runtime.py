from __future__ import annotations

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

    merged = graph_runtime.merge_graph_overlay(
        [{"song_id": "sdd-1", "title": "Local", "audio_relpath": "00/1.mp3"}]
    )

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
