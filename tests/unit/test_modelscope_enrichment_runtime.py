from __future__ import annotations

import json
from pathlib import Path

import pytest

from deploy.modelscope_space import enrichment_runtime


class _CatalogDriver:
    def __init__(self):
        self.calls = []

    def execute_query(self, query, **params):
        self.calls.append((query, params))
        return [], None, None


def test_catalog_upsert_adds_new_public_dataset_rows_for_vector_indexes() -> None:
    driver = _CatalogDriver()

    enrichment_runtime._upsert_catalog_rows(
        driver,
        [
            {
                "song_id": "fma-000002",
                "source_id": "2",
                "dataset": "fma_small_balanced",
                "title": "Open Road",
                "artist": "Open Artist",
                "genres": ["Rock"],
                "audio_relpath": "fma/000/000002.mp3",
                "license": "Attribution",
                "license_url": "https://creativecommons.org/licenses/by/3.0/",
                "source_url": "https://freemusicarchive.org/",
            }
        ],
    )

    assert len(driver.calls) == 1
    query, params = driver.calls[0]
    assert "MERGE (s:Song {music_id: row.song_id})" in query
    assert params["rows"][0]["dataset"] == "fma_small_balanced"
    assert params["rows"][0]["genres"] == ["Rock"]


def test_missing_vector_query_is_scoped_to_current_catalog() -> None:
    class Driver:
        def execute_query(self, query, **params):
            self.query = query
            self.params = params
            return [{"music_id": "fma-000002"}], None, None

    driver = Driver()
    result = enrichment_runtime._missing_ids(
        driver,
        "m2d2_embedding",
        768,
        {"sdd-1", "fma-000002"},
    )

    assert result == {"fma-000002"}
    assert "IN $song_ids" in driver.query
    assert driver.params["song_ids"] == ["fma-000002", "sdd-1"]


def test_status_file_survives_restart_and_renders_progress(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SOULTUNER_WORKSPACE_ROOT", str(tmp_path))
    enrichment_runtime._write_status("running", family="muq_embedding", completed=20, total=706)

    assert enrichment_runtime.enrichment_status()["completed"] == 20
    rendered = enrichment_runtime.status_markdown()
    assert "后台补齐中" in rendered
    assert "20/706" in rendered


def test_launch_waits_for_full_catalog_before_spawning(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SOULTUNER_ENABLE_AUDIO_ENRICHMENT", "1")
    monkeypatch.setenv("SOULTUNER_ENRICHMENT_MIN_TRACKS", "700")

    assert enrichment_runtime.launch_enrichment_if_requested(5) == {
        "state": "waiting-catalog",
        "tracks": 5,
        "minimum": 700,
    }


def test_catalog_rows_reject_path_traversal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    audio_root = tmp_path / "audio"
    audio_root.mkdir()
    outside = tmp_path / "outside.mp3"
    outside.write_bytes(b"audio")
    catalog = tmp_path / "catalog.jsonl"
    catalog.write_text(
        json.dumps({"song_id": "sdd-1", "audio_relpath": "../outside.mp3"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("SOULTUNER_CATALOG_PATH", str(catalog))
    monkeypatch.setenv("SOULTUNER_AUDIO_ROOT", str(audio_root))
    monkeypatch.setenv("SOULTUNER_ENRICHMENT_MIN_TRACKS", "1")

    with pytest.raises(ValueError):
        enrichment_runtime._catalog_rows()


def test_update_vector_uses_aura_database_and_exact_dimension(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeDriver:
        def __init__(self) -> None:
            self.kwargs = {}

        def execute_query(self, _query: str, **kwargs):
            self.kwargs = kwargs
            return ([{"eid": "1"}], None, None)

    monkeypatch.setenv("NEO4J_DATABASE", "soultuner")
    driver = FakeDriver()
    enrichment_runtime._update_vector(driver, "sdd-226", "muq_embedding", [0.1] * 512)

    assert driver.kwargs["song_id"] == "sdd-226"
    assert "source_id" not in driver.kwargs
    assert driver.kwargs["database_"] == "soultuner"
    with pytest.raises(ValueError, match="dimension mismatch"):
        enrichment_runtime._update_vector(driver, "sdd-226", "muq_embedding", [0.1])
