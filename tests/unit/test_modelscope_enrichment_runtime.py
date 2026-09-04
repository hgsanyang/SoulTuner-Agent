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


def test_default_enrichment_requires_muq_and_omar_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SOULTUNER_EMBEDDING_FAMILIES", raising=False)
    monkeypatch.delenv("SOULTUNER_BACKFILL_M2D_FALLBACK", raising=False)

    assert enrichment_runtime._required_families() == ("muq_embedding", "omar_embedding")


def test_m2d_backfill_is_explicitly_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SOULTUNER_EMBEDDING_FAMILIES", raising=False)
    monkeypatch.setenv("SOULTUNER_BACKFILL_M2D_FALLBACK", "1")

    assert enrichment_runtime._required_families() == (
        "muq_embedding",
        "omar_embedding",
        "m2d2_embedding",
    )


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


def test_bert_snapshot_uses_modelscope_and_persistent_workspace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("SOULTUNER_WORKSPACE_ROOT", str(tmp_path))
    captured: list[list[str]] = []

    def fake_run(command, **_kwargs):
        captured.append(command)
        target = Path(command[command.index("--local-dir") + 1])
        target.mkdir(parents=True, exist_ok=True)
        for name in enrichment_runtime.BERT_FILES:
            (target / name).write_bytes(b"model")

    monkeypatch.setattr(enrichment_runtime.subprocess, "run", fake_run)

    snapshot = enrichment_runtime._ensure_bert_snapshot()

    assert snapshot == tmp_path / "model_cache" / "bert-base-uncased"
    assert captured[0][:3] == [
        "modelscope",
        "download",
        enrichment_runtime.BERT_REPO_ID,
    ]
    assert all((snapshot / name).is_file() for name in enrichment_runtime.BERT_FILES)
