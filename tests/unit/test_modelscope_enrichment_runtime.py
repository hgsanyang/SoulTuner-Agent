from __future__ import annotations

import json
from pathlib import Path

import pytest

from deploy.modelscope_space import enrichment_runtime


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
    assert driver.kwargs["source_id"] == "226"
    assert driver.kwargs["database_"] == "soultuner"
    with pytest.raises(ValueError, match="dimension mismatch"):
        enrichment_runtime._update_vector(driver, "sdd-226", "muq_embedding", [0.1])
