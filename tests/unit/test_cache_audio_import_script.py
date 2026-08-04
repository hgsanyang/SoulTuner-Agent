import json
from pathlib import Path

import pytest

from scripts.import_netease_cache_audio import rollback_run


class _FakeNeo4j:
    def __init__(self, interactions: int = 0):
        self.interactions = interactions
        self.calls = []

    def execute_query(self, query, params):
        self.calls.append((query, params))
        if "sum(interactions)" in query:
            return [{"nodes": 1, "interactions": self.interactions}]
        return [{"count": 1}]


def _report(tmp_path: Path, created: Path) -> Path:
    path = tmp_path / "run.json"
    path.write_text(json.dumps({
        "run_id": "run-1",
        "job_id": None,
        "published": [{"created_files": [str(created)]}],
    }), encoding="utf-8")
    return path


def test_rollback_deletes_only_manifested_files_and_stamped_nodes(tmp_path, monkeypatch):
    created = tmp_path / "created.flac"
    created.write_bytes(b"fLaC")
    client = _FakeNeo4j()
    monkeypatch.setattr("retrieval.neo4j_client.get_neo4j_client", lambda: client)

    result = rollback_run(_report(tmp_path, created))

    assert result["removed_files"] == 1
    assert result["graph_deleted"] == 1
    assert not created.exists()
    assert all(params == {"run_id": "run-1"} for _, params in client.calls)


def test_rollback_refuses_after_user_interaction_and_keeps_files(tmp_path, monkeypatch):
    created = tmp_path / "created.flac"
    created.write_bytes(b"fLaC")
    monkeypatch.setattr(
        "retrieval.neo4j_client.get_neo4j_client",
        lambda: _FakeNeo4j(interactions=1),
    )

    with pytest.raises(RuntimeError, match="user interactions"):
        rollback_run(_report(tmp_path, created))

    assert created.exists()


def test_rollback_dry_run_reports_scope_without_mutation(tmp_path, monkeypatch):
    created = tmp_path / "created.flac"
    created.write_bytes(b"fLaC")
    client = _FakeNeo4j()
    monkeypatch.setattr("retrieval.neo4j_client.get_neo4j_client", lambda: client)

    result = rollback_run(_report(tmp_path, created), dry_run=True)

    assert result["dry_run"] is True
    assert result["would_remove_files"] == 1
    assert result["graph_would_delete"] == 1
    assert result["removed_files"] == 0
    assert result["graph_deleted"] == 0
    assert created.exists()
    assert len(client.calls) == 1
