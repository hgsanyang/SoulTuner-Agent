"""Opt-in tests against ONLY the dedicated loopback audit container."""
import os
import uuid

import pytest

from retrieval.neo4j_client import Neo4jClient, Neo4jQueryError

pytestmark = pytest.mark.skipif(os.getenv("SOULTUNER_AUDIT_NEO4J") != "1", reason="dedicated audit Neo4j not requested")


def test_real_memory_receipts_do_not_reapply_older_preferences(monkeypatch):
    from retrieval.user_memory import UserMemoryManager
    monkeypatch.setenv("NEO4J_URI", "bolt://127.0.0.1:27687")
    monkeypatch.setenv("NEO4J_USER", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "audit-unused")
    monkeypatch.setenv("NEO4J_DATABASE", "neo4j")
    monkeypatch.setattr(Neo4jClient, "_instance", None)
    client = Neo4jClient()
    owner = "audit-memory-" + uuid.uuid4().hex
    manager = UserMemoryManager(client)
    try:
        assert manager.update_semantic_preferences(owner, {"add_moods": ["Warm"]}, operation_id="first")
        assert manager.update_semantic_preferences(owner, {"avoid_moods": ["Warm"]}, operation_id="second")
        assert manager.update_semantic_preferences(owner, {"add_moods": ["Warm"]}, operation_id="first")
        row = client.execute_read_query(
            "MATCH (u:User {id:$owner}) RETURN u.add_moods AS liked,u.avoid_moods AS avoided", {"owner": owner},
        )[0]
        assert row == {"liked": [], "avoided": ["Warm"]}
        record = {"memory_key": "test-warm", "field": "add_moods", "value": "Warm", "confidence": 0.8,
                  "created_at": 1000, "expires_at": 9999999999999, "ledger_seq": 1, "ledger_record_id": "l2-first"}
        assert manager.upsert_inferred_preference(owner, record)
        newer = {**record, "ledger_seq": 2, "ledger_record_id": "l2-second", "confidence": 0.9}
        assert manager.upsert_inferred_preference(owner, newer)
        assert manager.upsert_inferred_preference(owner, record)
        row = client.execute_read_query(
            "MATCH (p:InferredPreference {user_id:$owner}) RETURN p.ledger_record_id AS id,p.confidence AS confidence",
            {"owner": owner},
        )[0]
        assert row == {"id": "l2-second", "confidence": 0.9}
        assert manager.delete_inferred_preference(owner, memory_key="test-warm", ledger_record_id="l2-second")
        assert manager.upsert_inferred_preference(owner, newer)
        assert client.execute_read_query(
            "MATCH (p:InferredPreference {user_id:$owner}) RETURN p.status AS status", {"owner": owner},
        ) == [{"status": "deleted"}]
    finally:
        client.execute_write_query("MATCH (p:InferredPreference {user_id:$owner}) DETACH DELETE p", {"owner": owner})
        client.execute_write_query("MATCH (r:MemoryWriteReceipt {user_id:$owner}) DELETE r", {"owner": owner})
        client.execute_write_query("MATCH (u:User {id:$owner}) DETACH DELETE u", {"owner": owner})
        client.close()


def test_real_database_failure_empty_identity_and_reconnect(monkeypatch):
    monkeypatch.setenv("NEO4J_URI", "bolt://127.0.0.1:27687")
    monkeypatch.setenv("NEO4J_USER", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "audit-unused")
    monkeypatch.setenv("NEO4J_DATABASE", "neo4j")
    monkeypatch.setattr(Neo4jClient, "_instance", None)
    client = Neo4jClient()
    run = uuid.uuid4().hex
    try:
        assert client.execute_read_query("MATCH (s:AuditSong {run:$run}) RETURN s", {"run": run}) == []
        with pytest.raises(Neo4jQueryError):
            client.execute_read_query("INVALID AUDIT QUERY")
        for key, artist in (("a", "A"), ("b", "B")):
            client.execute_write_query(
                "MERGE (s:AuditSong {run:$run, music_id:$key}) SET s.title='Same', s.artist=$artist",
                {"run": run, "key": key, "artist": artist},
            )
        rows = client.execute_read_query("MATCH (s:AuditSong {run:$run}) RETURN s.music_id AS id ORDER BY id", {"run": run})
        assert rows == [{"id": "a"}, {"id": "b"}]
        from retrieval.candidate_identity import candidate_identity, IDENTITY_MATCH
        for key in ("a", "b"):
            client.execute_write_query(
                "CREATE (:Song {audit_run:$run, music_id:$id, title:'Ambiguous', artist:'Same Artist', "
                "audio_url:'/static/audio/audit.mp3'})",
                {"run": run, "id": run + key},
            )
        identities = [candidate_identity({"music_id": run + "a"}),
                      candidate_identity({"title": "Ambiguous", "artist": "Same Artist"})]
        rows = client.execute_read_query(IDENTITY_MATCH + "RETURN identity.key AS key", {"identities": identities})
        assert rows == [{"key": identities[0]["key"]}]
        import json
        from retrieval import recall_sources
        monkeypatch.setattr(recall_sources, "get_neo4j_client", lambda: client)
        recalled = json.loads(recall_sources.graph_candidate_recall(
            {"song_entities": ["Ambiguous"]}, {}, limit=10))
        assert {row["music_id"] for row in recalled} == {run + "a", run + "b"}
        client.close()
        assert client.execute_read_query("RETURN 1 AS ok") == [{"ok": 1}]
    finally:
        client.execute_write_query("MATCH (s:Song {audit_run:$run}) DETACH DELETE s", {"run": run})
        client.execute_write_query("MATCH (s:AuditSong {run:$run}) DETACH DELETE s", {"run": run})
        client.close()
