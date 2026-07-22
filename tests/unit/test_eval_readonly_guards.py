from pathlib import Path


def test_runtime_loggers_are_disabled_in_eval_mode(monkeypatch, tmp_path: Path):
    from config.settings import settings
    from services.feedback_logger import log_exposure
    from services.llm_feedback_logger import feedback_log_enabled
    from services.teacher_log import teacher_log_enabled

    monkeypatch.setattr(settings, "eval_disable_side_effects", True)
    monkeypatch.setenv("MUSIC_FEEDBACK_DIR", str(tmp_path / "feedback"))
    monkeypatch.setenv("TEACHER_LOG", "1")

    assert teacher_log_enabled() is False
    assert feedback_log_enabled() is False
    log_exposure(query="quiet", recommendations=[{"title": "A"}], user_id="eval")
    assert not (tmp_path / "feedback" / "exposures.jsonl").exists()


def test_runtime_fingerprint_ignores_vector_properties():
    from tests.eval.runtime_fingerprint import neo4j_fingerprint

    class FakeClient:
        def execute_query(self, query):
            if "UNWIND keys" in query:
                return [
                    {"id": "1", "key": "title", "value": "Song"},
                    {"id": "1", "key": "muq_embedding", "value": [0.1, 0.2]},
                ]
            if "MATCH (a)-[r]" in query:
                return []
            return [{"id": "1", "labels": ["Song"]}]

    fingerprint = neo4j_fingerprint(FakeClient())
    assert fingerprint["node_properties"] == 1


def test_runtime_fingerprint_detects_changes():
    from tests.eval.runtime_fingerprint import fingerprint_changes

    assert fingerprint_changes({"neo4j": {"digest": "a"}}, {"neo4j": {"digest": "a"}}) == {}
    assert "neo4j" in fingerprint_changes(
        {"neo4j": {"digest": "a"}},
        {"neo4j": {"digest": "b"}},
    )
