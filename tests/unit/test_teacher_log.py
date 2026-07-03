import json

from services.teacher_log import log_teacher_example


def test_teacher_log_disabled_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("TEACHER_LOG", raising=False)
    monkeypatch.setenv("TEACHER_LOG_DIR", str(tmp_path))

    assert log_teacher_example("planner", inputs={"query": "secret"}, output={"x": 1}) is None
    assert not list(tmp_path.glob("*.jsonl"))


def test_teacher_log_hashes_text_by_default(tmp_path, monkeypatch):
    monkeypatch.setenv("TEACHER_LOG", "1")
    monkeypatch.delenv("TEACHER_LOG_STORE_TEXT", raising=False)
    monkeypatch.setenv("TEACHER_LOG_DIR", str(tmp_path))

    path = log_teacher_example("planner", inputs={"query": "想听雨天歌"}, output={"plan": "soft"})

    assert path is not None
    row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert row["text_mode"] == "hashed"
    assert row["inputs"]["query"]["chars"] == 5
    assert "想听雨天歌" not in path.read_text(encoding="utf-8")


def test_teacher_log_can_store_full_text_when_explicitly_enabled(tmp_path, monkeypatch):
    monkeypatch.setenv("TEACHER_LOG", "1")
    monkeypatch.setenv("TEACHER_LOG_STORE_TEXT", "1")
    monkeypatch.setenv("TEACHER_LOG_DIR", str(tmp_path))

    path = log_teacher_example("hyde", inputs={"query": "rainy"}, output={"caption": "soft rain"})

    assert path is not None
    row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert row["text_mode"] == "full"
    assert row["inputs"]["query"] == "rainy"
    assert row["output"]["caption"] == "soft rain"
