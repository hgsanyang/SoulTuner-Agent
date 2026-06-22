import os

from config.settings import load_project_dashscope_key


def test_project_dashscope_key_overrides_inherited_value(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("DASHSCOPE_API_KEY=project-test-key\n", encoding="utf-8")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "inherited-test-key")

    assert load_project_dashscope_key(env_file)
    assert os.environ["DASHSCOPE_API_KEY"] == "project-test-key"


def test_missing_project_env_does_not_change_process(tmp_path, monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "inherited-test-key")
    assert not load_project_dashscope_key(tmp_path / "missing.env")
    assert os.environ["DASHSCOPE_API_KEY"] == "inherited-test-key"
