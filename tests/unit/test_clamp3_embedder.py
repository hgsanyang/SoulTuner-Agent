import numpy as np

from retrieval.clamp3_embedder import (
    CLAMP3_EMBEDDING_DIM,
    Clamp3UnavailableError,
    _load_first_embedding,
    build_clamp3_embedding_command,
    clamp3_repo_dir,
    is_clamp3_available,
)


def test_clamp3_repo_dir_requires_env(monkeypatch):
    monkeypatch.delenv("CLAMP3_REPO_DIR", raising=False)

    assert not is_clamp3_available()
    try:
        clamp3_repo_dir()
    except Clamp3UnavailableError as exc:
        assert "CLAMP3_REPO_DIR" in str(exc)
    else:
        raise AssertionError("missing CLaMP3 repo should fail")


def test_build_clamp3_embedding_command_uses_global_vectors(tmp_path, monkeypatch):
    repo = tmp_path / "clamp3"
    repo.mkdir()
    monkeypatch.setenv("PYTHON", "python-test")

    command = build_clamp3_embedding_command(repo, tmp_path / "in", tmp_path / "out")

    assert command == [
        "python-test",
        str(repo / "clamp3_embd.py"),
        str(tmp_path / "in"),
        str(tmp_path / "out"),
        "--get_global",
    ]


def test_load_first_embedding_normalizes_768d_vector(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    np.save(out / "x.npy", np.ones((1, CLAMP3_EMBEDDING_DIM), dtype=np.float32))

    vector = _load_first_embedding(out)

    assert len(vector) == CLAMP3_EMBEDDING_DIM
    assert abs(sum(x * x for x in vector) - 1.0) < 1e-5


def test_load_first_embedding_rejects_wrong_dimension(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    np.save(out / "x.npy", np.ones((1, 12), dtype=np.float32))

    try:
        _load_first_embedding(out)
    except ValueError as exc:
        assert "768d" in str(exc)
    else:
        raise AssertionError("wrong CLaMP3 dimension should fail")
