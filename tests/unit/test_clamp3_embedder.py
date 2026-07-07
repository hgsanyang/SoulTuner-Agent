import numpy as np

from retrieval.clamp3_embedder import (
    CLAMP3_EMBEDDING_DIM,
    CLAMP3_PYTHON_ENV,
    Clamp3UnavailableError,
    _clamp3_subprocess_env,
    _load_first_embedding,
    _python_executable_dir,
    build_clamp3_embedding_command,
    clamp3_repo_dir,
    encode_audio_files_to_clamp3,
    encode_texts_to_clamp3,
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
    monkeypatch.delenv(CLAMP3_PYTHON_ENV, raising=False)

    command = build_clamp3_embedding_command(repo, tmp_path / "in", tmp_path / "out")

    assert command == [
        "python-test",
        str(repo / "clamp3_embd.py"),
        str(tmp_path / "in"),
        str(tmp_path / "out"),
        "--get_global",
    ]


def test_build_clamp3_embedding_command_prefers_clamp3_python(tmp_path, monkeypatch):
    repo = tmp_path / "clamp3"
    repo.mkdir()
    monkeypatch.setenv("PYTHON", "python-test")
    monkeypatch.setenv(CLAMP3_PYTHON_ENV, "clamp3-python")

    command = build_clamp3_embedding_command(repo, tmp_path / "in", tmp_path / "out")

    assert command[0] == "clamp3-python"


def test_clamp3_subprocess_env_prepends_python_dir(monkeypatch):
    monkeypatch.setenv(CLAMP3_PYTHON_ENV, r"C:\envs\clamp3\python.exe")
    monkeypatch.setenv("PATH", r"C:\Windows")

    env = _clamp3_subprocess_env()

    assert env["PYTHON"] == r"C:\envs\clamp3\python.exe"
    assert env["PATH"].startswith(r"C:\envs\clamp3")


def test_python_executable_dir_supports_native_paths(tmp_path):
    python_executable = tmp_path / "env" / "bin" / "python"

    assert _python_executable_dir(str(python_executable)) == str(python_executable.parent)


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


def test_encode_audio_files_to_clamp3_maps_prefixed_outputs(tmp_path, monkeypatch):
    audio_a = tmp_path / "a.mp3"
    audio_b = tmp_path / "b.mp3"
    audio_a.write_bytes(b"a")
    audio_b.write_bytes(b"b")

    def fake_run(input_dir, output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)
        for source in sorted(input_dir.iterdir()):
            np.save(output_dir / source.with_suffix(".npy").name, np.ones((CLAMP3_EMBEDDING_DIM,), dtype=np.float32))

    monkeypatch.setattr("retrieval.clamp3_embedder._run_clamp3", fake_run)

    vectors = encode_audio_files_to_clamp3([audio_a, audio_b])

    assert sorted(vectors) == [str(audio_a), str(audio_b)]
    assert all(len(vector) == CLAMP3_EMBEDDING_DIM for vector in vectors.values())


def test_encode_audio_files_to_clamp3_can_skip_missing_outputs(tmp_path, monkeypatch):
    audio_a = tmp_path / "a.mp3"
    audio_b = tmp_path / "b.mp3"
    audio_a.write_bytes(b"a")
    audio_b.write_bytes(b"b")

    def fake_run(input_dir, output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)
        first = sorted(input_dir.iterdir())[0]
        np.save(output_dir / first.with_suffix(".npy").name, np.ones((CLAMP3_EMBEDDING_DIM,), dtype=np.float32))

    monkeypatch.setattr("retrieval.clamp3_embedder._run_clamp3", fake_run)

    vectors = encode_audio_files_to_clamp3([audio_a, audio_b], strict=False)

    assert sorted(vectors) == [str(audio_a)]


def test_encode_texts_to_clamp3_maps_indexed_outputs(monkeypatch):
    def fake_run(input_dir, output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)
        for source in sorted(input_dir.iterdir()):
            np.save(output_dir / source.with_suffix(".npy").name, np.ones((CLAMP3_EMBEDDING_DIM,), dtype=np.float32))

    monkeypatch.setattr("retrieval.clamp3_embedder._run_clamp3", fake_run)

    vectors = encode_texts_to_clamp3(["quiet piano", "warm folk"])

    assert len(vectors) == 2
    assert all(len(vector) == CLAMP3_EMBEDDING_DIM for vector in vectors)
