"""Optional CLaMP3 embedding bridge for offline alignment bake-offs.

CLaMP3 is intentionally kept out of the online retrieval path for now.  The
official project extracts global vectors through its command-line scripts after
audio preprocessing, so this module delegates to a locally checked-out CLaMP3
repo when ``CLAMP3_REPO_DIR`` is configured.
"""

from __future__ import annotations

import os
import ntpath
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

CLAMP3_REPO_URL = "https://github.com/sanderwood/clamp3"
CLAMP3_EMBEDDING_DIM = 768
CLAMP3_REPO_ENV = "CLAMP3_REPO_DIR"
CLAMP3_PYTHON_ENV = "CLAMP3_PYTHON"


class Clamp3UnavailableError(RuntimeError):
    """Raised when the optional CLaMP3 repo/runtime is not configured."""


def clamp3_repo_dir() -> Path:
    raw = os.getenv(CLAMP3_REPO_ENV, "").strip()
    if not raw:
        raise Clamp3UnavailableError(
            f"{CLAMP3_REPO_ENV} is not set. Clone {CLAMP3_REPO_URL} and point "
            "this variable at the repo root to run the CLaMP3 bake-off."
        )
    repo = Path(raw).expanduser()
    if not (repo / "clamp3_embd.py").is_file():
        raise Clamp3UnavailableError(f"{CLAMP3_REPO_ENV} does not contain clamp3_embd.py: {repo}")
    return repo


def is_clamp3_available() -> bool:
    try:
        clamp3_repo_dir()
        return True
    except Clamp3UnavailableError:
        return False


def build_clamp3_embedding_command(repo_dir: Path, input_dir: Path, output_dir: Path) -> list[str]:
    python_executable = os.getenv(CLAMP3_PYTHON_ENV, "").strip() or os.getenv("PYTHON", "python")
    return [
        python_executable,
        str(repo_dir / "clamp3_embd.py"),
        str(input_dir),
        str(output_dir),
        "--get_global",
    ]


def _clamp3_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    python_executable = os.getenv(CLAMP3_PYTHON_ENV, "").strip()
    if python_executable:
        python_dir = _python_executable_dir(python_executable)
        env["PYTHON"] = python_executable
        env["PATH"] = python_dir + os.pathsep + env.get("PATH", "")
    return env


def _python_executable_dir(python_executable: str) -> str:
    expanded = os.path.expanduser(str(python_executable).strip())
    if "\\" in expanded or ntpath.splitdrive(expanded)[0]:
        return ntpath.dirname(expanded) or "."
    return str(Path(expanded).parent)


def _run_clamp3(input_dir: Path, output_dir: Path) -> None:
    repo = clamp3_repo_dir()
    command = build_clamp3_embedding_command(repo, input_dir, output_dir)
    subprocess.run(command, cwd=repo, env=_clamp3_subprocess_env(), check=True)


def _load_first_embedding(output_dir: Path) -> list[float]:
    files = sorted(output_dir.rglob("*.npy"))
    if not files:
        raise Clamp3UnavailableError(f"CLaMP3 did not produce any .npy embedding under {output_dir}")
    return _load_embedding_file(files[0])


def _load_embedding_file(path: Path) -> list[float]:
    arr = np.load(path).astype(np.float32)
    vector = arr.reshape(-1) if arr.ndim == 1 else arr.reshape(-1, arr.shape[-1]).mean(axis=0)
    if vector.shape[0] != CLAMP3_EMBEDDING_DIM:
        raise ValueError(f"Expected {CLAMP3_EMBEDDING_DIM}d CLaMP3 vector, got shape={arr.shape}")
    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector = vector / norm
    return vector.astype(np.float32).tolist()


def encode_text_to_clamp3(text: str) -> list[float]:
    """Encode a text query through the official CLaMP3 feature extractor."""
    with tempfile.TemporaryDirectory(prefix="soultuner_clamp3_text_") as tmp:
        root = Path(tmp)
        input_dir = root / "input"
        output_dir = root / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        (input_dir / "query.txt").write_text(str(text or "").strip(), encoding="utf-8")
        _run_clamp3(input_dir, output_dir)
        return _load_first_embedding(output_dir)


def encode_audio_file_to_clamp3(audio_path: str | Path) -> list[float]:
    """Encode one audio file through the official CLaMP3 feature extractor."""
    source = Path(audio_path)
    if not source.is_file():
        raise FileNotFoundError(str(source))
    with tempfile.TemporaryDirectory(prefix="soultuner_clamp3_audio_") as tmp:
        root = Path(tmp)
        input_dir = root / "input"
        output_dir = root / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        target = input_dir / source.name
        shutil.copy2(source, target)
        _run_clamp3(input_dir, output_dir)
        return _load_first_embedding(output_dir)


def encode_texts_to_clamp3(texts: list[str]) -> list[list[float]]:
    """Encode multiple text queries in one CLaMP3 subprocess."""
    if not texts:
        return []

    with tempfile.TemporaryDirectory(prefix="soultuner_clamp3_text_batch_") as tmp:
        root = Path(tmp)
        input_dir = root / "input"
        output_dir = root / "output"
        input_dir.mkdir(parents=True, exist_ok=True)

        expected: list[str] = []
        for index, text in enumerate(texts):
            name = f"{index:04d}.txt"
            (input_dir / name).write_text(str(text or "").strip(), encoding="utf-8")
            expected.append(Path(name).with_suffix(".npy").name)

        _run_clamp3(input_dir, output_dir)

        output_by_name = {path.name: path for path in output_dir.rglob("*.npy")}
        vectors: list[list[float]] = []
        for output_name in expected:
            output_file = output_by_name.get(output_name)
            if output_file is None:
                raise Clamp3UnavailableError(f"CLaMP3 did not produce {output_name}")
            vectors.append(_load_embedding_file(output_file))
        return vectors


def encode_audio_files_to_clamp3(audio_paths: list[str | Path], strict: bool = True) -> dict[str, list[float]]:
    """Encode multiple audio files in one CLaMP3 subprocess.

    The official CLaMP3 script supports directory inputs.  Running it once per
    song repeatedly reloads MERT and CLaMP3, which makes offline bake-offs much
    slower than necessary.  This helper prefixes copied filenames with a stable
    index so output ``.npy`` files can be mapped back to the original path.
    """
    sources = [Path(path) for path in audio_paths]
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(str(source))
    if not sources:
        return {}

    with tempfile.TemporaryDirectory(prefix="soultuner_clamp3_audio_batch_") as tmp:
        root = Path(tmp)
        input_dir = root / "input"
        output_dir = root / "output"
        input_dir.mkdir(parents=True, exist_ok=True)

        expected: dict[str, Path] = {}
        for index, source in enumerate(sources):
            safe_name = f"{index:04d}_{source.name}"
            target = input_dir / safe_name
            shutil.copy2(source, target)
            expected[target.with_suffix(".npy").name] = source

        _run_clamp3(input_dir, output_dir)

        vectors: dict[str, list[float]] = {}
        output_by_name = {path.name: path for path in output_dir.rglob("*.npy")}
        for output_name, source in expected.items():
            output_file = output_by_name.get(output_name)
            if output_file is None:
                if not strict:
                    continue
                raise Clamp3UnavailableError(f"CLaMP3 did not produce {output_name}")
            vectors[str(source)] = _load_embedding_file(output_file)
        return vectors
