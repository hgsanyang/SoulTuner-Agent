"""Optional CLaMP3 embedding bridge for offline alignment bake-offs.

CLaMP3 is intentionally kept out of the online retrieval path for now.  The
official project extracts global vectors through its command-line scripts after
audio preprocessing, so this module delegates to a locally checked-out CLaMP3
repo when ``CLAMP3_REPO_DIR`` is configured.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

CLAMP3_REPO_URL = "https://github.com/sanderwood/clamp3"
CLAMP3_EMBEDDING_DIM = 768
CLAMP3_REPO_ENV = "CLAMP3_REPO_DIR"


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
    return [
        os.getenv("PYTHON", "python"),
        str(repo_dir / "clamp3_embd.py"),
        str(input_dir),
        str(output_dir),
        "--get_global",
    ]


def _run_clamp3(input_dir: Path, output_dir: Path) -> None:
    repo = clamp3_repo_dir()
    command = build_clamp3_embedding_command(repo, input_dir, output_dir)
    subprocess.run(command, cwd=repo, check=True)


def _load_first_embedding(output_dir: Path) -> list[float]:
    files = sorted(output_dir.rglob("*.npy"))
    if not files:
        raise Clamp3UnavailableError(f"CLaMP3 did not produce any .npy embedding under {output_dir}")
    arr = np.load(files[0]).astype(np.float32)
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
