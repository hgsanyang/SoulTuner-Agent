"""Lazy M2D-CLAP text encoder for querying Aura's real audio vectors."""

from __future__ import annotations

import os
import sys
import threading
from functools import lru_cache
from pathlib import Path


_LOCK = threading.Lock()
_EXPECTED_DIM = 768
_BERT_FILES = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
)


def _source_root() -> Path:
    return Path(os.getenv("SOULTUNER_SOURCE_CACHE", "/mnt/workspace/soultuner/source")).expanduser()


def _checkpoint() -> Path:
    configured = os.getenv("M2D_CLAP_CHECKPOINT", "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path(
        "/mnt/workspace/soultuner/model_cache/m2d_clap/m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025/checkpoint-30.pth"
    )


def _bert_snapshot() -> Path:
    configured = os.getenv("SOULTUNER_BERT_BASE_UNCASED_PATH", "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path("/mnt/workspace/soultuner/model_cache/bert-base-uncased")


def _prepare_import() -> None:
    source = _source_root()
    checkpoint = _checkpoint()
    bert_snapshot = _bert_snapshot()
    if not (source / "retrieval" / "audio_embedder.py").is_file():
        raise FileNotFoundError(f"SoulTuner source cache is not ready: {source}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"M2D-CLAP checkpoint is not ready: {checkpoint}")
    missing_bert = [name for name in _BERT_FILES if not (bert_snapshot / name).is_file()]
    if missing_bert:
        raise FileNotFoundError("M2D text encoder cache is not ready: " + ", ".join(missing_bert))
    os.environ.setdefault("M2D_CLAP_WEIGHT_DIR", str(checkpoint.parent))
    os.environ.setdefault("M2D_CLAP_CHECKPOINT", str(checkpoint))
    # ``portable_m2d`` understands this deterministic environment name and
    # therefore never attempts to reach the blocked Hugging Face endpoint.
    os.environ["GOOGLE_BERT_BERT_BASE_UNCASED_PATH"] = str(bert_snapshot)
    os.environ["HF_HUB_OFFLINE"] = "1"
    value = str(source)
    if value not in sys.path:
        sys.path.insert(0, value)


@lru_cache(maxsize=32)
def encode_text_query(text: str) -> list[float] | None:
    """Encode one acoustic description in the same 768d space as Aura songs.

    Any cold-start/import failure is deliberately non-fatal: retrieval falls
    back to reviewed catalogue descriptions while exposing the backend name.
    """

    clean = str(text or "").strip()
    if not clean:
        return None
    try:
        with _LOCK:
            _prepare_import()
            from retrieval.audio_embedder import encode_text_to_embedding

            vector = [float(value) for value in encode_text_to_embedding(clean)]
        if len(vector) != _EXPECTED_DIM:
            raise ValueError(f"M2D query vector has {len(vector)} dimensions")
        return vector
    except Exception as exc:
        print(
            f"SoulTuner dense query fallback: {type(exc).__name__}: {exc}",
            flush=True,
        )
        return None
