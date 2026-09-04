"""Lazy MuQ-first text encoder for the public Space retrieval path.

Chinese queries are sent to MuQ unchanged. M2D-CLAP is retained for explicit
CPU compatibility profiles; the normal GPU deployment does not load it.
"""

from __future__ import annotations

import os
import sys
import threading
from functools import lru_cache
from pathlib import Path


_LOCK = threading.Lock()
_WARMUP_LOCK = threading.Lock()
_WARMUP_THREAD: threading.Thread | None = None
_WARMUP_STATE = "not-started"
_WARMUP_BACKEND = ""
_EXPECTED_DIMS = {"muq": 512, "m2d": 768}
_BERT_FILES = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
)


def _workspace_root() -> Path:
    configured = os.getenv("SOULTUNER_WORKSPACE_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return Path("/mnt/workspace/soultuner")


def _source_root() -> Path:
    return Path(os.getenv("SOULTUNER_SOURCE_CACHE", str(_workspace_root() / "source"))).expanduser()


def _checkpoint() -> Path:
    configured = os.getenv("M2D_CLAP_CHECKPOINT", "").strip()
    if configured:
        return Path(configured).expanduser()
    return _workspace_root() / (
        "model_cache/m2d_clap/"
        "m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025/checkpoint-30.pth"
    )


def _bert_snapshot() -> Path:
    configured = os.getenv("SOULTUNER_BERT_BASE_UNCASED_PATH", "").strip()
    if configured:
        return Path(configured).expanduser()
    return _workspace_root() / "model_cache/bert-base-uncased"


def _prepare_source_import(module_name: str) -> Path:
    source = _source_root()
    if not (source / "retrieval" / module_name).is_file():
        raise FileNotFoundError(f"SoulTuner source cache is not ready: {source}")
    value = str(source)
    if value not in sys.path:
        sys.path.insert(0, value)
    return source


def _prepare_muq_import() -> None:
    _prepare_source_import("muq_embedder.py")
    cache = _workspace_root() / "hf_cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache / "hub"))
    model_cache = cache / "hub" / "models--OpenMuQ--MuQ-MuLan-large" / "snapshots"
    cache_ready = model_cache.is_dir() and any(path.is_dir() for path in model_cache.iterdir())
    policy = os.getenv("SOULTUNER_DENSE_LOCAL_FILES_ONLY", "auto").strip().lower()
    local_only = cache_ready if policy == "auto" else policy not in {"0", "false", "no", "off"}
    # A fresh persistent disk may have complete Aura vectors but not the MuQ
    # text tower. Allow its background warmup to materialise the model once;
    # later starts are strictly offline and reuse the cached snapshot.
    os.environ["MUQ_MULAN_LOCAL_FILES_ONLY"] = "1" if local_only else "0"
    os.environ["HF_HUB_OFFLINE"] = "1" if local_only else "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "1" if local_only else "0"
    if not local_only:
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def _prepare_m2d_import() -> None:
    _prepare_source_import("audio_embedder.py")
    checkpoint = _checkpoint()
    bert_snapshot = _bert_snapshot()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"M2D-CLAP checkpoint is not ready: {checkpoint}")
    missing_bert = [name for name in _BERT_FILES if not (bert_snapshot / name).is_file()]
    if missing_bert:
        raise FileNotFoundError("M2D text encoder cache is not ready: " + ", ".join(missing_bert))
    os.environ.setdefault("M2D_CLAP_WEIGHT_DIR", str(checkpoint.parent))
    os.environ.setdefault("M2D_CLAP_CHECKPOINT", str(checkpoint))
    os.environ["GOOGLE_BERT_BERT_BASE_UNCASED_PATH"] = str(bert_snapshot)
    os.environ["HF_HUB_OFFLINE"] = "1"


def _primary_backend() -> str:
    configured = os.getenv("SOULTUNER_DENSE_PRIMARY_BACKEND", "muq").strip().lower()
    return configured if configured in _EXPECTED_DIMS else "muq"


def _backend_order() -> tuple[str, ...]:
    primary = _primary_backend()
    if primary == "muq" and os.getenv("SOULTUNER_ENABLE_M2D_FALLBACK", "0").strip() == "1":
        return ("muq", "m2d")
    return (primary,)


@lru_cache(maxsize=64)
def encode_text_query(text: str, backend: str = "muq") -> list[float] | None:
    """Encode one query using a named backend without translating its text."""

    clean = str(text or "").strip()
    selected = str(backend or "muq").strip().lower()
    if not clean or selected not in _EXPECTED_DIMS:
        return None
    try:
        with _LOCK:
            if selected == "muq":
                _prepare_muq_import()
                from retrieval.muq_embedder import encode_text_to_muq

                vector = [float(value) for value in encode_text_to_muq(clean)]
            else:
                _prepare_m2d_import()
                from retrieval.audio_embedder import encode_text_to_embedding

                vector = [float(value) for value in encode_text_to_embedding(clean)]
        if len(vector) != _EXPECTED_DIMS[selected]:
            raise ValueError(f"{selected} query vector has {len(vector)} dimensions")
        return vector
    except Exception as exc:
        print(
            f"SoulTuner {selected} dense query fallback: {type(exc).__name__}: {exc}",
            flush=True,
        )
        return None


def encode_primary_text_query(text: str, *, fallback_text: str | None = None) -> tuple[list[float] | None, str]:
    """Use Chinese-native MuQ; an explicit CPU compatibility profile may use M2D."""

    for backend in _backend_order():
        backend_text = text if backend == "muq" else (fallback_text or text)
        vector = encode_text_query(backend_text, backend)
        if vector:
            return vector, backend
    return None, "catalog"


def dense_warmup_status() -> str:
    with _WARMUP_LOCK:
        return _WARMUP_STATE


def dense_warmup_backend() -> str:
    with _WARMUP_LOCK:
        return _WARMUP_BACKEND


def launch_dense_text_warmup() -> str:
    """Warm the MuQ text tower once after persisted assets are available."""

    global _WARMUP_STATE, _WARMUP_THREAD
    if os.getenv("SOULTUNER_DENSE_TEXT_WARMUP", "1").strip() != "1":
        with _WARMUP_LOCK:
            _WARMUP_STATE = "disabled"
        return _WARMUP_STATE
    with _WARMUP_LOCK:
        if _WARMUP_STATE in {"starting", "ready"}:
            return _WARMUP_STATE
        _WARMUP_STATE = "starting"

    def run() -> None:
        global _WARMUP_BACKEND, _WARMUP_STATE
        vector, backend = encode_primary_text_query("适合安静放松、温暖而有空间感的音乐")
        with _WARMUP_LOCK:
            _WARMUP_BACKEND = backend if vector else ""
            expected = _EXPECTED_DIMS.get(backend, 0)
            _WARMUP_STATE = "ready" if vector and len(vector) == expected else "failed"
        print(f"SoulTuner dense text warmup: {_WARMUP_STATE} ({backend})", flush=True)

    thread = threading.Thread(target=run, name="soultuner-dense-warmup", daemon=True)
    with _WARMUP_LOCK:
        _WARMUP_THREAD = thread
    thread.start()
    return "starting"
