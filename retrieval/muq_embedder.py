"""MuQ-MuLan text/music embedding helpers.

MuQ-MuLan is used as the music-specific text-to-audio anchor.  The upstream
``from_pretrained`` path is currently incompatible with the installed
huggingface_hub version, so this module loads config + state dict manually.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

MUQ_REPO_ID = os.getenv("MUQ_MULAN_REPO_ID", "OpenMuQ/MuQ-MuLan-large")
MUQ_SAMPLE_RATE = 24000
MUQ_EMBEDDING_DIM = 512

_MUQ_MODEL = None
_MUQ_LOCK = threading.Lock()
_TEXT_EMB_LOCK = threading.Lock()
_TEXT_EMB_CACHE: dict[str, list[float]] = {}
_TEXT_EMB_CACHE_MAX = 32

if os.getenv("MUQ_MULAN_LOCAL_FILES_ONLY", "1").strip().lower() not in {"0", "false", "no", "off"}:
    # Set this at import time as well as load time; some HF/Transformers modules
    # cache offline flags when imported by downstream MuQ internals.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def _local_files_only() -> bool:
    return os.getenv("MUQ_MULAN_LOCAL_FILES_ONLY", "1").strip().lower() not in {"0", "false", "no", "off"}


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _use_fp16(device: torch.device) -> bool:
    value = os.getenv("MUQ_MULAN_FP16", "auto").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return device.type == "cuda"
    if value in {"0", "false", "no", "off"}:
        return False
    return device.type == "cuda"


def _download(repo_id: str, filename: str) -> str:
    from huggingface_hub import hf_hub_download

    local_only = _local_files_only()
    try:
        return hf_hub_download(repo_id, filename, local_files_only=local_only)
    except Exception:
        if local_only:
            logger.warning(
                "[MuQ-MuLan] %s not found in local cache. Set MUQ_MULAN_LOCAL_FILES_ONLY=0 for first-time download.",
                filename,
            )
        raise


def _local_snapshot(repo_id: str) -> str | None:
    if not _local_files_only():
        return None
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(repo_id, local_files_only=True)
    except Exception as exc:
        logger.warning("[MuQ-MuLan] local snapshot missing for %s: %s", repo_id, exc)
        return None


def _rewrite_config_to_local_snapshots(config: dict[str, Any]) -> dict[str, Any]:
    """Point nested MuQ/XLM-R model names at cached snapshots in offline mode."""
    if not _local_files_only():
        return config
    for section in ("audio_model", "text_model"):
        name = str((config.get(section) or {}).get("name") or "")
        if not name or "/" not in name:
            continue
        local_path = _local_snapshot(name)
        if local_path:
            config.setdefault(section, {})["name"] = local_path
    return config


def get_muq_model():
    """Return a lazily-loaded MuQ-MuLan model singleton."""
    global _MUQ_MODEL
    if _MUQ_MODEL is not None:
        return _MUQ_MODEL

    with _MUQ_LOCK:
        if _MUQ_MODEL is not None:
            return _MUQ_MODEL

        device = _get_device()
        logger.info("[MuQ-MuLan] Loading %s on %s", MUQ_REPO_ID, device)
        if _local_files_only():
            # MuQ-MuLan internally loads an OpenMuQ audio backbone.  Keep that
            # internal HuggingFace lookup offline as well, otherwise a cached
            # outer model can still block startup on HEAD retries.
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"

        from muq import MuQMuLan

        config_path = _download(MUQ_REPO_ID, "config.json")
        with open(config_path, encoding="utf-8") as fh:
            config: dict[str, Any] = json.load(fh)
        config = _rewrite_config_to_local_snapshots(config)

        model = MuQMuLan(config=config)
        state_path = _download(MUQ_REPO_ID, "pytorch_model.bin")
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            logger.warning(
                "[MuQ-MuLan] state dict loaded with missing=%s unexpected=%s",
                len(missing),
                len(unexpected),
            )

        model = model.to(device)
        if _use_fp16(device):
            model = model.half()
        model.eval()
        _MUQ_MODEL = model
        logger.info("[MuQ-MuLan] Loaded successfully; fp16=%s", _use_fp16(device))
        return _MUQ_MODEL


def _normalise_output(vector: torch.Tensor) -> list[float]:
    arr = vector.detach().float().cpu().numpy().reshape(-1).astype(np.float32)
    return arr.tolist()


def encode_audio_to_muq(audio_array: np.ndarray, sample_rate: int = MUQ_SAMPLE_RATE) -> list[float]:
    """Encode a mono waveform to a 512d MuQ-MuLan audio embedding.

    ``sample_rate`` should normally be 24kHz.  Other rates are resampled here
    for safety, but ingestion should load at 24kHz directly to avoid duplicate
    work.
    """
    if sample_rate != MUQ_SAMPLE_RATE:
        import librosa

        audio_array = librosa.resample(audio_array.astype(np.float32), orig_sr=sample_rate, target_sr=MUQ_SAMPLE_RATE)

    model = get_muq_model()
    device = next(model.parameters()).device
    dtype = torch.float16 if _use_fp16(device) else torch.float32
    with torch.no_grad():
        wav = torch.tensor(audio_array, dtype=dtype, device=device).unsqueeze(0)
        features = model(wavs=wav).squeeze(0)
    return _normalise_output(features)


def encode_text_to_muq(text: str) -> list[float]:
    """Encode natural-language text to a 512d MuQ-MuLan text embedding."""
    key = str(text or "").strip()
    if key in _TEXT_EMB_CACHE:
        return _TEXT_EMB_CACHE[key]

    with _TEXT_EMB_LOCK:
        if key in _TEXT_EMB_CACHE:
            return _TEXT_EMB_CACHE[key]
    model = get_muq_model()
    with torch.no_grad():
        features = model(texts=[key]).squeeze(0)
    embedding = _normalise_output(features)
    with _TEXT_EMB_LOCK:
        if len(_TEXT_EMB_CACHE) >= _TEXT_EMB_CACHE_MAX:
            _TEXT_EMB_CACHE.pop(next(iter(_TEXT_EMB_CACHE)))
        _TEXT_EMB_CACHE[key] = embedding
    return embedding
