"""Optional multilingual semantic scorer for query-to-memory relevance."""

from __future__ import annotations

import os
from pathlib import Path
from threading import Lock
from typing import Any


class MemorySemanticScorerUnavailable(RuntimeError):
    pass


class BgeMemorySemanticScorer:
    """Lazy local-only BGE reranker used on the small per-user memory set."""

    _models: dict[tuple[str, str], Any] = {}
    _lock = Lock()

    def __init__(self, *, model_name: str = "BAAI/bge-reranker-v2-m3", device: str | None = None):
        self.model_name = model_name
        requested = str(device or os.getenv("MEMORY_RELEVANCE_DEVICE", "auto")).strip() or "auto"
        self.device = _resolve_device(requested)

    @property
    def name(self) -> str:
        return "bge-reranker-v2-m3"

    def score(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        model = self._get_model()
        values = model.predict(
            [[query, document] for document in documents],
            batch_size=min(16, len(documents)),
            show_progress_bar=False,
        )
        return [max(0.0, min(1.0, float(value))) for value in values]

    def preflight(self) -> dict[str, str]:
        """Load the pinned local model now and expose auditable capabilities."""

        self._get_model()
        source = Path(_local_model_source(self.model_name))
        revision = source.name if source.name != self.model_name else "unresolved"
        return {
            "backend": self.name,
            "model": self.model_name,
            "revision": revision,
            "device": self.device,
            "local_only": "true",
        }

    def _get_model(self):
        key = (self.model_name, self.device)
        with self._lock:
            cached = self._models.get(key)
            if cached is not None:
                return cached
            try:
                from sentence_transformers import CrossEncoder

                source = _local_model_source(self.model_name)
                model = CrossEncoder(source, device=self.device, local_files_only=True)
            except Exception as exc:
                raise MemorySemanticScorerUnavailable("local BGE memory scorer is unavailable") from exc
            self._models[key] = model
            return model


def _local_model_source(model_name: str) -> str:
    cache_root = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    model_dir = cache_root / ("models--" + model_name.replace("/", "--")) / "snapshots"
    snapshots = sorted((path for path in model_dir.glob("*") if path.is_dir()), reverse=True)
    return str(snapshots[0]) if snapshots else model_name


def _resolve_device(requested: str) -> str:
    normalized = requested.strip().lower()
    if normalized != "auto":
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"
