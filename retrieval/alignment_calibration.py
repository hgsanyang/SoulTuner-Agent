"""Optional text/audio embedding gap calibration for dense retrieval."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any


def _norm(vector: list[float]) -> float:
    return math.sqrt(sum(float(x) * float(x) for x in vector))


def _normalize(vector: list[float]) -> list[float]:
    n = _norm(vector)
    return [float(x) / n for x in vector] if n > 0 else vector


def build_bias_calibration(
    pairs: list[tuple[list[float], list[float]]],
    *,
    scale: float = 1.0,
    normalize: bool = True,
) -> dict[str, Any]:
    """Build a simple text->audio gap calibration from matched vector pairs.

    Each pair is (text_query_vector, positive_audio_centroid).  The resulting
    bias is the average centroid-text delta.  It is deliberately small and
    auditable: no labels from evaluation rows are hidden inside the runtime
    code, and production only applies it when an explicit JSON path is set.
    """
    valid_pairs = [
        (text, target)
        for text, target in pairs
        if len(text) == len(target) and len(text) > 0
    ]
    if not valid_pairs:
        return {"scale": scale, "bias": [], "normalize": normalize, "num_pairs": 0}
    dim = len(valid_pairs[0][0])
    deltas = [0.0 for _ in range(dim)]
    for text, target in valid_pairs:
        for i, (source_value, target_value) in enumerate(zip(text, target)):
            deltas[i] += float(target_value) - float(source_value)
    bias = [value / len(valid_pairs) for value in deltas]
    return {
        "scale": float(scale),
        "bias": bias,
        "normalize": bool(normalize),
        "num_pairs": len(valid_pairs),
    }


def _load_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def apply_alignment_calibration(vector: list[float], backend: str) -> list[float]:
    """Apply optional calibration from MUSIC_ALIGNMENT_CALIBRATION_PATH.

    Expected JSON shape:
    {
      "muq": {"scale": 1.0, "bias": [..512 floats..], "normalize": true},
      "m2d": {"scale": 1.0, "bias": [..768 floats..], "normalize": true}
    }

    Missing files, missing backend keys, or dimension mismatch are treated as a
    no-op so production retrieval remains safely reversible.
    """
    path = os.getenv("MUSIC_ALIGNMENT_CALIBRATION_PATH", "").strip()
    if not path:
        return vector
    try:
        payload = _load_payload(Path(path))
        spec = payload.get(str(backend).lower()) or {}
        bias = spec.get("bias")
        scale = float(spec.get("scale", 1.0))
        if not isinstance(bias, list) or len(bias) != len(vector):
            return vector
        adjusted = [float(value) * scale + float(delta) for value, delta in zip(vector, bias)]
        return _normalize(adjusted) if spec.get("normalize", True) else adjusted
    except Exception:
        return vector
