"""Optional text/audio embedding adjustments for dense retrieval."""

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


def _apply_linear_adapter(vector: list[float], spec: dict[str, Any]) -> list[float]:
    matrix = spec.get("matrix")
    bias = spec.get("bias", [])
    if not isinstance(matrix, list) or not matrix:
        return vector
    if not all(isinstance(row, list) and len(row) == len(vector) for row in matrix):
        return vector
    if bias and (not isinstance(bias, list) or len(bias) != len(matrix)):
        return vector

    adjusted = []
    for row_index, row in enumerate(matrix):
        value = sum(float(weight) * float(source) for weight, source in zip(row, vector))
        if bias:
            value += float(bias[row_index])
        adjusted.append(value)

    mix = float(spec.get("mix", 1.0))
    mix = max(0.0, min(1.0, mix))
    if len(adjusted) == len(vector) and mix < 1.0:
        adjusted = [
            (1.0 - mix) * float(original) + mix * float(mapped)
            for original, mapped in zip(vector, adjusted)
        ]
    return _normalize(adjusted) if spec.get("normalize", True) else adjusted


def apply_alignment_adapter(vector: list[float], backend: str) -> list[float]:
    """Apply an optional learned text-side adapter.

    Expected JSON shape:
    {
      "backends": {
        "muq": {
          "type": "linear",
          "input_dim": 512,
          "output_dim": 512,
          "matrix": [[...], ...],
          "bias": [...],
          "normalize": true,
          "mix": 1.0
        }
      }
    }

    Missing files, unknown backends, non-linear types, or dimension mismatches
    are no-ops. This keeps adapter rollout reversible and safe by default.
    """
    path = os.getenv("MUSIC_ALIGNMENT_ADAPTER_PATH", "").strip()
    if not path:
        return vector
    try:
        payload = _load_payload(Path(path))
        backends = payload.get("backends") if isinstance(payload.get("backends"), dict) else payload
        spec = (backends or {}).get(str(backend).lower()) or {}
        if spec.get("type", "linear") != "linear":
            return vector
        input_dim = int(spec.get("input_dim", len(vector)))
        output_dim = int(spec.get("output_dim", len(vector)))
        if input_dim != len(vector) or output_dim <= 0:
            return vector
        return _apply_linear_adapter(vector, spec)
    except Exception:
        return vector


def apply_alignment_calibration(vector: list[float], backend: str) -> list[float]:
    """Apply optional calibration and adapter for a backend.

    Calibration JSON shape:
    {
      "muq": {"scale": 1.0, "bias": [..512 floats..], "normalize": true},
      "m2d": {"scale": 1.0, "bias": [..768 floats..], "normalize": true}
    }

    Adapter JSON is read from MUSIC_ALIGNMENT_ADAPTER_PATH and applied after
    calibration. Missing files, missing backend keys, or dimension mismatch are
    treated as no-ops so production retrieval remains safely reversible.
    """
    path = os.getenv("MUSIC_ALIGNMENT_CALIBRATION_PATH", "").strip()
    adjusted = vector
    try:
        if path:
            payload = _load_payload(Path(path))
            spec = payload.get(str(backend).lower()) or {}
            bias = spec.get("bias")
            scale = float(spec.get("scale", 1.0))
            if isinstance(bias, list) and len(bias) == len(vector):
                adjusted = [float(value) * scale + float(delta) for value, delta in zip(vector, bias)]
                adjusted = _normalize(adjusted) if spec.get("normalize", True) else adjusted
    except Exception:
        adjusted = vector
    return apply_alignment_adapter(adjusted, backend)
