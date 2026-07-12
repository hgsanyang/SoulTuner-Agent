"""Zero-shot acoustic attribute probes derived from MuQ-MuLan embeddings.

The probes are intentionally lightweight: compare each song's MuQ audio
embedding with paired text anchors, then percentile-normalise the raw deltas
across the catalog.  These fields are soft ranking evidence only, never hard
filters.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping


PROBE_VERSION = "muq_zeroshot_v1"


@dataclass(frozen=True)
class ProbeAxis:
    name: str
    positive_prompt: str
    negative_prompt: str
    output_field: str


PROBE_AXES: tuple[ProbeAxis, ...] = (
    ProbeAxis(
        name="vocalness",
        positive_prompt="music with clear lead vocals and singing voice",
        negative_prompt="purely instrumental music without vocals or lyrics",
        output_field="acoustic_vocalness",
    ),
    ProbeAxis(
        name="drumness",
        positive_prompt="music with prominent drums percussion beat and rhythmic groove",
        negative_prompt="music with no drums no percussion sparse ambient texture",
        output_field="acoustic_drumness",
    ),
    ProbeAxis(
        name="energy",
        positive_prompt="high energy loud intense fast driving music",
        negative_prompt="very low energy quiet soft calm sleep ambient music",
        output_field="acoustic_energy",
    ),
)


def _to_float_vector(value: Any) -> list[float]:
    if value is None:
        return []
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return []


def cosine_similarity(a: Iterable[Any], b: Iterable[Any]) -> float:
    av = _to_float_vector(a)
    bv = _to_float_vector(b)
    if not av or not bv or len(av) != len(bv):
        return 0.0
    dot = sum(x * y for x, y in zip(av, bv))
    an = math.sqrt(sum(x * x for x in av))
    bn = math.sqrt(sum(y * y for y in bv))
    return float(dot / (an * bn)) if an > 0 and bn > 0 else 0.0


def raw_probe_delta(audio_embedding: Iterable[Any], positive_embedding: Iterable[Any], negative_embedding: Iterable[Any]) -> float:
    """Return a signed zero-shot probe delta for one axis."""
    return cosine_similarity(audio_embedding, positive_embedding) - cosine_similarity(audio_embedding, negative_embedding)


def percentile_scores(raw_scores: Mapping[str, float], *, neutral: float = 0.5) -> dict[str, float]:
    """Map raw deltas to empirical catalog percentiles with tie mid-ranks."""
    if not raw_scores:
        return {}
    values = [float(v) for v in raw_scores.values()]
    if len(values) == 1 or math.isclose(min(values), max(values)):
        return {key: neutral for key in raw_scores}
    denominator = float(len(values) - 1)
    result: dict[str, float] = {}
    for key, raw_value in raw_scores.items():
        value = float(raw_value)
        below = sum(1 for other in values if other < value)
        tied = sum(1 for other in values if math.isclose(other, value))
        mid_rank = below + (tied - 1) / 2.0
        result[key] = mid_rank / denominator
    return result


def build_probe_text_embeddings(encoder) -> dict[str, tuple[list[float], list[float]]]:
    """Encode probe anchors with the supplied text encoder."""
    embeddings: dict[str, tuple[list[float], list[float]]] = {}
    for axis in PROBE_AXES:
        embeddings[axis.name] = (
            list(encoder(axis.positive_prompt)),
            list(encoder(axis.negative_prompt)),
        )
    return embeddings


def score_catalog_embeddings(
    song_embeddings: Mapping[str, Iterable[Any]],
    probe_embeddings: Mapping[str, tuple[Iterable[Any], Iterable[Any]]],
) -> dict[str, dict[str, float]]:
    """Return percentile-normalised probe fields keyed by song id/title."""
    raw_by_axis: dict[str, dict[str, float]] = {axis.name: {} for axis in PROBE_AXES}
    for song_key, audio_embedding in song_embeddings.items():
        for axis in PROBE_AXES:
            pair = probe_embeddings.get(axis.name)
            if not pair:
                continue
            raw_by_axis[axis.name][song_key] = raw_probe_delta(audio_embedding, pair[0], pair[1])

    normalised_by_axis = {name: percentile_scores(values) for name, values in raw_by_axis.items()}
    result: dict[str, dict[str, float]] = {song_key: {} for song_key in song_embeddings}
    for axis in PROBE_AXES:
        for song_key, value in normalised_by_axis.get(axis.name, {}).items():
            result[song_key][axis.output_field] = round(float(value), 6)
    for song_key, fields in result.items():
        if "acoustic_vocalness" in fields:
            fields["acoustic_instrumentalness"] = round(1.0 - fields["acoustic_vocalness"], 6)
        if "acoustic_energy" in fields:
            fields["acoustic_low_energy"] = round(1.0 - fields["acoustic_energy"], 6)
    return result
