"""Graph + Dense retrieval over the public synthetic SoulTuner catalog."""

from __future__ import annotations

import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any


DATA = Path(__file__).resolve().parent / "data" / "catalog.jsonl"


@lru_cache(maxsize=1)
def load_catalog() -> tuple[dict[str, Any], ...]:
    rows = [
        json.loads(line)
        for line in DATA.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return tuple(rows)


def _cosine(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    denominator = math.sqrt(sum(a * a for a in left) * sum(b * b for b in right)) or 1.0
    return numerator / denominator


def _query_vector(query: str, dimensions: int = 8) -> list[float]:
    raw = hashlib.sha256(query.encode("utf-8")).digest()
    values = [((raw[index] / 255.0) * 2.0) - 1.0 for index in range(dimensions)]
    if "低音" in query or "bass" in query.casefold():
        values[0] += 1.5
    if "鼓" in query or "节奏" in query:
        values[1] += 1.5
    if "温暖" in query or "治愈" in query:
        values[2] += 1.5
    if "安静" in query or "睡前" in query:
        values[3] -= 1.5
    norm = math.sqrt(sum(value * value for value in values)) or 1.0
    return [value / norm for value in values]


def _graph_score(query: str, row: dict[str, Any], plan: dict[str, Any]) -> float:
    searchable = " ".join(
        [
            row["language"],
            str(row["decade"]),
            *row["genres"],
            *row["moods"],
            *row["scenarios"],
        ]
    )
    terms = [
        *plan["hints"].get("mood", []),
        *plan["hints"].get("scenario", []),
        *plan["hints"].get("genre", []),
    ]
    language = plan["hard"].get("language")
    if language:
        terms.append(language)
    era = plan["metadata"].get("era")
    if era:
        terms.append(str(era).replace("年代", ""))
    terms.extend(term for term in ["温暖", "治愈", "专注", "活力"] if term in query)
    unique_terms = {term for term in terms if term}
    if not unique_terms:
        return 0.5
    matches = sum(1 for term in unique_terms if term in searchable)
    return matches / len(unique_terms)


def _acoustic_score(query: str, row: dict[str, Any]) -> float:
    score = (_cosine(_query_vector(query), row["demo_embedding"]) + 1.0) / 2.0
    acoustic = row["acoustic"]
    if "低音" in query or "bass" in query.casefold():
        score = (score + acoustic["bass"]) / 2.0
    if "鼓" in query or "节奏" in query:
        score = (score + acoustic["percussion"]) / 2.0
    if "温暖" in query or "治愈" in query:
        score = (score + acoustic["warmth"]) / 2.0
    if "运动" in query or "高能量" in query:
        score = (score + acoustic["energy"]) / 2.0
    if "安静" in query or "睡前" in query:
        score = (score + (1.0 - acoustic["energy"])) / 2.0
    return score


def _preference_score(row: dict[str, Any], preference_tags: set[str]) -> float:
    if not preference_tags:
        return 0.0
    row_tags = set(row["moods"] + row["genres"] + row["scenarios"])
    return len(row_tags & preference_tags) / len(preference_tags)


def _reason(row: dict[str, Any], graph_score: float, dense_score: float) -> str:
    tags = "、".join((row["moods"] + row["genres"] + row["scenarios"])[:3])
    if dense_score > graph_score + 0.12:
        return f"听感向量更接近当前描述，并带有{tags}特征。"
    if graph_score > dense_score + 0.12:
        return f"目录标签与请求中的{tags}约束高度一致。"
    return f"目录条件和听感相似度共同支持，核心特征为{tags}。"


def retrieve(
    query: str,
    plan: dict[str, Any],
    route: dict[str, Any],
    top_k: int = 8,
    preference_tags: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Return deterministic fused results; only public synthetic data is read."""
    preferences = preference_tags or set()
    scored: list[dict[str, Any]] = []
    for row in load_catalog():
        graph_score = _graph_score(query, row, plan)
        dense_score = _acoustic_score(query, row)
        preference_score = _preference_score(row, preferences)
        base_score = route["graph_weight"] * graph_score + route["dense_weight"] * dense_score
        final_score = min(1.0, base_score * 0.9 + preference_score * 0.1)
        scored.append(
            {
                "song_id": row["song_id"],
                "title": row["title"],
                "artist": row["artist"],
                "language": row["language"],
                "decade": row["decade"],
                "tags": row["moods"] + row["genres"] + row["scenarios"],
                "graph_score": round(graph_score, 3),
                "dense_score": round(dense_score, 3),
                "preference_score": round(preference_score, 3),
                "final_score": round(final_score, 3),
                "reason": _reason(row, graph_score, dense_score),
                "audio_available": bool(row.get("audio_available")),
            }
        )
    return sorted(scored, key=lambda item: (-item["final_score"], item["song_id"]))[:top_k]
