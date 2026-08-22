"""Graph + Dense retrieval over SoulTuner's licensed public demo catalog."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

try:
    from .graph_runtime import merge_graph_overlay
    from .graph_runtime import vector_query_scores
    from .dense_runtime import encode_text_query
except ImportError:  # ModelScope uploads this directory as a flat application.
    from graph_runtime import merge_graph_overlay
    from graph_runtime import vector_query_scores
    from dense_runtime import encode_text_query


DATA = Path(__file__).resolve().parent / "data" / "catalog.jsonl"
_AUDIO_SUFFIXES = {".mp3", ".flac", ".ogg", ".opus", ".wav", ".m4a", ".aiff"}
_TAG_ALIASES = {
    "旅行": ("travel", "journey", "road", "driving", "trip"),
    "驾驶": ("driving", "road", "travel", "car"),
    "公路": ("road", "driving", "travel", "journey"),
    "摇滚": ("rock", "guitar", "alternative", "metal", "punk"),
    "流行": ("pop", "singer-songwriter"),
    "电子": ("electronic", "electronica", "synth"),
    "爵士": ("jazz", "swing"),
    "民谣": ("folk", "acoustic", "singer-songwriter"),
    "说唱": ("rap", "hiphop", "hip-hop"),
    "嘻哈": ("hiphop", "hip-hop", "rap"),
    "温暖": ("warm", "gentle", "comforting"),
    "治愈": ("healing", "hopeful", "gentle", "warm"),
    "平静": ("calm", "peaceful", "quiet", "relaxing"),
    "活力": ("energetic", "upbeat", "lively"),
    "雨天": ("rain", "rainy", "cozy", "intimate"),
    "夜晚": ("night", "nighttime", "late-night"),
}


def catalog_path() -> Path:
    configured = os.getenv("SOULTUNER_CATALOG_PATH", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    persistent = Path("/mnt/workspace/soultuner/open_audio/catalog.jsonl")
    return persistent if persistent.is_file() else DATA


def audio_root() -> Path:
    configured = os.getenv("SOULTUNER_AUDIO_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    persistent = Path("/mnt/workspace/soultuner/open_audio/audio")
    return persistent if persistent.is_dir() else (DATA.parent / "audio").resolve()


@lru_cache(maxsize=4)
def _load_catalog(path: str) -> tuple[dict[str, Any], ...]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    return tuple(rows)


def load_catalog() -> tuple[dict[str, Any], ...]:
    return merge_graph_overlay(_load_catalog(str(catalog_path())))


def _string_list(row: dict[str, Any], *keys: str) -> list[str]:
    values: list[str] = []
    for key in keys:
        raw = row.get(key) or []
        if isinstance(raw, str):
            raw = [raw]
        if isinstance(raw, (list, tuple)):
            values.extend(str(item).strip() for item in raw if str(item).strip())
    return list(dict.fromkeys(values))


def _row_tags(row: dict[str, Any]) -> list[str]:
    return _string_list(row, "moods", "moods_themes", "genres", "scenarios", "instruments")


def _row_language(row: dict[str, Any]) -> str:
    return str(row.get("language") or "未知")


def _row_decade(row: dict[str, Any]) -> int:
    try:
        return int(row.get("decade"))
    except (TypeError, ValueError):
        release_date = str(row.get("release_date") or "")
        year = int(release_date[:4]) if release_date[:4].isdigit() else 0
        return (year // 10) * 10 if year else 0


def _catalog_description(row: dict[str, Any]) -> str:
    captions = row.get("captions") or []
    caption_text = [str(item.get("text") or "") if isinstance(item, dict) else str(item) for item in captions]
    return " ".join(
        [
            str(row.get("title") or ""),
            str(row.get("artist") or ""),
            *_row_tags(row),
            *caption_text,
        ]
    ).casefold()


def resolve_audio_source(row: dict[str, Any]) -> str | None:
    """Return a safe Gradio-playable path/URL from a catalog row.

    Local catalog values are constrained to SOULTUNER_AUDIO_ROOT so a public
    row can never make Gradio expose an arbitrary file from /mnt/workspace.
    """

    root = audio_root()
    values = (
        row.get("audio_relpath"),
        row.get("audio_path"),
        row.get("audio_url"),
        row.get("preview_url"),
    )
    for raw in values:
        value = str(raw or "").strip()
        if not value:
            continue
        parsed = urlparse(value)
        if parsed.scheme in {"http", "https"} and parsed.netloc:
            return value
        if parsed.scheme or parsed.netloc:
            continue
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = root / candidate
        try:
            resolved = candidate.expanduser().resolve()
            resolved.relative_to(root)
        except (OSError, RuntimeError, ValueError):
            continue
        if resolved.is_file() and resolved.suffix.casefold() in _AUDIO_SUFFIXES:
            return str(resolved)
    return None


@lru_cache(maxsize=256)
def _cover_data_url(path: str) -> str:
    payload = Path(path).read_bytes()
    if len(payload) > 512 * 1024:
        raise ValueError("cover fallback is too large")
    return "data:image/svg+xml;base64," + base64.b64encode(payload).decode("ascii")


def resolve_cover_source(row: dict[str, Any]) -> str:
    remote = str(row.get("cover_url") or "").strip()
    if remote.startswith(("http://", "https://")):
        return remote
    relative = str(row.get("cover_fallback_path") or "").strip()
    if not relative:
        return ""
    root = catalog_path().parent.resolve()
    try:
        candidate = (root / relative).resolve()
        candidate.relative_to(root)
    except (OSError, RuntimeError, ValueError):
        return ""
    if not candidate.is_file() or candidate.suffix.casefold() != ".svg":
        return ""
    try:
        return _cover_data_url(str(candidate))
    except (OSError, ValueError):
        return ""


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
            _row_language(row),
            str(_row_decade(row)),
            *_row_tags(row),
            _catalog_description(row),
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
    unique_terms = {str(term).casefold() for term in terms if term}
    if not unique_terms:
        return 0.5
    matches = sum(
        1
        for term in unique_terms
        if any(alias.casefold() in searchable for alias in (term, *_TAG_ALIASES.get(term, ())))
    )
    return matches / len(unique_terms)


def _description_score(query: str, row: dict[str, Any]) -> float:
    """Score the open-audio catalog before materialised MuQ vectors exist."""

    searchable = _catalog_description(row)
    concepts = (
        (("安静", "静谧", "quiet"), ("quiet", "calm", "meditative", "soft", "floaty", "lounge")),
        (("氛围", "空间感", "ambient"), ("ambient", "atmospheric", "floaty", "forest", "lounge")),
        (("不压抑", "明亮", "希望", "hopeful"), ("hopeful", "warm", "bright", "cool", "groovy")),
        (("低音", "bass"), ("bass", "bassline")),
        (("鼓", "节奏", "drum"), ("drum", "percussion", "beat", "fast-paced")),
        (("温暖", "治愈", "warm"), ("warm", "hopeful", "gentle", "easylistening")),
        (("摇滚", "metal", "rock"), ("rock", "metal", "guitar")),
    )
    matched = 0
    requested = 0
    descriptor_hits = 0
    lowered_query = query.casefold()
    for triggers, descriptors in concepts:
        if any(trigger in lowered_query for trigger in triggers):
            requested += 1
            current_hits = sum(descriptor in searchable for descriptor in descriptors)
            descriptor_hits += current_hits
            if current_hits:
                matched += 1
    english_terms = {token for token in re.findall(r"[a-z][a-z0-9-]+", lowered_query) if len(token) > 2}
    overlap = len(english_terms & set(re.findall(r"[a-z][a-z0-9-]+", searchable)))
    return min(
        0.95,
        0.30 + (matched / max(1, requested)) * 0.4 + min(descriptor_hits, 7) * 0.035 + min(overlap, 2) * 0.05,
    )


def _acoustic_score(query: str, row: dict[str, Any]) -> tuple[float, str]:
    embedding = row.get("demo_embedding")
    acoustic = row.get("acoustic")
    if not isinstance(embedding, list) or not embedding or not isinstance(acoustic, dict):
        return _description_score(query, row), "catalog_descriptions"

    score = (_cosine(_query_vector(query, len(embedding)), embedding) + 1.0) / 2.0
    if "低音" in query or "bass" in query.casefold():
        score = (score + float(acoustic.get("bass", score))) / 2.0
    if "鼓" in query or "节奏" in query:
        score = (score + float(acoustic.get("percussion", score))) / 2.0
    if "温暖" in query or "治愈" in query:
        score = (score + float(acoustic.get("warmth", score))) / 2.0
    if "运动" in query or "高能量" in query:
        score = (score + float(acoustic.get("energy", score))) / 2.0
    if "安静" in query or "睡前" in query:
        score = (score + (1.0 - float(acoustic.get("energy", 0.5)))) / 2.0
    return score, "demo_embedding"


def _preference_score(row: dict[str, Any], preference_tags: set[str]) -> float:
    if not preference_tags:
        return 0.0
    row_tags = set(_row_tags(row))
    return len(row_tags & preference_tags) / len(preference_tags)


def _reason(row: dict[str, Any], graph_score: float, dense_score: float, dense_source: str) -> str:
    tags = "、".join(_row_tags(row)[:3]) or "人工描述"
    if dense_source == "catalog_descriptions":
        return f"目录标签与人工听感描述共同支持，核心特征为{tags}。"
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
    """Return deterministic fused results; only reviewed public demo data is read."""
    preferences = preference_tags or set()
    dense_scores: dict[str, float] = {}
    dense_backend = "catalog_descriptions"
    if float(route.get("dense_weight") or 0.0) > 0:
        acoustic_queries = plan.get("acoustic_queries") or []
        semantic_query = str(acoustic_queries[0] if acoustic_queries else query).strip()
        query_vector = encode_text_query(semantic_query)
        if query_vector:
            dense_scores, vector_status = vector_query_scores(
                query_vector,
                index_name="song_m2d2_index",
                limit=max(120, int(top_k) * 24),
            )
            if vector_status.get("state") == "ready" and dense_scores:
                dense_backend = "m2d2_aura"
    scored: list[dict[str, Any]] = []
    for row in load_catalog():
        graph_score = _graph_score(query, row, plan)
        if dense_backend == "m2d2_aura":
            dense_score = dense_scores.get(str(row.get("song_id") or ""), 0.0)
            dense_source = dense_backend
        else:
            dense_score, dense_source = _acoustic_score(query, row)
        preference_score = _preference_score(row, preferences)
        base_score = route["graph_weight"] * graph_score + route["dense_weight"] * dense_score
        final_score = min(1.0, base_score * 0.9 + preference_score * 0.1)
        scored.append(
            {
                # Keep the catalog row only until ranking is complete.  Audio
                # and cover validation touch the persistent filesystem, so
                # doing that work for all 1,806 candidates made every request
                # pay thousands of stat/resolve calls before the first card
                # could be shown.  Only the returned slate needs those fields.
                "_catalog_row": row,
                "song_id": row["song_id"],
                "title": row["title"],
                "artist": row["artist"],
                "language": _row_language(row),
                "decade": _row_decade(row),
                "tags": _row_tags(row),
                "graph_score": round(graph_score, 3),
                "graph_backend": str(row.get("graph_backend") or "local_catalog"),
                "enrichment_status": str(row.get("enrichment_status") or "pending"),
                "dense_score": round(dense_score, 3),
                "preference_score": round(preference_score, 3),
                "final_score": round(final_score, 3),
                "reason": _reason(row, graph_score, dense_score, dense_source),
                "dense_source": dense_source,
                "license": str(row.get("license") or row.get("license_id") or ""),
                "license_url": str(row.get("license_url") or ""),
                "attribution": str(row.get("attribution") or ""),
                "source_url": str(row.get("source_url") or ""),
                "cover_fallback_path": str(row.get("cover_fallback_path") or ""),
                "cover_attribution": str(row.get("cover_attribution") or ""),
                "cover_source_page_url": str(row.get("cover_source_page_url") or ""),
                "cover_provider": str(row.get("cover_provider") or ""),
            }
        )
    selected = sorted(scored, key=lambda item: (-item["final_score"], item["song_id"]))[:top_k]
    for item in selected:
        catalog_row = item.pop("_catalog_row")
        audio_source = resolve_audio_source(catalog_row)
        item["audio_available"] = bool(audio_source)
        item["audio_source"] = audio_source
        item["cover_url"] = resolve_cover_source(catalog_row)
    return selected
