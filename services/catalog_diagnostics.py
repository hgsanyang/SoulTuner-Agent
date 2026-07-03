"""Catalog and recommendation-bias diagnostics for UI and A3 review."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from services.feedback_logger import SLATE_FEEDBACK_FILE, load_jsonl


UNKNOWN_VALUES = {"", "unknown", "none", "null", "na", "n/a", "未知", "未标注"}


def _clean_label(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    canonical = text.casefold()
    aliases = {
        "r&b": "R&B",
        "hip-hop": "Hip-Hop",
        "hip hop": "Hip-Hop",
        "lo-fi": "Lo-fi",
        "lofi": "Lo-fi",
        "edm": "EDM",
    }
    if canonical in aliases:
        return aliases[canonical]
    if text.isascii():
        return text.title()
    return text


def _is_known(value: Any) -> bool:
    return _clean_label(value).casefold() not in UNKNOWN_VALUES


def _iter_values(value: Any) -> list[str]:
    seen: set[str] = set()
    values: list[str] = []
    raw_values = value if isinstance(value, list) else [value]
    for raw in raw_values:
        item = _clean_label(raw)
        key = item.casefold()
        if key and _is_known(item) and key not in seen:
            seen.add(key)
            values.append(item)
    return values


def _top(counter: Counter[str], limit: int = 10) -> list[dict[str, Any]]:
    total = sum(counter.values()) or 1
    return [
        {"label": label, "count": count, "ratio": round(count / total, 4)}
        for label, count in counter.most_common(limit)
    ]


def _coverage(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    total = len(rows)
    known = sum(1 for row in rows if _iter_values(row.get(field)))
    return {
        "known": known,
        "missing": max(total - known, 0),
        "ratio": round(known / total, 4) if total else 0.0,
    }


def _push_warning(warnings: list[dict[str, Any]], code: str, message: str, severity: str = "info") -> None:
    warnings.append({"code": code, "severity": severity, "message": message})


def summarize_catalog_bias(
    catalog_rows: list[dict[str, Any]],
    exposures: list[dict[str, Any]],
    slate_feedback: list[dict[str, Any]],
    *,
    recent_exposure_limit: int = 50,
) -> dict[str, Any]:
    """Build a deterministic diagnostic report from catalog rows and logs."""
    language = Counter()
    genres = Counter()
    moods = Counter()
    themes = Counter()
    scenarios = Counter()
    sources = Counter()
    playable = 0
    muq = 0
    m2d = 0

    for row in catalog_rows:
        if _is_known(row.get("audio_url")):
            playable += 1
        if row.get("has_muq_embedding"):
            muq += 1
        if row.get("has_m2d2_embedding"):
            m2d += 1
        for value in _iter_values(row.get("language")):
            language[value] += 1
        for value in _iter_values(row.get("genres")):
            genres[value] += 1
        for value in _iter_values(row.get("moods")):
            moods[value] += 1
        for value in _iter_values(row.get("themes")):
            themes[value] += 1
        for value in _iter_values(row.get("scenarios")):
            scenarios[value] += 1
        for value in _iter_values(row.get("source")):
            sources[value] += 1

    recent_exposures = sorted(exposures, key=lambda row: int(row.get("ts") or 0), reverse=True)[:recent_exposure_limit]
    exposed_sources = Counter()
    exposed_labels = Counter()
    exposed_genres = Counter()
    exposed_artists = Counter()
    exposed_count = 0
    for exposure in recent_exposures:
        for item in exposure.get("items") or []:
            exposed_count += 1
            for value in _iter_values(item.get("source")):
                exposed_sources[value] += 1
            for value in _iter_values(item.get("recall_sources")):
                exposed_labels[value] += 1
            for value in _iter_values(item.get("genres")):
                exposed_genres[value] += 1
            for value in _iter_values(item.get("artist")):
                exposed_artists[value] += 1

    feedback_ratings = Counter(str(row.get("rating") or "").strip() for row in slate_feedback if row.get("rating"))
    feedback_reasons = Counter()
    for row in slate_feedback:
        for reason in _iter_values(row.get("reasons")):
            feedback_reasons[reason] += 1

    warnings: list[dict[str, Any]] = []
    total = len(catalog_rows)
    for name, counter, threshold in (
        ("language", language, 0.75),
        ("genre", genres, 0.45),
        ("mood", moods, 0.45),
    ):
        if counter and total:
            top_label, top_count = counter.most_common(1)[0]
            ratio = top_count / total
            if ratio >= threshold:
                _push_warning(
                    warnings,
                    f"{name}_concentration",
                    f"{name} 分布集中：{top_label} 占比约 {ratio:.0%}",
                    "warning",
                )
    if total and playable / total < 0.85:
        _push_warning(warnings, "playable_coverage", f"可播放覆盖率约 {playable / total:.0%}", "warning")
    if total and muq / total < 0.75:
        _push_warning(warnings, "muq_coverage", f"MuQ 文搜音向量覆盖率约 {muq / total:.0%}", "info")
    if exposed_sources:
        top_label, top_count = exposed_sources.most_common(1)[0]
        if exposed_count and top_count / exposed_count > 0.65:
            _push_warning(
                warnings,
                "recent_source_concentration",
                f"最近推荐来源集中：{top_label} 占比约 {top_count / exposed_count:.0%}",
                "info",
            )
    if feedback_reasons.get("太像我的旧歌单") or feedback_reasons.get("想发现更多新歌"):
        _push_warning(
            warnings,
            "user_wants_discovery",
            "近期歌单反馈显示用户希望减少旧歌单惯性、增加发现感",
            "warning",
        )

    return {
        "catalog": {
            "total_songs": total,
            "playable_songs": playable,
            "muq_embedding_songs": muq,
            "m2d2_embedding_songs": m2d,
            "coverage": {
                "language": _coverage(catalog_rows, "language"),
                "genres": _coverage(catalog_rows, "genres"),
                "moods": _coverage(catalog_rows, "moods"),
                "themes": _coverage(catalog_rows, "themes"),
                "scenarios": _coverage(catalog_rows, "scenarios"),
            },
            "top": {
                "languages": _top(language),
                "genres": _top(genres),
                "moods": _top(moods),
                "themes": _top(themes),
                "scenarios": _top(scenarios),
                "sources": _top(sources),
            },
        },
        "recent_recommendations": {
            "exposures": len(recent_exposures),
            "items": exposed_count,
            "top_sources": _top(exposed_sources),
            "top_recall_sources": _top(exposed_labels),
            "top_genres": _top(exposed_genres),
            "top_artists": _top(exposed_artists),
        },
        "slate_feedback": {
            "count": len(slate_feedback),
            "ratings": _top(feedback_ratings),
            "reasons": _top(feedback_reasons),
        },
        "warnings": warnings,
    }


def load_feedback_diagnostics(feedback_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return (
        load_jsonl(feedback_dir / "exposures.jsonl"),
        load_jsonl(feedback_dir / SLATE_FEEDBACK_FILE),
    )
