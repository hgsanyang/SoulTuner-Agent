"""Offline, human-mediated catalog tag review.

The exporter builds compact tasks that can be handed to a web LLM.  The
importer treats the returned JSONL as untrusted input: identities must match
the frozen task manifest, categorical values are cleaned/capped by the shared
tag policy, and model-reported confidence never becomes the catalog score.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import urlparse

from services.catalog_enrichment import prepare_tag_enrichment
from services.tag_policy import clean_tag_payload

MAX_LYRICS_CHARS = 1200
MAX_EVIDENCE_URLS = 8
_LRC_TIMESTAMP_RE = re.compile(r"^(?:\[(?:\d{1,3}:)?\d{1,2}(?:[.:]\d{1,3})?\])+\s*")
_LRC_METADATA_RE = re.compile(r"^\[(?:ar|al|ti|au|by|offset|re|ve):", re.IGNORECASE)


class OfflineTagReviewError(ValueError):
    """Raised when an external result violates the frozen review contract."""


def make_task(record: Mapping[str, Any], *, lyrics_chars: int = MAX_LYRICS_CHARS) -> dict[str, Any]:
    music_id = _text(record.get("song_id") or record.get("source_id") or record.get("music_id"), 80)
    title = _text(record.get("title"), 220)
    artist = _text(record.get("artist"), 220)
    if not music_id or not title or not artist:
        raise OfflineTagReviewError("task requires music_id, title, and artist")
    identity = f"{music_id}\n{title.casefold()}\n{artist.casefold()}"
    task_id = "tag-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    lyrics = _read_lyrics(record.get("lrc_path"), max_chars=lyrics_chars)
    return {
        "task_id": task_id,
        "music_id": music_id,
        "title": title,
        "artist": artist,
        "album": _text(record.get("album"), 220),
        "release_year": _year(record.get("release_year")),
        "lyrics_excerpt": lyrics,
        "requested_fields": [
            "genres", "moods", "themes", "scenarios", "language", "region", "vibe"
        ],
    }


def build_review_prompt(tasks: Iterable[Mapping[str, Any]]) -> str:
    rows = [dict(task) for task in tasks]
    return """你是音乐目录标签审校员。请对下面每首歌进行联网检索，并结合提供的元数据与歌词片段生成结构化标签。

规则：
1. genres/moods/themes/scenarios 各 0-5 个，不确定就留空，不要凑数。
2. 区分原曲与 Live、重制、翻唱；年份优先原始发行年份，无法确认就留空。
3. 事实信息必须附公开来源 URL；仅从歌词推断的情绪/主题可将 evidence_basis 写为 lyrics。
4. 不要修改 task_id、music_id、title、artist。
5. 只返回 JSONL：每行一个 JSON 对象，不要 Markdown 代码块或解释段落。
6. 可以在 taxonomy_feedback 中提出标签新增、合并、歧义意见，但它不会自动写入标签体系。

每行输出结构：
{"task_id":"...","music_id":"...","title":"...","artist":"...","genres":[],"moods":[],"themes":[],"scenarios":[],"language":"","region":"","vibe":"","evidence_urls":[],"evidence_basis":"web|lyrics|metadata|mixed","decision_reason":"","missing_information":[],"taxonomy_feedback":{"suggested_additions":[],"suggested_merges":[],"uncertain_fields":[]}}

待处理任务：
""" + "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n"


def validate_result(result: Mapping[str, Any], tasks: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    task_id = _text(result.get("task_id"), 80)
    task = tasks.get(task_id)
    if task is None:
        raise OfflineTagReviewError(f"unknown task_id: {task_id or '<empty>'}")
    for field, limit in (("music_id", 80), ("title", 220), ("artist", 220)):
        expected = _text(task.get(field), limit)
        actual = _text(result.get(field), limit)
        if actual != expected:
            raise OfflineTagReviewError(f"{task_id}: {field} does not match frozen task")

    tags = clean_tag_payload(dict(result))
    evidence_urls = _valid_urls(result.get("evidence_urls"))
    evidence_basis = _text(result.get("evidence_basis"), 20).casefold()
    if evidence_basis not in {"web", "lyrics", "metadata", "mixed"}:
        evidence_basis = "web" if evidence_urls else "metadata"
    confidence = 0.70 if evidence_urls else 0.62
    enriched = prepare_tag_enrichment(
        tags,
        source="external_web_llm",
        confidence=confidence,
    )
    return {
        "task_id": task_id,
        "music_id": _text(task.get("music_id"), 80),
        "title": _text(task.get("title"), 220),
        "artist": _text(task.get("artist"), 220),
        **enriched,
        "language": _text(result.get("language"), 40),
        "region": _text(result.get("region"), 80),
        "vibe": _text(result.get("vibe"), 80),
        "evidence_urls": evidence_urls,
        "evidence_basis": evidence_basis,
        "decision_reason": _text(result.get("decision_reason"), 500),
        "missing_information": _clean_text_list(result.get("missing_information"), 8, 120),
        "taxonomy_feedback": _clean_feedback(result.get("taxonomy_feedback")),
    }


def apply_validated_tags(client: Any, row: Mapping[str, Any], *, replace_existing: bool = False) -> bool:
    """Write one validated row. Existing reviewed tags are protected by default."""
    query = """
    MATCH (s:Song)
    WHERE toString(s.music_id) = $music_id OR toString(s.source_id) = $music_id
    WITH s LIMIT 1
    WHERE $replace_existing OR coalesce(s.tag_source, '') IN ['', 'unknown', 'deferred']
    SET s.vibe = $vibe,
        s.language = $language,
        s.region = $region,
        s.tag_source = $tag_source,
        s.tag_confidence_json = $tag_confidence_json,
        s.tag_sources_json = $tag_sources_json,
        s.tag_evidence_urls_json = $tag_evidence_urls_json,
        s.tag_evidence_basis = $tag_evidence_basis,
        s.tag_decision_reason = $tag_decision_reason,
        s.updated_at = timestamp()
    WITH s
    OPTIONAL MATCH (s)-[old_m:HAS_MOOD]->(:Mood) DELETE old_m
    WITH s
    OPTIONAL MATCH (s)-[old_t:HAS_THEME]->(:Theme) DELETE old_t
    WITH s
    OPTIONAL MATCH (s)-[old_sc:FITS_SCENARIO]->(:Scenario) DELETE old_sc
    WITH s
    OPTIONAL MATCH (s)-[old_g:BELONGS_TO_GENRE]->(:Genre) DELETE old_g
    WITH s
    OPTIONAL MATCH (s)-[old_l:HAS_LANGUAGE]->(:Language) DELETE old_l
    WITH s
    OPTIONAL MATCH (s)-[old_r:IN_REGION]->(:Region) DELETE old_r
    WITH s
    FOREACH (value IN $moods | MERGE (n:Mood {name: value}) MERGE (s)-[:HAS_MOOD]->(n))
    WITH s
    FOREACH (value IN $themes | MERGE (n:Theme {name: value}) MERGE (s)-[:HAS_THEME]->(n))
    WITH s
    FOREACH (value IN $scenarios | MERGE (n:Scenario {name: value}) MERGE (s)-[:FITS_SCENARIO]->(n))
    WITH s
    FOREACH (value IN $genres | MERGE (n:Genre {name: value}) MERGE (s)-[:BELONGS_TO_GENRE]->(n))
    WITH s
    FOREACH (_ IN CASE WHEN $language <> '' THEN [1] ELSE [] END |
        MERGE (n:Language {name: $language}) MERGE (s)-[:HAS_LANGUAGE]->(n))
    WITH s
    FOREACH (_ IN CASE WHEN $region <> '' THEN [1] ELSE [] END |
        MERGE (n:Region {name: $region}) MERGE (s)-[:IN_REGION]->(n))
    RETURN elementId(s) AS eid
    """
    params = {
        **{key: row.get(key) or [] for key in ("genres", "moods", "themes", "scenarios")},
        "music_id": row["music_id"],
        "vibe": row.get("vibe") or "",
        "language": row.get("language") or "",
        "region": row.get("region") or "",
        "tag_source": row.get("tag_source") or "external_web_llm",
        "tag_confidence_json": row.get("tag_confidence_json") or "{}",
        "tag_sources_json": row.get("tag_sources_json") or "{}",
        "tag_evidence_urls_json": json.dumps(row.get("evidence_urls") or [], ensure_ascii=False),
        "tag_evidence_basis": row.get("evidence_basis") or "",
        "tag_decision_reason": row.get("decision_reason") or "",
        "replace_existing": bool(replace_existing),
    }
    return bool(client.execute_query(query, params))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise OfflineTagReviewError(f"line {line_number}: invalid JSON: {exc.msg}") from exc
        if not isinstance(value, dict):
            raise OfflineTagReviewError(f"line {line_number}: expected JSON object")
        rows.append(value)
    return rows


def _read_lyrics(path_value: Any, *, max_chars: int) -> str:
    path = Path(str(path_value or ""))
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or _LRC_METADATA_RE.match(stripped):
            continue
        stripped = _LRC_TIMESTAMP_RE.sub("", stripped).strip()
        if stripped:
            lines.append(stripped)
    return "\n".join(lines)[: max(0, int(max_chars))]


def _valid_urls(values: Any) -> list[str]:
    urls: list[str] = []
    for value in values or []:
        text = _text(value, 800)
        parsed = urlparse(text)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or text in urls:
            continue
        urls.append(text)
        if len(urls) >= MAX_EVIDENCE_URLS:
            break
    return urls


def _clean_feedback(value: Any) -> dict[str, list[str]]:
    source = value if isinstance(value, Mapping) else {}
    return {
        "suggested_additions": _clean_text_list(source.get("suggested_additions"), 10, 100),
        "suggested_merges": _clean_text_list(source.get("suggested_merges"), 10, 160),
        "uncertain_fields": _clean_text_list(source.get("uncertain_fields"), 10, 80),
    }


def _clean_text_list(values: Any, limit: int, length: int) -> list[str]:
    result: list[str] = []
    for value in values or []:
        text = _text(value, length)
        if text and text.casefold() not in {item.casefold() for item in result}:
            result.append(text)
        if len(result) >= limit:
            break
    return result


def _text(value: Any, limit: int) -> str:
    return " ".join(str(value or "").strip().split())[:limit]


def _year(value: Any) -> int | None:
    try:
        year = int(value or 0)
    except (TypeError, ValueError):
        return None
    return year if 1900 <= year <= 2100 else None
