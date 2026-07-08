"""Backfill auditable catalog metadata into Neo4j.

This script performs only source-backed updates:
- Song.artist from existing PERFORMED_BY relationships.
- release/source/retention facts from local online-acquired metadata JSON.
- HAS_LANGUAGE relationships from existing Song.language properties.
- normalized artist/title keys for duplicate diagnosis.
- release_year from already source-backed KnowledgeCard metadata.

It never deletes duplicate Song nodes and never invents release years.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.neo4j_client import get_neo4j_client  # noqa: E402
from services.catalog_enrichment import prepare_tag_enrichment  # noqa: E402

DEFAULT_ONLINE_META_DIR = PROJECT_ROOT.parent / "data" / "online_acquired" / "metadata"


def normalize_artist_name(value: Any) -> str:
    text = str(value or "").casefold().strip()
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[·・,，/／|｜]+", "、", text)
    text = re.sub(r"[^\w\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af、]+", "", text)
    return text.strip("、")


def normalize_title(value: Any) -> str:
    text = str(value or "").casefold().strip()
    text = re.sub(r"\([^)]*(?:live|remaster|cover|伴奏|翻唱|版|edit)[^)]*\)", "", text)
    text = re.sub(r"（[^）]*(?:live|remaster|cover|伴奏|翻唱|版|edit)[^）]*）", "", text)
    text = re.sub(r"[\s\-_/|｜:：,，.。]+", "", text)
    return text


def artist_string(raw_artists: Any) -> str:
    names: list[str] = []
    for item in raw_artists or []:
        if isinstance(item, list) and item:
            name = str(item[0] or "").strip()
        elif isinstance(item, dict):
            name = str(item.get("name") or "").strip()
        else:
            name = str(item or "").strip()
        if name:
            names.append(name)
    return "、".join(dict.fromkeys(names))


def release_year_from_metadata(meta: Mapping[str, Any]) -> int | None:
    raw_year = meta.get("release_year")
    try:
        year = int(raw_year)
        if 1900 <= year <= 2100:
            return year
    except Exception:
        pass

    publish_time = meta.get("publishTime") or meta.get("publish_time")
    try:
        timestamp_ms = int(publish_time)
        if timestamp_ms > 0:
            # Avoid pulling in pandas/dateutil for one deterministic field.
            import datetime as _dt

            return _dt.datetime.fromtimestamp(timestamp_ms / 1000, _dt.timezone.utc).year
    except Exception:
        return None
    return None


def load_online_metadata(meta_dir: Path = DEFAULT_ONLINE_META_DIR) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(meta_dir.glob("*_meta.json")) if meta_dir.exists() else []:
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        music_id = str(meta.get("musicId") or meta.get("music_id") or meta.get("source_id") or "").strip()
        title = str(meta.get("musicName") or meta.get("title") or "").strip()
        artist = artist_string(meta.get("artist") or meta.get("artists"))
        if not music_id and not title:
            continue
        rows.append(
            {
                "music_id": music_id,
                "title": title,
                "artist": artist,
                "artist_normalized": normalize_artist_name(artist),
                "title_normalized": normalize_title(title),
                "album": str(meta.get("album") or "").strip(),
                "release_year": release_year_from_metadata(meta),
                "publish_time": meta.get("publishTime") or meta.get("publish_time"),
                "source": meta.get("source") or "online",
                "source_platform": meta.get("source_platform") or meta.get("metadata_source") or "",
                "source_id": str(meta.get("source_id") or meta.get("musicId") or "").strip(),
                "metadata_source": meta.get("metadata_source") or "",
                "audio_retention": meta.get("audio_retention") or "",
                "is_trial": bool(meta.get("is_trial")) if "is_trial" in meta else None,
                "cover_url": meta.get("cover_url") or "",
            }
        )
    return rows


def backfill_artist_properties(client: Any, *, dry_run: bool = False) -> int:
    rows = client.execute_query(
        """
        MATCH (s:Song)
        OPTIONAL MATCH (s)-[:PERFORMED_BY]->(a:Artist)
        WITH s, [name IN collect(DISTINCT a.name)
                 WHERE name IS NOT NULL AND trim(name) <> ''] AS artists
        WHERE coalesce(s.artist, '') = '' AND size(artists) > 0
        RETURN elementId(s) AS eid, artists
        """
    )
    if dry_run:
        return len(rows)
    updated = 0
    for row in rows:
        artist = "、".join(row.get("artists") or [])
        if not artist:
            continue
        result = client.execute_query(
            """
            MATCH (s:Song)
            WHERE elementId(s) = $eid
            SET s.artist = $artist,
                s.artist_normalized = $artist_normalized
            RETURN count(s) AS n
            """,
            {
                "eid": row["eid"],
                "artist": artist,
                "artist_normalized": normalize_artist_name(artist),
            },
        )
        updated += int(result[0].get("n") or 0) if result else 0
    return updated


def backfill_title_artist_keys(client: Any, *, dry_run: bool = False) -> int:
    rows = client.execute_query(
        """
        MATCH (s:Song)
        RETURN elementId(s) AS eid, s.title AS title, s.artist AS artist
        """
    )
    if dry_run:
        return len(rows)
    updated = 0
    for row in rows:
        result = client.execute_query(
            """
            MATCH (s:Song)
            WHERE elementId(s) = $eid
            SET s.title_normalized = $title_normalized,
                s.artist_normalized = $artist_normalized,
                s.duplicate_key = $duplicate_key
            RETURN count(s) AS n
            """,
            {
                "eid": row["eid"],
                "title_normalized": normalize_title(row.get("title")),
                "artist_normalized": normalize_artist_name(row.get("artist")),
                "duplicate_key": f"{normalize_title(row.get('title'))}::{normalize_artist_name(row.get('artist'))}",
            },
        )
        updated += int(result[0].get("n") or 0) if result else 0
    return updated


def backfill_online_metadata(
    client: Any,
    metadata_rows: list[dict[str, Any]],
    *,
    dry_run: bool = False,
) -> int:
    if dry_run:
        return len(metadata_rows)
    updated = 0
    for row in metadata_rows:
        result = client.execute_query(
            """
            MATCH (s:Song)
            WHERE ($music_id <> '' AND s.music_id = $music_id)
               OR ($source_id <> '' AND s.source_id = $source_id)
               OR (
                    coalesce(s.title_normalized, '') = $title_normalized
                    AND coalesce(s.artist_normalized, '') = $artist_normalized
               )
            SET s.source = coalesce(NULLIF($source, ''), s.source),
                s.source_platform = coalesce(NULLIF($source_platform, ''), s.source_platform),
                s.source_id = coalesce(NULLIF($source_id, ''), s.source_id),
                s.metadata_source = coalesce(NULLIF($metadata_source, ''), s.metadata_source),
                s.audio_retention = coalesce(NULLIF($audio_retention, ''), s.audio_retention),
                s.release_year = coalesce($release_year, s.release_year),
                s.publish_time = coalesce($publish_time, s.publish_time),
                s.album = coalesce(NULLIF($album, ''), s.album),
                s.artist = coalesce(NULLIF($artist, ''), s.artist),
                s.artist_normalized = coalesce(NULLIF($artist_normalized, ''), s.artist_normalized),
                s.title_normalized = coalesce(NULLIF($title_normalized, ''), s.title_normalized),
                s.duplicate_key = coalesce(NULLIF($duplicate_key, ''), s.duplicate_key),
                s.cover_url = coalesce(NULLIF($cover_url, ''), s.cover_url),
                s.is_trial = coalesce($is_trial, s.is_trial)
            RETURN count(DISTINCT s) AS n
            """,
            {
                **row,
                "duplicate_key": f"{row.get('title_normalized', '')}::{row.get('artist_normalized', '')}",
            },
        )
        updated += int(result[0].get("n") or 0) if result else 0
    return updated


def backfill_language_relationships(client: Any, *, dry_run: bool = False) -> int:
    rows = client.execute_query(
        """
        MATCH (s:Song)
        WHERE properties(s)['language'] IS NOT NULL
          AND trim(toString(properties(s)['language'])) <> ''
          AND toLower(trim(toString(properties(s)['language']))) <> 'unknown'
          AND NOT EXISTS { MATCH (s)-[:HAS_LANGUAGE]->(:Language) }
        RETURN count(s) AS n
        """
    )
    count = int(rows[0].get("n") or 0) if rows else 0
    if dry_run or count <= 0:
        return count
    result = client.execute_query(
        """
        MATCH (s:Song)
        WHERE properties(s)['language'] IS NOT NULL
          AND trim(toString(properties(s)['language'])) <> ''
          AND toLower(trim(toString(properties(s)['language']))) <> 'unknown'
          AND NOT EXISTS { MATCH (s)-[:HAS_LANGUAGE]->(:Language) }
        WITH s, trim(toString(properties(s)['language'])) AS language
        MERGE (lang:Language {name: language})
        MERGE (s)-[:HAS_LANGUAGE]->(lang)
        RETURN count(s) AS n
        """
    )
    return int(result[0].get("n") or 0) if result else 0


def backfill_release_year_from_knowledge_cards(client: Any, *, dry_run: bool = False) -> int:
    rows = client.execute_query(
        """
        MATCH (s:Song)-[:HAS_KNOWLEDGE]->(k:KnowledgeCard)
        WHERE properties(s)['release_year'] IS NULL
          AND properties(k)['release_year'] IS NOT NULL
          AND toInteger(k.release_year) >= 1900
          AND toInteger(k.release_year) <= 2100
          AND coalesce(toFloat(k.confidence), 0.0) >= 0.55
        RETURN count(DISTINCT s) AS n
        """
    )
    count = int(rows[0].get("n") or 0) if rows else 0
    if dry_run or count <= 0:
        return count
    result = client.execute_query(
        """
        MATCH (s:Song)-[:HAS_KNOWLEDGE]->(k:KnowledgeCard)
        WHERE properties(s)['release_year'] IS NULL
          AND properties(k)['release_year'] IS NOT NULL
          AND toInteger(k.release_year) >= 1900
          AND toInteger(k.release_year) <= 2100
          AND coalesce(toFloat(k.confidence), 0.0) >= 0.55
        WITH s, k
        ORDER BY coalesce(toFloat(k.confidence), 0.0) DESC, k.updated_at DESC
        WITH s, collect(k)[0] AS best
        SET s.release_year = toInteger(best.release_year),
            s.release_year_source = 'knowledge_card',
            s.updated_at = timestamp()
        RETURN count(s) AS n
        """
    )
    return int(result[0].get("n") or 0) if result else 0


def backfill_tag_provenance_from_relationships(
    client: Any,
    *,
    dry_run: bool = False,
    source: str = "legacy_graph",
    confidence: float = 0.78,
) -> int:
    """Backfill auditable tag provenance from existing graph relationships.

    Older ingests already created Genre/Mood/Theme/Scenario/Language nodes, but
    did not store where those tags came from on the Song node.  This promotes
    the already visible graph tags into the standard provenance JSON fields
    without creating any new labels or inventing tags.
    """

    rows = client.execute_query(
        """
        MATCH (s:Song)
        WHERE coalesce(s.tag_source, '') = ''
        OPTIONAL MATCH (s)-[:BELONGS_TO_GENRE]->(g:Genre)
        OPTIONAL MATCH (s)-[:HAS_MOOD]->(m:Mood)
        OPTIONAL MATCH (s)-[:HAS_THEME]->(t:Theme)
        OPTIONAL MATCH (s)-[:FITS_SCENARIO]->(sc:Scenario)
        OPTIONAL MATCH (s)-[:HAS_LANGUAGE]->(l:Language)
        WITH s,
             [x IN collect(DISTINCT g.name) WHERE x IS NOT NULL AND trim(x) <> ''] AS genres,
             [x IN collect(DISTINCT m.name) WHERE x IS NOT NULL AND trim(x) <> ''] AS moods,
             [x IN collect(DISTINCT t.name) WHERE x IS NOT NULL AND trim(x) <> ''] AS themes,
             [x IN collect(DISTINCT sc.name) WHERE x IS NOT NULL AND trim(x) <> ''] AS scenarios,
             [x IN collect(DISTINCT l.name) WHERE x IS NOT NULL AND trim(x) <> ''] AS languages
        WHERE size(genres) + size(moods) + size(themes) + size(scenarios) + size(languages) > 0
        RETURN elementId(s) AS eid,
               genres, moods, themes, scenarios, languages
        """
    )
    if dry_run:
        return len(rows)
    updated = 0
    for row in rows:
        enriched = prepare_tag_enrichment(
            {
                "genres": row.get("genres") or [],
                "moods": row.get("moods") or [],
                "themes": row.get("themes") or [],
                "scenarios": row.get("scenarios") or [],
                "languages": row.get("languages") or [],
            },
            source=source,
            confidence=confidence,
        )
        result = client.execute_query(
            """
            MATCH (s:Song)
            WHERE elementId(s) = $eid
            SET s.tag_source = $tag_source,
                s.tag_confidence_json = $tag_confidence_json,
                s.tag_sources_json = $tag_sources_json,
                s.updated_at = timestamp()
            RETURN count(s) AS n
            """,
            {
                "eid": row["eid"],
                "tag_source": enriched["tag_source"],
                "tag_confidence_json": enriched["tag_confidence_json"],
                "tag_sources_json": enriched["tag_sources_json"],
            },
        )
        updated += int(result[0].get("n") or 0) if result else 0
    return updated


def mark_unplayable_stubs(client: Any, *, dry_run: bool = False) -> int:
    rows = client.execute_query(
        """
        MATCH (s:Song)
        WHERE coalesce(properties(s)['unplayable_stub'], false) <> true
          AND (s.audio_url IS NULL OR trim(toString(s.audio_url)) = '')
        RETURN count(s) AS n
        """
    )
    count = int(rows[0].get("n") or 0) if rows else 0
    if dry_run or count <= 0:
        return count
    result = client.execute_query(
        """
        MATCH (s:Song)
        WHERE coalesce(properties(s)['unplayable_stub'], false) <> true
          AND (s.audio_url IS NULL OR trim(toString(s.audio_url)) = '')
        SET s.unplayable_stub = true,
            s.audio_status = coalesce(s.audio_status, 'missing'),
            s.acquire_status = coalesce(s.acquire_status, 'needs_audio'),
            s.updated_at = timestamp()
        RETURN count(s) AS n
        """
    )
    return int(result[0].get("n") or 0) if result else 0


def duplicate_diagnosis(client: Any, *, limit: int = 20) -> list[dict[str, Any]]:
    rows = client.execute_query(
        """
        MATCH (s:Song)
        WITH coalesce(s.duplicate_key, '') AS key, collect({
            music_id: s.music_id,
            title: s.title,
            artist: s.artist,
            source: s.source,
            audio_url: s.audio_url,
            audio_retention: s.audio_retention,
            acquire_status: s.acquire_status,
            unplayable_stub: coalesce(s.unplayable_stub, false)
        }) AS songs, count(s) AS n
        WHERE key <> '' AND n > 1
        RETURN key, n, songs[0..5] AS examples
        ORDER BY n DESC, key
        LIMIT $limit
        """,
        {"limit": int(limit)},
    )
    for row in rows:
        row["resolution_hint"] = duplicate_resolution_hint(row.get("examples") or [])
    return rows


def duplicate_resolution_hint(examples: list[dict[str, Any]]) -> dict[str, str]:
    """Return an auditable duplicate strategy without mutating catalog data."""

    playable = [
        row for row in examples
        if row.get("audio_url") and row.get("unplayable_stub") is not True
    ]
    saved = [row for row in playable if row.get("audio_retention") == "saved"]
    local = [
        row for row in playable
        if str(row.get("source") or "").lower() in {"local", "netease", "ingested"}
    ]
    if saved:
        canonical = saved[0]
        reason = "prefer saved playable audio as canonical"
    elif local:
        canonical = local[0]
        reason = "prefer local/ingested playable audio as canonical"
    elif playable:
        canonical = playable[0]
        reason = "prefer any playable audio as review candidate"
    elif examples:
        canonical = examples[0]
        reason = "no playable audio; keep manual review only"
    else:
        canonical = {}
        reason = "empty duplicate group"

    action = "manual_review"
    if canonical and playable:
        action = "review_merge_metadata_only"
    if canonical and not playable:
        action = "do_not_merge_until_audio_resolved"
    return {
        "action": action,
        "canonical_music_id": str(canonical.get("music_id") or ""),
        "reason": reason,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill source-backed catalog metadata into Neo4j.")
    parser.add_argument("--online-meta-dir", default=str(DEFAULT_ONLINE_META_DIR))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--duplicate-limit", type=int, default=20)
    args = parser.parse_args()

    client = get_neo4j_client()
    metadata_rows = load_online_metadata(Path(args.online_meta_dir))
    result = {
        "dry_run": bool(args.dry_run),
        "online_metadata_rows": len(metadata_rows),
        "artist_property_updates": backfill_artist_properties(client, dry_run=args.dry_run),
        "title_artist_key_updates": backfill_title_artist_keys(client, dry_run=args.dry_run),
        "online_metadata_updates": backfill_online_metadata(client, metadata_rows, dry_run=args.dry_run),
        "language_relationship_updates": backfill_language_relationships(client, dry_run=args.dry_run),
        "tag_provenance_updates": backfill_tag_provenance_from_relationships(client, dry_run=args.dry_run),
        "release_year_from_knowledge_updates": backfill_release_year_from_knowledge_cards(
            client,
            dry_run=args.dry_run,
        ),
        "unplayable_stub_updates": mark_unplayable_stubs(client, dry_run=args.dry_run),
        "duplicates": duplicate_diagnosis(client, limit=args.duplicate_limit) if not args.dry_run else [],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
