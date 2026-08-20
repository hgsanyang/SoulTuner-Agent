"""Bridge a verified Song Describer manifest into SoulTuner's existing flywheel.

The default mode is a read-only preview.  ``--commit`` performs the same
two-stage path as user-confirmed songs: quick Neo4j metadata insertion followed
by a filesystem ingest job that lets the existing GPU worker compute MuQ/M2D
embeddings.  It does not introduce a second ingestion implementation.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from tools.data.song_describer_pipeline import hash_file, parse_license_url


def _audio_path(cache_dir: Path, relpath: str) -> Path:
    return cache_dir / "audio" / PurePosixPath(relpath)


def manifest_row_to_ingest_song(row: Mapping[str, Any], cache_dir: Path) -> dict[str, Any]:
    """Validate one manifest row and map it to the established ingest contract."""

    required = ("song_id", "title", "artist", "audio_relpath", "audio_sha256", "audio_license")
    missing = [name for name in required if not row.get(name)]
    if missing:
        raise ValueError(f"manifest row {row.get('song_id') or '<unknown>'} missing: {', '.join(missing)}")
    if not row.get("audio_available"):
        raise ValueError(f"manifest row {row['song_id']} does not have materialised audio")
    audio_path = _audio_path(cache_dir, str(row["audio_relpath"]))
    if not audio_path.is_file():
        raise FileNotFoundError(audio_path)
    actual_sha = hash_file(audio_path)
    if actual_sha != row["audio_sha256"]:
        raise ValueError(f"audio SHA-256 mismatch for {row['song_id']}")
    licence = dict(row["audio_license"])
    parse_license_url(str(licence.get("url") or ""))
    release_date = str(row.get("release_date") or "")
    release_year = int(release_date[:4]) if len(release_date) >= 4 and release_date[:4].isdigit() else None
    relpath = str(PurePosixPath(str(row["audio_relpath"])))
    return {
        "song_id": str(row["song_id"]),
        "source_id": str(row["track_id"]),
        "title": str(row["title"]),
        "artist": str(row["artist"]),
        "album": str(row.get("album") or "Unknown"),
        "album_id": str(row.get("album_id") or ""),
        "duration": int(float(row.get("duration_seconds") or 0)),
        "release_year": release_year,
        "ext": "mp3",
        "file_basename": Path(relpath).stem,
        "audio_path": str(audio_path.resolve()),
        "audio_url": f"/static/mtg_audio/{relpath}",
        "cover_url": "",
        "lrc_url": "",
        "source": "song_describer",
        "platform": "song_describer",
        "metadata_source": "song_describer_v1.0.0",
        "audio_retention": "saved",
        "requested_by": "explicit_open_dataset_import",
        "tagging_mode": "deferred",
        "genres": list(row.get("genres") or []),
        "instruments": list(row.get("instruments") or []),
        "moods_themes": list(row.get("moods_themes") or []),
        "captions": list(row.get("captions") or []),
        "dataset_url": str(row.get("dataset_url") or ""),
        "source_url": str(row.get("source_url") or ""),
        "audio_license": licence,
        "audio_sha256": actual_sha,
    }


def load_ingest_songs(manifest_path: Path, cache_dir: Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    songs: list[dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                song = manifest_row_to_ingest_song(json.loads(line), cache_dir)
            except Exception as exc:
                raise ValueError(f"invalid manifest line {line_number}: {exc}") from exc
            songs.append(song)
            if limit is not None and len(songs) >= limit:
                break
    if not songs:
        raise ValueError("manifest has no ingestable songs")
    return songs


def write_provenance_and_tags(songs: Sequence[Mapping[str, Any]]) -> None:
    """Attach open-data provenance and graph tags after the normal quick ingest."""

    from retrieval.neo4j_client import get_neo4j_client

    client = get_neo4j_client()
    query = """
    MATCH (s:Song)
    WHERE toString(s.music_id) = $music_id OR toString(s.source_id) = $source_id
    SET s.source = 'song_describer',
        s.source_platform = 'song_describer',
        s.metadata_source = 'song_describer_v1.0.0',
        s.source_url = $source_url,
        s.dataset_url = $dataset_url,
        s.audio_sha256 = $audio_sha256,
        s.audio_license_id = $audio_license_id,
        s.audio_license_url = $audio_license_url,
        s.audio_attribution = $audio_attribution,
        s.audio_license_json = $audio_license_json,
        s.captions_json = $captions_json,
        s.description = $description,
        s.instruments_json = $instruments_json,
        s.updated_at = timestamp()
    WITH s
    FOREACH (genre IN $genres |
        MERGE (g:Genre {name: genre})
        MERGE (s)-[:BELONGS_TO_GENRE]->(g)
    )
    WITH s
    FOREACH (mood IN $moods_themes |
        MERGE (m:Mood {name: mood})
        MERGE (s)-[:HAS_MOOD]->(m)
    )
    RETURN elementId(s) AS eid
    """
    for song in songs:
        licence = dict(song["audio_license"])
        captions = list(song.get("captions") or [])
        result = client.execute_query(
            query,
            {
                "music_id": str(song["song_id"]),
                "source_id": str(song["source_id"]),
                "source_url": str(song.get("source_url") or ""),
                "dataset_url": str(song.get("dataset_url") or ""),
                "audio_sha256": str(song["audio_sha256"]),
                "audio_license_id": str(licence.get("id") or ""),
                "audio_license_url": str(licence.get("url") or ""),
                "audio_attribution": str(licence.get("attribution_text") or ""),
                "audio_license_json": json.dumps(licence, ensure_ascii=False, sort_keys=True),
                "captions_json": json.dumps(captions, ensure_ascii=False, sort_keys=True),
                "description": str(captions[0].get("text") if captions else ""),
                "instruments_json": json.dumps(song.get("instruments") or [], ensure_ascii=False),
                "genres": list(song.get("genres") or []),
                "moods_themes": list(song.get("moods_themes") or []),
            },
        )
        if not result:
            raise RuntimeError(f"Neo4j provenance update could not find {song['song_id']}")


async def commit_to_existing_flywheel(songs: list[dict[str, Any]], *, batch_size: int) -> list[str]:
    from services.ingest_queue import enqueue_songs
    from tools.acquire_music import _quick_ingest_to_neo4j

    await _quick_ingest_to_neo4j(songs)
    write_provenance_and_tags(songs)
    job_ids: list[str] = []
    for start in range(0, len(songs), max(1, batch_size)):
        job_ids.append(enqueue_songs(songs[start : start + max(1, batch_size)]))
    return job_ids


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Queue verified Song Describer tracks in SoulTuner's ingest flywheel")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--commit", action="store_true", help="Write Neo4j metadata and enqueue GPU enrichment jobs")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cache_dir = args.cache_dir.expanduser().resolve()
    manifest = args.manifest or cache_dir / "artifacts" / "song_describer_validated.jsonl"
    songs = load_ingest_songs(manifest, cache_dir, limit=args.limit)
    result: dict[str, Any] = {
        "mode": "commit" if args.commit else "preview",
        "manifest": str(manifest),
        "song_count": len(songs),
        "song_ids": [song["song_id"] for song in songs],
        "licenses": sorted({song["audio_license"]["id"] for song in songs}),
    }
    if args.commit:
        result["job_ids"] = asyncio.run(commit_to_existing_flywheel(songs, batch_size=args.batch_size))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
