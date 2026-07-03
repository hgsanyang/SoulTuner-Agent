"""Backfill optional CLaMP3 embeddings into Neo4j Song nodes.

This is an offline bake-off utility, not part of the default online stack.
It requires a local checkout of https://github.com/sanderwood/clamp3 and
``CLAMP3_REPO_DIR`` pointing at that repo.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CLAMP3_EMBEDDING_DIM = 768
DEFAULT_BATCH_SIZE = 20

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def candidate_audio_roots(explicit_audio_dir: str | None = None) -> list[Path]:
    roots: list[Path] = []
    if explicit_audio_dir:
        roots.append(Path(explicit_audio_dir))
    env_audio = os.getenv("MUSIC_AUDIO_DATA_DIR")
    if env_audio:
        roots.append(Path(env_audio))
    data_root = os.getenv("MUSIC_DATA_PATH")
    if data_root:
        roots.append(Path(data_root) / "processed_audio" / "audio")
    roots.extend([
        PROJECT_ROOT / "data" / "processed_audio" / "audio",
        PROJECT_ROOT.parent / "data" / "processed_audio" / "audio",
    ])

    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        resolved = root.expanduser()
        key = str(resolved).lower()
        if key not in seen:
            seen.add(key)
            unique.append(resolved)
    return unique


def resolve_audio_path(audio_url: str | None, roots: list[Path]) -> Path | None:
    if not audio_url:
        return None
    raw = str(audio_url).replace("\\", "/")
    basename = raw.rsplit("/", 1)[-1]
    if not basename:
        return None
    direct = Path(raw)
    if direct.is_file():
        return direct
    for root in roots:
        candidate = root / basename
        if candidate.is_file():
            return candidate
    return None


def get_driver():
    load_env_file(PROJECT_ROOT / ".env")
    from neo4j import GraphDatabase

    uri = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")
    return GraphDatabase.driver(uri, auth=(user, password))


def create_clamp3_vector_index(driver) -> None:
    query = f"""
    CREATE VECTOR INDEX song_clamp3_index IF NOT EXISTS
    FOR (s:Song) ON (s.clamp3_embedding)
    OPTIONS {{
        indexConfig: {{
            `vector.dimensions`: {CLAMP3_EMBEDDING_DIM},
            `vector.similarity_function`: 'cosine'
        }}
    }}
    """
    with driver.session() as session:
        session.run(query).consume()


def fetch_candidate_songs(driver, missing_only: bool) -> list[dict[str, Any]]:
    where_missing = "AND s.clamp3_embedding IS NULL" if missing_only else ""
    query = f"""
    MATCH (s:Song)
    WHERE s.music_id IS NOT NULL
      AND s.audio_url IS NOT NULL
      {where_missing}
    RETURN s.music_id AS music_id, s.title AS title, s.audio_url AS audio_url
    ORDER BY s.title
    """
    with driver.session() as session:
        return [record.data() for record in session.run(query)]


def write_embeddings(driver, vectors: dict[str, list[float]], batch_size: int) -> int:
    rows = [{"music_id": music_id, "embedding": embedding} for music_id, embedding in vectors.items()]
    query = """
    UNWIND $rows AS row
    MATCH (s:Song {music_id: row.music_id})
    SET s.clamp3_embedding = row.embedding,
        s.updated_at = timestamp()
    RETURN count(s) AS written
    """
    written = 0
    with driver.session() as session:
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            result = session.run(query, {"rows": batch}).single()
            written += int(result["written"] if result else 0)
            logger.info("Wrote CLaMP3 embeddings %s/%s", min(start + batch_size, len(rows)), len(rows))
    return written


def build_vectors_from_audio(rows: list[dict[str, Any]], roots: list[Path]) -> tuple[dict[str, list[float]], int, int]:
    from retrieval.clamp3_embedder import encode_audio_file_to_clamp3

    vectors: dict[str, list[float]] = {}
    missing = 0
    errors = 0
    for index, row in enumerate(rows, start=1):
        music_id = str(row["music_id"])
        audio_path = resolve_audio_path(row.get("audio_url"), roots)
        if audio_path is None:
            missing += 1
            continue
        try:
            vector = encode_audio_file_to_clamp3(audio_path)
            if len(vector) != CLAMP3_EMBEDDING_DIM:
                raise ValueError(f"wrong dimension: {len(vector)}")
            vectors[music_id] = vector
        except Exception as exc:
            errors += 1
            if errors <= 5:
                logger.warning("Failed to encode %s (%s): %s", row.get("title") or music_id, audio_path.name, exc)
        if index % 20 == 0:
            logger.info("Encoded %s/%s | vectors=%s missing=%s errors=%s", index, len(rows), len(vectors), missing, errors)
    return vectors, missing, errors


def coverage_report(driver) -> dict[str, int]:
    query = """
    MATCH (s:Song)
    RETURN count(s) AS total,
           count(s.m2d2_embedding) AS m2d2,
           count(s.muq_embedding) AS muq,
           count(s.clamp3_embedding) AS clamp3
    """
    with driver.session() as session:
        record = session.run(query).single()
    return {
        "total": int(record["total"] if record else 0),
        "m2d2": int(record["m2d2"] if record else 0),
        "muq": int(record["muq"] if record else 0),
        "clamp3": int(record["clamp3"] if record else 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill optional CLaMP3 embeddings into Neo4j")
    parser.add_argument("--audio-dir", default="", help="Audio directory override")
    parser.add_argument("--all", action="store_true", help="Process all songs, not only missing clamp3_embedding")
    parser.add_argument("--dry-run", action="store_true", help="Show counts and exit without writing")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit for smoke bake-off")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()

    from retrieval.clamp3_embedder import Clamp3UnavailableError, clamp3_repo_dir

    try:
        repo = clamp3_repo_dir()
        logger.info("Using CLaMP3 repo: %s", repo)
    except Clamp3UnavailableError as exc:
        raise SystemExit(str(exc)) from exc

    roots = candidate_audio_roots(args.audio_dir or None)
    driver = get_driver()
    with driver:
        create_clamp3_vector_index(driver)
        rows = fetch_candidate_songs(driver, missing_only=not args.all)
        if args.limit > 0:
            rows = rows[: args.limit]
        logger.info("Candidate songs: %s | coverage=%s", len(rows), coverage_report(driver))
        if args.dry_run:
            return
        vectors, missing, errors = build_vectors_from_audio(rows, roots)
        written = write_embeddings(driver, vectors, args.batch_size)
        logger.info("Done: written=%s missing_files=%s errors=%s coverage=%s", written, missing, errors, coverage_report(driver))


if __name__ == "__main__":
    main()
