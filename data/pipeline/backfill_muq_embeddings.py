"""Backfill MuQ-MuLan embeddings into Neo4j Song nodes.

Default mode encodes local audio files at 24kHz.  For reproducible bake-off
reuse, pass ``--npz ../_muq_bakeoff/muq_corpus.npz`` to write a previously
validated {music_id -> 512d vector} bundle directly into Neo4j.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MUQ_EMBEDDING_DIM = 512
MAX_AUDIO_SECONDS = 60
DEFAULT_BATCH_SIZE = 50

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

    roots.extend(
        [
            PROJECT_ROOT / "data" / "processed_audio" / "audio",
            PROJECT_ROOT.parent / "data" / "processed_audio" / "audio",
        ]
    )

    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        resolved = str(root.expanduser())
        key = resolved.lower()
        if key not in seen:
            seen.add(key)
            unique.append(Path(resolved))
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


def load_npz_embeddings(path: Path) -> dict[str, list[float]]:
    npz = np.load(path, allow_pickle=True)
    ids = [str(item) for item in npz["ids"].tolist()]
    vecs = np.asarray(npz["vecs"], dtype=np.float32)
    if vecs.ndim != 2 or vecs.shape[1] != MUQ_EMBEDDING_DIM:
        raise ValueError(f"Expected {MUQ_EMBEDDING_DIM}d MuQ vectors, got shape={vecs.shape}")
    return {music_id: vec.astype(np.float32).tolist() for music_id, vec in zip(ids, vecs, strict=True)}


def get_driver():
    load_env_file(PROJECT_ROOT / ".env")
    from neo4j import GraphDatabase

    uri = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")
    return GraphDatabase.driver(uri, auth=(user, password))


def create_muq_vector_index(driver) -> None:
    query = f"""
    CREATE VECTOR INDEX song_muq_index IF NOT EXISTS
    FOR (s:Song) ON (s.muq_embedding)
    OPTIONS {{
        indexConfig: {{
            `vector.dimensions`: {MUQ_EMBEDDING_DIM},
            `vector.similarity_function`: 'cosine'
        }}
    }}
    """
    with driver.session() as session:
        session.run(query).consume()


def fetch_candidate_songs(driver, missing_only: bool) -> list[dict[str, Any]]:
    where_missing = "AND s.muq_embedding IS NULL" if missing_only else ""
    query = f"""
    MATCH (s:Song)
    WHERE s.music_id IS NOT NULL
      AND s.audio_url IS NOT NULL
      AND s.m2d2_embedding IS NOT NULL
      {where_missing}
    RETURN s.music_id AS music_id, s.title AS title, s.audio_url AS audio_url
    ORDER BY s.title
    """
    with driver.session() as session:
        return [record.data() for record in session.run(query)]


def write_embeddings(driver, vectors: dict[str, list[float]], batch_size: int) -> int:
    items = [{"music_id": music_id, "embedding": embedding} for music_id, embedding in vectors.items()]
    written = 0
    query = """
    UNWIND $rows AS row
    MATCH (s:Song {music_id: row.music_id})
    SET s.muq_embedding = row.embedding,
        s.updated_at = timestamp()
    RETURN count(s) AS written
    """
    with driver.session() as session:
        for start in range(0, len(items), batch_size):
            batch = items[start : start + batch_size]
            result = session.run(query, {"rows": batch}).single()
            written += int(result["written"] if result else 0)
            logger.info("Wrote MuQ embeddings %s/%s", min(start + batch_size, len(items)), len(items))
    return written


def encode_song_file(audio_path: Path) -> list[float]:
    import librosa
    from retrieval.muq_embedder import encode_audio_to_muq

    wav, _ = librosa.load(audio_path, sr=24000, mono=True, duration=MAX_AUDIO_SECONDS)
    if wav.size < 24000:
        raise ValueError("audio shorter than 1 second")
    return encode_audio_to_muq(wav, sample_rate=24000)


def build_vectors_from_audio(rows: list[dict[str, Any]], audio_roots: list[Path]) -> tuple[dict[str, list[float]], int, int]:
    vectors: dict[str, list[float]] = {}
    missing_file = 0
    errors = 0
    for index, row in enumerate(rows, start=1):
        music_id = str(row["music_id"])
        audio_path = resolve_audio_path(row.get("audio_url"), audio_roots)
        if audio_path is None:
            missing_file += 1
            continue
        try:
            vectors[music_id] = encode_song_file(audio_path)
        except Exception as exc:
            errors += 1
            if errors <= 5:
                logger.warning("Failed to encode %s (%s): %s", row.get("title") or music_id, audio_path.name, exc)
        if index % 50 == 0:
            logger.info("Encoded %s/%s | vectors=%s missing=%s errors=%s", index, len(rows), len(vectors), missing_file, errors)
    return vectors, missing_file, errors


def coverage_report(driver) -> dict[str, int]:
    query = """
    MATCH (s:Song)
    RETURN count(s) AS total,
           count(s.m2d2_embedding) AS m2d2,
           count(s.muq_embedding) AS muq
    """
    with driver.session() as session:
        record = session.run(query).single()
    return {
        "total": int(record["total"] if record else 0),
        "m2d2": int(record["m2d2"] if record else 0),
        "muq": int(record["muq"] if record else 0),
    }


def vector_indexes(driver) -> list[dict[str, Any]]:
    with driver.session() as session:
        return [
            record.data()
            for record in session.run(
                "SHOW INDEXES YIELD name, type, state WHERE type = 'VECTOR' RETURN name, type, state ORDER BY name"
            )
        ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill MuQ-MuLan embeddings into Neo4j")
    parser.add_argument("--npz", type=str, default="", help="Optional bakeoff .npz with ids/vecs to write directly")
    parser.add_argument("--audio-dir", type=str, default="", help="Audio directory override for encode mode")
    parser.add_argument("--all", action="store_true", help="Process all songs, not only missing muq_embedding")
    parser.add_argument("--dry-run", action="store_true", help="Show counts and exit without writing")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--report", type=str, default="")
    args = parser.parse_args()

    driver = get_driver()
    try:
        create_muq_vector_index(driver)
        before = coverage_report(driver)
        rows = fetch_candidate_songs(driver, missing_only=not args.all)
        logger.info("Before coverage: %s", json.dumps(before, ensure_ascii=False))
        logger.info("Candidate songs: %s", len(rows))
        if args.dry_run:
            return

        if args.npz:
            all_vectors = load_npz_embeddings(Path(args.npz))
            candidate_ids = {str(row["music_id"]) for row in rows}
            vectors = {music_id: vec for music_id, vec in all_vectors.items() if music_id in candidate_ids}
            missing_file = 0
            errors = 0
            logger.info("Loaded %s vectors from npz; matched candidates=%s", len(all_vectors), len(vectors))
        else:
            vectors, missing_file, errors = build_vectors_from_audio(rows, candidate_audio_roots(args.audio_dir or None))

        written = write_embeddings(driver, vectors, batch_size=max(1, args.batch_size)) if vectors else 0
        after = coverage_report(driver)
        indexes = vector_indexes(driver)
        report = {
            "created_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
            "source": "npz" if args.npz else "audio",
            "before": before,
            "after": after,
            "candidate_songs": len(rows),
            "vectors_prepared": len(vectors),
            "written": written,
            "missing_file": missing_file,
            "errors": errors,
            "indexes": indexes,
        }
        if args.report:
            report_path = Path(args.report)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            logger.info("Report: %s", report_path)
        logger.info("After coverage: %s", json.dumps(after, ensure_ascii=False))
        logger.info("Vector indexes: %s", json.dumps(indexes, ensure_ascii=False))
    finally:
        driver.close()


if __name__ == "__main__":
    main()
