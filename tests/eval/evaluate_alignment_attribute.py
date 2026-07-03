"""Evaluate attribute precision for text-to-audio retrieval backends.

This is the recommendation-facing alignment ruler for M3.  It measures whether
natural-language attribute queries retrieve songs with matching catalog labels.
It is deterministic and does not call any LLM.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "tests" / "eval" / "results"

FROZEN_ATTRIBUTE_QUERIES = [
    {"id": "lang_zh_chinese", "query_language": "zh", "text": "中文歌曲，华语流行", "target": {"field": "language", "equals": "chinese"}},
    {"id": "lang_en_chinese", "query_language": "en", "text": "a song sung in Mandarin Chinese", "target": {"field": "language", "equals": "chinese"}},
    {"id": "lang_zh_japanese", "query_language": "zh", "text": "日语歌曲，日本乐队", "target": {"field": "language", "equals": "japanese"}},
    {"id": "lang_en_japanese", "query_language": "en", "text": "a song sung in Japanese", "target": {"field": "language", "equals": "japanese"}},
    {"id": "lang_zh_english", "query_language": "zh", "text": "英文歌曲，欧美流行", "target": {"field": "language", "equals": "english"}},
    {"id": "lang_en_english", "query_language": "en", "text": "a song sung in English", "target": {"field": "language", "equals": "english"}},
    {"id": "lang_zh_instrumental", "query_language": "zh", "text": "纯音乐，没有人声的器乐", "target": {"field": "language", "equals": "instrumental"}},
    {"id": "lang_en_instrumental", "query_language": "en", "text": "instrumental music with no vocals", "target": {"field": "language", "equals": "instrumental"}},
    {"id": "genre_zh_rock", "query_language": "zh", "text": "激烈的摇滚乐，电吉他", "target": {"field": "genres", "contains_any": ["rock", "metal", "punk"]}},
    {"id": "genre_en_rock", "query_language": "en", "text": "aggressive rock music with electric guitars", "target": {"field": "genres", "contains_any": ["rock", "metal", "punk"]}},
    {"id": "genre_zh_electronic", "query_language": "zh", "text": "电子音乐，合成器和节拍", "target": {"field": "genres", "contains_any": ["electronic", "edm", "dance"]}},
    {"id": "genre_en_electronic", "query_language": "en", "text": "electronic dance music with synth beat", "target": {"field": "genres", "contains_any": ["electronic", "edm", "dance"]}},
    {"id": "genre_zh_folk", "query_language": "zh", "text": "民谣，木吉他，温暖的人声", "target": {"field": "genres", "contains_any": ["folk", "acoustic", "singer-songwriter"]}},
    {"id": "genre_en_folk", "query_language": "en", "text": "warm acoustic folk song", "target": {"field": "genres", "contains_any": ["folk", "acoustic", "singer-songwriter"]}},
    {"id": "mood_zh_sad", "query_language": "zh", "text": "悲伤忧郁的伤感歌曲", "target": {"field": "moods", "contains_any": ["melancholy", "sad", "lonely"]}},
    {"id": "mood_en_sad", "query_language": "en", "text": "sad melancholic emotional song", "target": {"field": "moods", "contains_any": ["melancholy", "sad", "lonely"]}},
    {"id": "mood_zh_energetic", "query_language": "zh", "text": "热血激昂、充满力量的歌", "target": {"field": "moods", "contains_any": ["energetic", "passionate", "happy"]}},
    {"id": "mood_en_energetic", "query_language": "en", "text": "energetic upbeat powerful song", "target": {"field": "moods", "contains_any": ["energetic", "passionate", "happy"]}},
    {"id": "mood_zh_relaxing", "query_language": "zh", "text": "放松舒缓，适合安静休息", "target": {"field": "moods", "contains_any": ["relaxing", "peaceful", "healing"]}},
    {"id": "mood_en_relaxing", "query_language": "en", "text": "relaxing peaceful gentle music", "target": {"field": "moods", "contains_any": ["relaxing", "peaceful", "healing"]}},
    {"id": "scenario_zh_driving", "query_language": "zh", "text": "适合开车公路旅行听的歌", "target": {"field": "scenarios", "contains_any": ["driving", "travel"]}},
    {"id": "scenario_en_driving", "query_language": "en", "text": "music for driving and road trips", "target": {"field": "scenarios", "contains_any": ["driving", "travel"]}},
    {"id": "scenario_zh_late_night", "query_language": "zh", "text": "深夜独处时听的歌", "target": {"field": "scenarios", "contains_any": ["late night", "rainy day"]}},
    {"id": "scenario_en_late_night", "query_language": "en", "text": "late night lonely listening music", "target": {"field": "scenarios", "contains_any": ["late night", "rainy day"]}},
]


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _git_info() -> dict[str, Any]:
    def run_git(args: list[str]) -> str:
        try:
            return subprocess.check_output(
                ["git", *args],
                cwd=REPO_ROOT,
                text=True,
                encoding="utf-8",
                stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:
            return "unknown"

    status = run_git(["status", "--short"])
    return {
        "sha": run_git(["rev-parse", "--short=12", "HEAD"]),
        "branch": run_git(["branch", "--show-current"]),
        "dirty": bool(status),
    }


def _normalise_matrix(vectors: list[list[float]]) -> np.ndarray:
    if not vectors:
        return np.zeros((0, 0), dtype=np.float32)
    matrix = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _normalise_vector(vector: list[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(arr)
    return arr if norm == 0 else arr / norm


def _clean_list(values: Any) -> list[str]:
    if isinstance(values, list):
        return [str(item).lower() for item in values if item]
    return [str(values).lower()] if values else []


def label_matches(labels: dict[str, Any], target: dict[str, Any]) -> bool:
    field = str(target["field"])
    values = _clean_list(labels.get(field))
    if "equals" in target:
        return any(value == str(target["equals"]).lower() for value in values)
    wanted = [str(item).lower() for item in target.get("contains_any", [])]
    return any(want in value or value in want for value in values for want in wanted)


def precision_at_k(ranked_ids: list[str], labels_by_id: dict[str, dict[str, Any]], target: dict[str, Any], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    top = ranked_ids[:k]
    if not top:
        return 0.0
    hits = sum(1 for music_id in top if label_matches(labels_by_id.get(music_id, {}), target))
    return hits / len(top)


def _fetch_common_corpus() -> tuple[list[str], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    _load_env_file(REPO_ROOT / ".env")
    from neo4j import GraphDatabase

    uri = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")
    query = """
    MATCH (s:Song)
    WHERE s.music_id IS NOT NULL
      AND s.m2d2_embedding IS NOT NULL
      AND s.muq_embedding IS NOT NULL
    OPTIONAL MATCH (s)-[:BELONGS_TO_GENRE]->(g:Genre)
    OPTIONAL MATCH (s)-[:HAS_MOOD]->(m:Mood)
    OPTIONAL MATCH (s)-[:FITS_SCENARIO]->(sc:Scenario)
    RETURN s.music_id AS music_id,
           s.title AS title,
           s.language AS language,
           collect(DISTINCT g.name) AS genres,
           collect(DISTINCT m.name) AS moods,
           collect(DISTINCT sc.name) AS scenarios,
           s.m2d2_embedding AS m2d2_embedding,
           s.muq_embedding AS muq_embedding,
           properties(s)['clamp3_embedding'] AS clamp3_embedding
    ORDER BY s.music_id
    """
    with GraphDatabase.driver(uri, auth=(user, password)) as driver:
        with driver.session() as session:
            rows = [record.data() for record in session.run(query)]

    ids = [str(row["music_id"]) for row in rows]
    labels_by_id = {
        str(row["music_id"]): {
            "title": row.get("title") or "",
            "language": (row.get("language") or "").lower(),
            "genres": _clean_list(row.get("genres")),
            "moods": _clean_list(row.get("moods")),
            "scenarios": _clean_list(row.get("scenarios")),
        }
        for row in rows
    }
    corpora = {
        "m2d": {"ids": ids, "matrix": _normalise_matrix([row["m2d2_embedding"] for row in rows])},
        "muq": {"ids": ids, "matrix": _normalise_matrix([row["muq_embedding"] for row in rows])},
    }
    clamp3_rows = [row for row in rows if row.get("clamp3_embedding")]
    if clamp3_rows:
        corpora["clamp3"] = {
            "ids": [str(row["music_id"]) for row in clamp3_rows],
            "matrix": _normalise_matrix([row["clamp3_embedding"] for row in clamp3_rows]),
        }
    return ids, labels_by_id, corpora


def _rank_ids(query_vector: list[float], matrix: np.ndarray, ids: list[str]) -> list[str]:
    vector = _normalise_vector(query_vector)
    scores = matrix @ vector
    order = np.argsort(-scores, kind="mergesort")
    return [ids[int(index)] for index in order]


def evaluate_attribute_alignment(
    k: int = 10,
    calibration_path: str = "",
    query_variants: bool = False,
    include_clamp3: bool = False,
) -> dict[str, Any]:
    from retrieval.audio_embedder import encode_text_to_embedding
    from retrieval.alignment_calibration import apply_alignment_calibration
    from retrieval.muq_embedder import MUQ_REPO_ID, encode_text_to_muq
    from tools.semantic_search import build_dense_query_variants, _mean_vectors

    previous_calibration_path = os.environ.get("MUSIC_ALIGNMENT_CALIBRATION_PATH")
    if calibration_path:
        os.environ["MUSIC_ALIGNMENT_CALIBRATION_PATH"] = calibration_path
    ids, labels_by_id, corpora = _fetch_common_corpus()
    rows = []
    grouped: dict[str, dict[str, list[float]]] = {}
    backends = ["m2d", "muq"]
    clamp3_status = "not_requested"
    clamp3_model = ""
    encode_text_to_clamp3 = None
    if include_clamp3:
        if "clamp3" not in corpora:
            clamp3_status = "missing_corpus"
        else:
            try:
                from retrieval.clamp3_embedder import (
                    CLAMP3_REPO_URL,
                    clamp3_repo_dir,
                    encode_text_to_clamp3 as _encode_text_to_clamp3,
                )

                clamp3_repo_dir()
                encode_text_to_clamp3 = _encode_text_to_clamp3
                clamp3_status = "available"
                clamp3_model = CLAMP3_REPO_URL
                backends.append("clamp3")
            except Exception as exc:
                clamp3_status = f"unavailable: {exc}"

    def _encode_text(text: str, backend: str) -> list[float]:
        if backend == "clamp3":
            if encode_text_to_clamp3 is None:
                raise RuntimeError("CLaMP3 text encoder is unavailable")
            encoder = encode_text_to_clamp3
        else:
            encoder = encode_text_to_muq if backend == "muq" else encode_text_to_embedding
        if not query_variants:
            return encoder(text)
        variants = build_dense_query_variants(text)
        vectors = [encoder(variant) for variant in variants if variant]
        return _mean_vectors(vectors) if len(vectors) > 1 else vectors[0]

    try:
        for query in FROZEN_ATTRIBUTE_QUERIES:
            lang = query["query_language"]
            precision_by_backend: dict[str, float | None] = {}
            top_ids_by_backend: dict[str, list[str]] = {}
            grouped.setdefault(lang, {backend: [] for backend in backends})
            for backend in backends:
                corpus = corpora.get(backend)
                if not corpus or not len(corpus["ids"]):
                    precision_by_backend[backend] = None
                    top_ids_by_backend[backend] = []
                    continue
                query_vector = apply_alignment_calibration(
                    _encode_text(query["text"], backend),
                    backend,
                )
                ranked = _rank_ids(query_vector, corpus["matrix"], corpus["ids"])
                precision = precision_at_k(ranked, labels_by_id, query["target"], k)
                precision_by_backend[backend] = precision
                top_ids_by_backend[backend] = ranked[:k]
                grouped[lang].setdefault(backend, []).append(precision)
            rows.append(
                {
                    **query,
                    "precision_at_k": precision_by_backend,
                    "top_ids": top_ids_by_backend,
                }
            )
    finally:
        if calibration_path:
            if previous_calibration_path is None:
                os.environ.pop("MUSIC_ALIGNMENT_CALIBRATION_PATH", None)
            else:
                os.environ["MUSIC_ALIGNMENT_CALIBRATION_PATH"] = previous_calibration_path

    def mean(values: list[float]) -> float:
        return float(sum(values) / len(values)) if values else 0.0

    by_language = {
        lang: {backend: mean(values) for backend, values in scores.items()}
        for lang, scores in sorted(grouped.items())
    }
    all_scores = {
        backend: [
            row["precision_at_k"][backend]
            for row in rows
            if row["precision_at_k"].get(backend) is not None
        ]
        for backend in backends
    }
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "git": _git_info(),
        "models": {
            "m2d": "M2D-CLAP",
            "muq": MUQ_REPO_ID,
            "clamp3": clamp3_model,
        },
        "corpus": {
            "common_m2d_muq_songs": len(ids),
            "clamp3_songs": len(corpora.get("clamp3", {}).get("ids", [])),
        },
        "calibration_path": calibration_path or "",
        "query_variants": bool(query_variants),
        "include_clamp3": bool(include_clamp3),
        "clamp3_status": clamp3_status,
        "k": k,
        "query_count": len(FROZEN_ATTRIBUTE_QUERIES),
        "metrics": {
            "overall": {backend: mean(values) for backend, values in all_scores.items()},
            "by_query_language": by_language,
        },
        "queries": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate M2D vs MuQ attribute precision@K")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--output", default="")
    parser.add_argument("--calibration-path", default="", help="Optional alignment calibration JSON")
    parser.add_argument("--query-variants", action="store_true", help="Evaluate multi-view dense query variants")
    parser.add_argument("--include-clamp3", action="store_true", help="Include optional CLaMP3 backend if clamp3_embedding exists")
    args = parser.parse_args()

    report = evaluate_attribute_alignment(
        k=args.k,
        calibration_path=args.calibration_path,
        query_variants=args.query_variants,
        include_clamp3=args.include_clamp3,
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(args.output) if args.output else RESULTS_DIR / (
        "alignment_attribute_eval_"
        + report["git"]["sha"]
        + "_"
        + datetime.now().strftime("%Y%m%d_%H%M%S")
        + ".json"
    )
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    metrics = report["metrics"]
    print("=" * 72)
    print("Attribute Text-to-Audio Alignment Eval")
    print("=" * 72)
    print(f"Git: {report['git']['branch']} @ {report['git']['sha']} | dirty={report['git']['dirty']}")
    print(f"Corpus common M2D/MuQ songs: {report['corpus']['common_m2d_muq_songs']}")
    if report.get("include_clamp3"):
        print(f"CLaMP3 corpus songs: {report['corpus']['clamp3_songs']} | status={report['clamp3_status']}")
    print(f"Queries: {report['query_count']} | P@{report['k']}")
    if report.get("calibration_path"):
        print(f"Calibration: {report['calibration_path']}")
    if report.get("query_variants"):
        print("Query variants: enabled")
    print("Overall: " + " | ".join(f"{name.upper()}={value:.3f}" for name, value in metrics["overall"].items()))
    for lang, values in metrics["by_query_language"].items():
        print(f"{lang}: " + " | ".join(f"{name.upper()}={value:.3f}" for name, value in values.items()))
    print(f"Report: {out}")


if __name__ == "__main__":
    main()
