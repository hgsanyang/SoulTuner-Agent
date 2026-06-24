"""Diagnose text-to-audio alignment ruler failure modes.

This script is intentionally offline with respect to LLMs: it only uses the
frozen captions, Neo4j audio vectors, and the local M2D-CLAP text encoder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from tests.eval.alignment_metrics import first_relevant_rank, summarize_alignment_ranks
from tests.eval.evaluate_alignment import DEFAULT_GOLD, RESULTS_DIR, _git_info, _load_env_file, _load_gold

REPO_ROOT = Path(__file__).resolve().parents[2]

ACOUSTIC_CAPTIONS: dict[str, str] = {
    "185712": (
        "A mid-tempo Mandarin pop rock track with acoustic guitar strums, warm male vocals, "
        "steady drums, and a bright singalong chorus."
    ),
    "32070215": (
        "A long ambient recording of low thunder rumble and rain noise with no vocals, "
        "very slow motion, and a soft broadband texture."
    ),
    "29154677": (
        "A mellow indie pop track with gentle male vocals, clean guitars, soft drums, "
        "and a wistful mid-tempo groove."
    ),
    "1142673": (
        "A fast roots rock song with raspy male vocals, driving electric guitar, punchy drums, "
        "and a bright boogie rhythm."
    ),
    "19188939": (
        "A noisy experimental rock track with distorted guitars, raw drums, tense vocals, "
        "and an abrasive high-energy texture."
    ),
    "447926067": (
        "A warm Mandarin folk song with acoustic guitar, relaxed male vocals, light percussion, "
        "and a gentle walking tempo."
    ),
    "486855953": (
        "A playful acoustic folk-pop song with light strummed guitar, soft female vocals, "
        "simple percussion, and a sunny relaxed feel."
    ),
    "2744750011": (
        "A slow indie rock track with subdued Mandarin vocals, reverb guitar, sparse drums, "
        "and a smoky melancholic atmosphere."
    ),
    "1642632": (
        "An instrumental ambient piece with soft drones, sparse experimental textures, "
        "no clear beat, and a quiet atmospheric tone."
    ),
    "22714590": (
        "An upbeat Japanese indie rock song with bright electric guitars, energetic drums, "
        "youthful male vocals, and a fast driving tempo."
    ),
    "1305648755": (
        "A dark post-punk rock song with deep vocals, echoing guitars, steady drums, "
        "and a gloomy mid-tempo pulse."
    ),
    "29734857": (
        "A cinematic instrumental piece with repeating piano or organ patterns, swelling strings, "
        "a gradual build, and spacious orchestral ambience."
    ),
    "18317149": (
        "A classic rock ballad with powerful layered vocals, piano, electric guitar, drums, "
        "and a dramatic gospel-like chorus."
    ),
    "1974125151": (
        "A polished mid-tempo pop track with bright synths, light drums, smooth male vocals, "
        "and a bittersweet dance-pop groove."
    ),
    "1449626870": (
        "A gentle acoustic folk song with fingerpicked guitar, intimate male vocals, soft harmonies, "
        "and a calm warm texture."
    ),
}

SANITY_QUERIES = [
    {
        "id": "distorted_fast_rock",
        "phrase": "aggressive distorted electric guitar, fast tempo, screaming vocals",
        "expected_terms": {"rock", "metal", "punk", "guitar", "hard", "noise", "experimental"},
    },
    {
        "id": "cinematic_piano_strings",
        "phrase": "soft solo piano, slow tempo, cinematic strings, no vocals",
        "expected_terms": {"classical", "ambient", "cinematic", "instrumental", "piano", "strings"},
    },
    {
        "id": "rain_thunder_sleep",
        "phrase": "ambient rain and thunder field recording, no rhythm, for sleep",
        "expected_terms": {"ambient", "sleep", "rain", "thunder", "nature", "instrumental"},
    },
    {
        "id": "warm_acoustic_folk",
        "phrase": "warm acoustic guitar and gentle male vocal, relaxed folk song",
        "expected_terms": {"folk", "acoustic", "singer-songwriter", "warm", "relaxing"},
    },
    {
        "id": "dreamy_electronic_night",
        "phrase": "dreamy electronic synth beat with reverb, late night mood",
        "expected_terms": {"electronic", "dreamy", "indie", "urban", "late night"},
    },
]

STOPWORDS = {
    "a",
    "an",
    "and",
    "for",
    "in",
    "of",
    "the",
    "to",
    "with",
    "song",
    "track",
    "music",
    "moods",
    "themes",
    "contexts",
    "suited",
    "conveying",
}


def _fetch_diagnostic_corpus() -> list[dict[str, Any]]:
    _load_env_file(REPO_ROOT / ".env")
    from neo4j import GraphDatabase

    uri = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")
    query = """
    MATCH (s:Song)
    WHERE s.m2d2_embedding IS NOT NULL AND s.music_id IS NOT NULL
    OPTIONAL MATCH (s)-[:PERFORMED_BY]->(a:Artist)
    OPTIONAL MATCH (s)-[:BELONGS_TO_GENRE]->(g:Genre)
    OPTIONAL MATCH (s)-[:HAS_MOOD]->(m:Mood)
    OPTIONAL MATCH (s)-[:HAS_THEME]->(t:Theme)
    OPTIONAL MATCH (s)-[:FITS_SCENARIO]->(sc:Scenario)
    RETURN s.music_id AS music_id,
           elementId(s) AS element_id,
           s.title AS title,
           collect(DISTINCT a.name) AS artists,
           s.language AS language,
           s.region AS region,
           s.vibe AS vibe,
           collect(DISTINCT g.name) AS genres,
           collect(DISTINCT m.name) AS moods,
           collect(DISTINCT t.name) AS themes,
           collect(DISTINCT sc.name) AS scenarios,
           s.m2d2_embedding AS embedding
    """
    with GraphDatabase.driver(uri, auth=(user, password)) as driver:
        with driver.session() as session:
            return [record.data() for record in session.run(query)]


def _normalise_matrix(vectors: list[list[float]]) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _normalise_vector(vector: list[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(arr)
    return arr if norm == 0 else arr / norm


def _rank_vector(query: list[float] | np.ndarray, corpus_ids: list[str], corpus_matrix: np.ndarray) -> tuple[list[str], np.ndarray]:
    query_vector = _normalise_vector(query)
    scores = corpus_matrix @ query_vector
    order = np.argsort(-scores, kind="mergesort")
    return [corpus_ids[int(index)] for index in order], scores


def _public_song(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "music_id": str(row.get("music_id") or ""),
        "title": row.get("title") or "",
        "artists": [str(item) for item in row.get("artists") or [] if item],
        "language": row.get("language") or "",
        "vibe": row.get("vibe") or "",
        "genres": [str(item) for item in row.get("genres") or [] if item],
        "moods": [str(item) for item in row.get("moods") or [] if item],
        "scenarios": [str(item) for item in row.get("scenarios") or [] if item],
    }


def _song_blob(row: dict[str, Any]) -> str:
    values = [
        row.get("title") or "",
        row.get("language") or "",
        row.get("vibe") or "",
        *(row.get("artists") or []),
        *(row.get("genres") or []),
        *(row.get("moods") or []),
        *(row.get("themes") or []),
        *(row.get("scenarios") or []),
    ]
    return " ".join(str(value).lower().replace("_", " ") for value in values if value)


def _audio_self_check(gold_items: list[dict[str, Any]], corpus_ids: list[str], corpus_matrix: np.ndarray) -> dict[str, Any]:
    id_to_index: dict[str, int] = {}
    for index, music_id in enumerate(corpus_ids):
        id_to_index.setdefault(music_id, index)

    ranks: list[int | None] = []
    examples: list[dict[str, Any]] = []
    for item in gold_items:
        music_id = str(item["music_id"])
        index = id_to_index.get(music_id)
        if index is None:
            ranks.append(None)
            examples.append({"music_id": music_id, "rank": None, "reason": "missing_from_corpus"})
            continue
        ranked_ids, _scores = _rank_vector(corpus_matrix[index], corpus_ids, corpus_matrix)
        rank = first_relevant_rank(ranked_ids, [music_id])
        ranks.append(rank)
        if rank != 1:
            examples.append({"music_id": music_id, "title": item.get("title") or "", "rank": rank, "top5": ranked_ids[:5]})

    return {"metrics": summarize_alignment_ranks(ranks, ks=(1, 5, 10)), "non_top1_examples": examples[:20]}


def _text_embedding(text: str) -> np.ndarray:
    from retrieval.audio_embedder import encode_text_to_embedding

    return _normalise_vector(encode_text_to_embedding(text))


def _evaluate_text_set(
    items: list[dict[str, Any]],
    corpus_ids: list[str],
    corpus_rows: list[dict[str, Any]],
    corpus_matrix: np.ndarray,
) -> dict[str, Any]:
    rows_by_id = {str(row["music_id"]): row for row in corpus_rows}
    id_to_indices: dict[str, list[int]] = {}
    for index, music_id in enumerate(corpus_ids):
        id_to_indices.setdefault(music_id, []).append(index)

    ranks: list[int | None] = []
    matched_scores: list[float] = []
    random_scores: list[float] = []
    per_item: list[dict[str, Any]] = []
    for item in items:
        music_id = str(item["music_id"])
        query = _text_embedding(item["caption"])
        ranked_ids, scores = _rank_vector(query, corpus_ids, corpus_matrix)
        rank = first_relevant_rank(ranked_ids, [music_id])
        ranks.append(rank)

        own_indices = id_to_indices.get(music_id, [])
        if own_indices:
            matched_scores.append(float(max(scores[index] for index in own_indices)))
        random_index = _stable_random_other_index(music_id, len(corpus_ids), own_indices)
        random_scores.append(float(scores[random_index]))

        per_item.append(
            {
                "music_id": music_id,
                "title": item.get("title") or rows_by_id.get(music_id, {}).get("title") or "",
                "caption": item["caption"],
                "rank": rank,
                "top10": [_public_song(rows_by_id.get(candidate_id, {"music_id": candidate_id})) for candidate_id in ranked_ids[:10]],
            }
        )

    return {
        "metrics": summarize_alignment_ranks(ranks, ks=(1, 5, 10)),
        "cosine": {
            "matched": _summary_stats(matched_scores),
            "random": _summary_stats(random_scores),
            "mean_gap": float(np.mean(matched_scores) - np.mean(random_scores)) if matched_scores and random_scores else 0.0,
        },
        "per_item": per_item,
    }


def _stable_random_other_index(music_id: str, size: int, own_indices: list[int]) -> int:
    if size <= 1:
        return 0
    own = set(own_indices)
    start = int(hashlib.sha256(f"{music_id}:alignment-random".encode("utf-8")).hexdigest()[:8], 16) % size
    for offset in range(size):
        index = (start + offset) % size
        if index not in own:
            return index
    return start


def _summary_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "mean": 0.0, "std": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0}
    arr = np.asarray(values, dtype=np.float32)
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
    }


def _caption_tokens(text: str) -> set[str]:
    tokens = set(re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", text.lower()))
    return {token for token in tokens if token not in STOPWORDS and len(token) > 1}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _caption_distinctiveness(gold_items: list[dict[str, Any]]) -> dict[str, Any]:
    captions = [str(item["caption"]) for item in gold_items]
    normalised = [re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", caption.lower())).strip() for caption in captions]
    token_sets = [_caption_tokens(caption) for caption in captions]

    near_pairs: list[dict[str, Any]] = []
    nearest_scores: list[float] = []
    for i, left in enumerate(token_sets):
        best = 0.0
        best_j = None
        for j in range(i + 1, len(token_sets)):
            score = _jaccard(left, token_sets[j])
            best = max(best, score)
            if score >= 0.72:
                near_pairs.append(
                    {
                        "left_id": str(gold_items[i]["music_id"]),
                        "right_id": str(gold_items[j]["music_id"]),
                        "score": round(score, 3),
                    }
                )
                if best_j is None:
                    best_j = j
        nearest_scores.append(best)

    signatures: dict[str, int] = {}
    for item in gold_items:
        metadata = item.get("metadata") or {}
        signature = json.dumps(
            {
                "language": metadata.get("language") or "",
                "genres": (metadata.get("genres") or [])[:2],
                "moods": (metadata.get("moods") or [])[:2],
                "vibe": metadata.get("vibe") or "",
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        signatures[signature] = signatures.get(signature, 0) + 1

    repeated_signature_groups = sorted([count for count in signatures.values() if count > 1], reverse=True)
    return {
        "caption_count": len(captions),
        "exact_duplicate_count": len(normalised) - len(set(normalised)),
        "near_duplicate_pair_count_jaccard_0_72": len(near_pairs),
        "near_duplicate_pair_rate": len(near_pairs) / max(1, math.comb(len(captions), 2)),
        "nearest_neighbor_jaccard": _summary_stats(nearest_scores),
        "repeated_metadata_signature_groups": repeated_signature_groups[:10],
        "near_duplicate_examples": near_pairs[:20],
    }


def _sanity_queries(
    corpus_ids: list[str],
    corpus_rows: list[dict[str, Any]],
    corpus_matrix: np.ndarray,
) -> list[dict[str, Any]]:
    rows_by_id = {str(row["music_id"]): row for row in corpus_rows}
    results = []
    for query in SANITY_QUERIES:
        query_vector = _text_embedding(query["phrase"])
        ranked_ids, _scores = _rank_vector(query_vector, corpus_ids, corpus_matrix)
        top10_rows = [rows_by_id.get(candidate_id, {"music_id": candidate_id}) for candidate_id in ranked_ids[:10]]
        expected_terms = {str(term).lower() for term in query["expected_terms"]}
        heuristic_hits = []
        for row in top10_rows:
            blob = _song_blob(row)
            matched = sorted(term for term in expected_terms if term in blob)
            heuristic_hits.append({"music_id": str(row.get("music_id") or ""), "matched_terms": matched})
        results.append(
            {
                "id": query["id"],
                "phrase": query["phrase"],
                "expected_terms": sorted(expected_terms),
                "top10_expected_term_hits": sum(1 for hit in heuristic_hits if hit["matched_terms"]),
                "top10": [_public_song(row) for row in top10_rows],
                "heuristic_hits": heuristic_hits,
            }
        )
    return results


def run_diagnostics(gold: dict[str, Any]) -> dict[str, Any]:
    from retrieval.audio_embedder import _M2D_CLAP_CHECKPOINT

    corpus_rows = _fetch_diagnostic_corpus()
    corpus_ids = [str(row["music_id"]) for row in corpus_rows]
    corpus_matrix = _normalise_matrix([row["embedding"] for row in corpus_rows])
    gold_items = gold["items"]

    acoustic_items = [
        {
            "music_id": music_id,
            "title": next((item.get("title") or "" for item in gold_items if str(item["music_id"]) == music_id), ""),
            "caption": caption,
        }
        for music_id, caption in ACOUSTIC_CAPTIONS.items()
    ]
    acoustic_music_ids = {item["music_id"] for item in acoustic_items}
    matched_tag_items = [
        {
            "music_id": str(item["music_id"]),
            "title": item.get("title") or "",
            "caption": item["caption"],
        }
        for item in gold_items
        if str(item["music_id"]) in acoustic_music_ids
    ]

    tag_text_eval = _evaluate_text_set(gold_items, corpus_ids, corpus_rows, corpus_matrix)
    acoustic_eval = _evaluate_text_set(acoustic_items, corpus_ids, corpus_rows, corpus_matrix)
    matched_tag_eval = _evaluate_text_set(matched_tag_items, corpus_ids, corpus_rows, corpus_matrix)

    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "git": _git_info(),
        "model": {
            "name": "M2D-CLAP",
            "checkpoint": str(_M2D_CLAP_CHECKPOINT),
        },
        "corpus": {
            "songs_with_m2d2_embedding": len(corpus_rows),
            "unique_music_ids": len(set(corpus_ids)),
            "duplicate_music_id_count": len(corpus_ids) - len(set(corpus_ids)),
            "embedding_dim": len(corpus_rows[0]["embedding"]) if corpus_rows else 0,
        },
        "gold": {
            "path": str(DEFAULT_GOLD),
            "items": len(gold_items),
            "frozen": bool(gold.get("frozen")),
            "caption_source": sorted({item.get("caption_source", "") for item in gold_items}),
        },
        "diagnostics": {
            "audio_to_audio_self_check": _audio_self_check(gold_items, corpus_ids, corpus_matrix),
            "tag_caption_full": {
                "metrics": tag_text_eval["metrics"],
                "modality_gap": tag_text_eval["cosine"],
            },
            "caption_ab_same_15": {
                "tag_caption_metrics": matched_tag_eval["metrics"],
                "acoustic_caption_metrics": acoustic_eval["metrics"],
                "tag_caption_modality_gap": matched_tag_eval["cosine"],
                "acoustic_caption_modality_gap": acoustic_eval["cosine"],
                "acoustic_items": acoustic_eval["per_item"],
            },
            "caption_distinctiveness": _caption_distinctiveness(gold_items),
            "text_side_sanity": _sanity_queries(corpus_ids, corpus_rows, corpus_matrix),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose M2D-CLAP text-to-audio alignment evaluation")
    parser.add_argument("--gold", default=str(DEFAULT_GOLD))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    gold_path = Path(args.gold)
    gold = _load_gold(gold_path, limit=args.limit)
    report = run_diagnostics(gold)
    report["gold"]["path"] = str(gold_path)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(args.output) if args.output else RESULTS_DIR / (
        "alignment_diagnosis_"
        + report["git"]["sha"]
        + "_"
        + datetime.now().strftime("%Y%m%d_%H%M%S")
        + ".json"
    )
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    diagnostics = report["diagnostics"]
    audio_metrics = diagnostics["audio_to_audio_self_check"]["metrics"]
    tag_metrics = diagnostics["tag_caption_full"]["metrics"]
    ab = diagnostics["caption_ab_same_15"]
    distinction = diagnostics["caption_distinctiveness"]
    print("=" * 72)
    print("Text-to-Audio Alignment Diagnosis")
    print("=" * 72)
    print(f"Git: {report['git']['branch']} @ {report['git']['sha']} | dirty={report['git']['dirty']}")
    print(f"Corpus: {report['corpus']['songs_with_m2d2_embedding']} songs | dim={report['corpus']['embedding_dim']}")
    print(
        "Audio->audio self-check: "
        f"R@1={audio_metrics['recall_at_1']:.3f} "
        f"R@10={audio_metrics['recall_at_10']:.3f} "
        f"MRR={audio_metrics['mrr']:.3f} "
        f"misses={audio_metrics['misses']}"
    )
    print(
        "Tag captions full: "
        f"R@1={tag_metrics['recall_at_1']:.3f} "
        f"R@10={tag_metrics['recall_at_10']:.3f} "
        f"MRR={tag_metrics['mrr']:.3f}"
    )
    print(
        "A/B same 15: "
        f"tag R@10={ab['tag_caption_metrics']['recall_at_10']:.3f}, "
        f"acoustic R@10={ab['acoustic_caption_metrics']['recall_at_10']:.3f}; "
        f"tag MRR={ab['tag_caption_metrics']['mrr']:.3f}, "
        f"acoustic MRR={ab['acoustic_caption_metrics']['mrr']:.3f}"
    )
    print(
        "Modality gap full: "
        f"matched={diagnostics['tag_caption_full']['modality_gap']['matched']['mean']:.4f}, "
        f"random={diagnostics['tag_caption_full']['modality_gap']['random']['mean']:.4f}, "
        f"gap={diagnostics['tag_caption_full']['modality_gap']['mean_gap']:.4f}"
    )
    print(
        "Caption distinctiveness: "
        f"exact_dupes={distinction['exact_duplicate_count']}, "
        f"near_pairs={distinction['near_duplicate_pair_count_jaccard_0_72']}, "
        f"nearest_jaccard_p50={distinction['nearest_neighbor_jaccard']['p50']:.3f}"
    )
    print(f"Report: {out}")


if __name__ == "__main__":
    main()
