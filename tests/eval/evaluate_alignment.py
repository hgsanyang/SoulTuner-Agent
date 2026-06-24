"""Evaluate frozen text-caption to audio-vector alignment.

This harness isolates M2D-CLAP text->audio retrieval from the end-to-end Agent.
It does not call the Planner, HyDE, or any LLM during evaluation.
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

from tests.eval.alignment_metrics import first_relevant_rank, summarize_alignment_ranks

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GOLD = REPO_ROOT / "tests" / "eval" / "alignment_gold_captions.json"
RESULTS_DIR = REPO_ROOT / "tests" / "eval" / "results"


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


def _load_gold(path: Path, limit: int | None = None) -> dict[str, Any]:
    gold = json.loads(path.read_text(encoding="utf-8"))
    items = gold.get("items") or []
    if limit is not None:
        items = items[:limit]
    return {**gold, "items": items}


def _fetch_audio_corpus() -> list[dict[str, Any]]:
    _load_env_file(REPO_ROOT / ".env")
    from neo4j import GraphDatabase

    uri = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")
    query = """
    MATCH (s:Song)
    WHERE s.m2d2_embedding IS NOT NULL AND s.music_id IS NOT NULL
    OPTIONAL MATCH (s)-[:PERFORMED_BY]->(a:Artist)
    RETURN s.music_id AS music_id,
           elementId(s) AS element_id,
           s.title AS title,
           collect(DISTINCT a.name) AS artists,
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


def _rank_caption(
    caption: str,
    corpus_ids: list[str],
    corpus_matrix: np.ndarray,
) -> list[str]:
    from retrieval.audio_embedder import encode_text_to_embedding

    query = np.asarray(encode_text_to_embedding(caption), dtype=np.float32)
    norm = np.linalg.norm(query)
    if norm > 0:
        query = query / norm
    scores = corpus_matrix @ query
    order = np.argsort(-scores, kind="mergesort")
    return [corpus_ids[int(index)] for index in order]


def evaluate_alignment(gold: dict[str, Any]) -> dict[str, Any]:
    from retrieval.audio_embedder import _M2D_CLAP_CHECKPOINT

    corpus_rows = _fetch_audio_corpus()
    corpus_ids = [str(row["music_id"]) for row in corpus_rows]
    corpus_matrix = _normalise_matrix([row["embedding"] for row in corpus_rows])
    corpus_titles = {
        str(row["music_id"]): {
            "title": row.get("title") or "",
            "artists": [str(a) for a in (row.get("artists") or []) if a],
        }
        for row in corpus_rows
    }

    per_item = []
    ranks: list[int | None] = []
    for item in gold["items"]:
        gold_id = str(item["music_id"])
        ranked_ids = _rank_caption(item["caption"], corpus_ids, corpus_matrix)
        rank = first_relevant_rank(ranked_ids, [gold_id])
        ranks.append(rank)
        top10 = ranked_ids[:10]
        per_item.append(
            {
                "music_id": gold_id,
                "title": item.get("title") or "",
                "caption": item["caption"],
                "rank": rank,
                "hit_at_1": rank is not None and rank <= 1,
                "hit_at_5": rank is not None and rank <= 5,
                "hit_at_10": rank is not None and rank <= 10,
                "top10": [
                    {
                        "music_id": candidate_id,
                        **corpus_titles.get(candidate_id, {}),
                    }
                    for candidate_id in top10
                ],
            }
        )

    metrics = summarize_alignment_ranks(ranks, ks=(1, 5, 10))
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
        },
        "gold": {
            "path": str(DEFAULT_GOLD),
            "schema_version": gold.get("schema_version"),
            "items": len(gold["items"]),
            "frozen": bool(gold.get("frozen")),
            "caption_source": sorted({item.get("caption_source", "") for item in gold["items"]}),
        },
        "metrics": metrics,
        "per_item": per_item,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate M2D-CLAP text-to-audio alignment on frozen captions")
    parser.add_argument("--gold", default=str(DEFAULT_GOLD))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    gold_path = Path(args.gold)
    gold = _load_gold(gold_path, limit=args.limit)
    report = evaluate_alignment(gold)
    report["gold"]["path"] = str(gold_path)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(args.output) if args.output else RESULTS_DIR / (
        "alignment_eval_"
        + report["git"]["sha"]
        + "_"
        + datetime.now().strftime("%Y%m%d_%H%M%S")
        + ".json"
    )
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if not args.quiet:
        metrics = report["metrics"]
        print("=" * 64)
        print("Text-to-Audio Alignment Eval")
        print("=" * 64)
        print(f"Git: {report['git']['branch']} @ {report['git']['sha']} | dirty={report['git']['dirty']}")
        print(f"Model: {report['model']['name']} | checkpoint={report['model']['checkpoint']}")
        print(f"Corpus songs with m2d2_embedding: {report['corpus']['songs_with_m2d2_embedding']}")
        print(f"Gold captions: {report['gold']['items']} | frozen={report['gold']['frozen']}")
        print(
            "Metrics: "
            f"R@1={metrics['recall_at_1']:.3f} "
            f"R@5={metrics['recall_at_5']:.3f} "
            f"R@10={metrics['recall_at_10']:.3f} "
            f"MRR={metrics['mrr']:.3f} "
            f"misses={metrics['misses']}"
        )
        print(f"Report: {out}")


if __name__ == "__main__":
    main()
