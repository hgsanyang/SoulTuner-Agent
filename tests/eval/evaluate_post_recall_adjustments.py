"""Evaluate post-recall score adjustments.

This script is intentionally independent from LLM planning.  It answers two
questions:

1. Safety: can the bounded delta dominate content relevance?
2. Direction: do freshness/long-tail/exposure signals move candidates in the
   expected direction?

When Neo4j is available it also samples real catalog metadata and reports the
delta distribution on actual songs.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
from typing import Any

from retrieval.post_recall_adjustments import (
    DAY_MS,
    DEFAULT_CONFIG,
    apply_post_recall_adjustments,
)


NOW_MS = 1_800_000_000_000


def _candidate(title: str, score: float, affinity: float = 0.0) -> dict[str, Any]:
    return {
        "song": {"title": title, "artist": "artist"},
        "similarity_score": score,
        "_rrf_score": score,
        "_graph_affinity": affinity,
    }


def _rank(items: list[dict[str, Any]], score_field: str) -> list[str]:
    return [
        item["song"]["title"]
        for item in sorted(items, key=lambda item: item.get(score_field, 0), reverse=True)
    ]


def _pairwise_flips(
    before: list[dict[str, Any]],
    after: list[dict[str, Any]],
    *,
    strong_gap: float,
) -> dict[str, Any]:
    before_score = {item["song"]["title"]: float(item["similarity_score"]) for item in before}
    after_score = {item["song"]["title"]: float(item["similarity_score"]) for item in after}
    titles = list(before_score)
    strong_flips = []
    near_tie_flips = []
    for i, left in enumerate(titles):
        for right in titles[i + 1 :]:
            before_cmp = before_score[left] - before_score[right]
            after_cmp = after_score[left] - after_score[right]
            if before_cmp == 0 or after_cmp == 0 or before_cmp * after_cmp > 0:
                continue
            event = {
                "pair": [left, right],
                "base_gap": round(abs(before_cmp), 6),
                "after_gap": round(abs(after_cmp), 6),
            }
            if abs(before_cmp) > strong_gap:
                strong_flips.append(event)
            else:
                near_tie_flips.append(event)
    return {
        "strong_gap_threshold": strong_gap,
        "strong_gap_flips": strong_flips,
        "near_tie_flips": near_tie_flips,
    }


def evaluate_synthetic() -> dict[str, Any]:
    base = [
        _candidate("content best", 0.90, affinity=0.0),
        _candidate("recent fresh", 0.86, affinity=0.2),
        _candidate("overexposed near tie", 0.855, affinity=0.4),
        _candidate("personal fit", 0.84, affinity=1.0),
        _candidate("longtail discovery", 0.835, affinity=0.5),
        _candidate("weak fresh", 0.62, affinity=1.0),
    ]
    metadata = {
        "content best": {"updated_at": NOW_MS - 120 * DAY_MS, "ts_beta": 1.3, "ts_last_exposed_at": NOW_MS - 30 * DAY_MS},
        "recent fresh": {"updated_at": NOW_MS - DAY_MS, "ts_beta": 1.0},
        "overexposed near tie": {"updated_at": NOW_MS - 30 * DAY_MS, "ts_beta": 18.0, "ts_last_exposed_at": NOW_MS},
        "personal fit": {"updated_at": NOW_MS - 60 * DAY_MS, "ts_beta": 2.0, "ts_last_exposed_at": NOW_MS - 20 * DAY_MS},
        "longtail discovery": {"updated_at": NOW_MS - 40 * DAY_MS, "ts_beta": 1.0},
        "weak fresh": {"updated_at": NOW_MS, "ts_beta": 1.0},
    }
    adjusted = apply_post_recall_adjustments(
        [dict(item, song=dict(item["song"])) for item in base],
        metadata_by_title=metadata,
        apply_to_similarity=True,
        now_ms=NOW_MS,
    )
    by_title = {item["song"]["title"]: item for item in adjusted}
    deltas = [float(item["_post_recall_delta"]) for item in adjusted]
    strong_gap = 2 * DEFAULT_CONFIG.delta_limit
    return {
        "config": DEFAULT_CONFIG.__dict__,
        "before_rank": _rank(base, "similarity_score"),
        "after_rank": _rank(adjusted, "similarity_score"),
        "max_abs_delta": round(max(abs(delta) for delta in deltas), 6),
        "safety": _pairwise_flips(base, adjusted, strong_gap=strong_gap),
        "direction_checks": {
            "fresh_positive": by_title["recent fresh"]["_post_recall_delta"] > 0,
            "longtail_positive": by_title["longtail discovery"]["_post_recall_delta"] > 0,
            "overexposed_negative": by_title["overexposed near tie"]["_post_recall_delta"] < 0,
            "weak_fresh_does_not_overtake_content_best": (
                by_title["weak fresh"]["similarity_score"]
                < by_title["content best"]["similarity_score"]
            ),
        },
        "items": [
            {
                "title": item["song"]["title"],
                "base": next(base_item["similarity_score"] for base_item in base if base_item["song"]["title"] == item["song"]["title"]),
                "adjusted": item["similarity_score"],
                "delta": item["_post_recall_delta"],
                "personal": item["_post_personal_score"],
                "freshness": item["_post_freshness_score"],
                "longtail": item["_post_longtail_score"],
                "exposure_penalty": item["_post_exposure_penalty"],
                "effective_exposure": item["_post_effective_exposure"],
            }
            for item in adjusted
        ],
    }


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "min": 0, "max": 0, "mean": 0}
    return {
        "count": len(values),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
        "mean": round(sum(values) / len(values), 6),
    }


def evaluate_live_catalog(limit: int) -> dict[str, Any]:
    from retrieval.neo4j_client import get_neo4j_client

    query = """
    MATCH (s:Song)
    WITH s, properties(s) AS props
    RETURN s.title AS title,
           coalesce(s.artist, '') AS artist,
           coalesce(s.updated_at, 0) AS updated_at,
           coalesce(s.ts_alpha, 1) AS ts_alpha,
           coalesce(s.ts_beta, 1) AS ts_beta,
           coalesce(props['ts_last_exposed_at'], 0) AS ts_last_exposed_at
    ORDER BY coalesce(s.updated_at, 0) DESC, s.title ASC
    LIMIT $limit
    """
    rows = get_neo4j_client().execute_query(query, {"limit": limit}) or []
    candidates = []
    metadata = {}
    for index, row in enumerate(rows):
        title = str(row.get("title") or "")
        if not title:
            continue
        # Deterministic descending base scores simulate an already content-ranked
        # candidate list.  The experiment measures the bounded adjustment shape.
        base_score = 0.95 - 0.35 * (index / max(len(rows) - 1, 1))
        candidates.append(
            _candidate(
                title,
                round(base_score, 6),
                affinity=0.0,
            )
        )
        metadata[title] = dict(row)

    adjusted = apply_post_recall_adjustments(
        candidates,
        metadata_by_title=metadata,
        apply_to_similarity=True,
    )
    deltas = [float(item["_post_recall_delta"]) for item in adjusted]
    exposures = [float(item["_post_effective_exposure"]) for item in adjusted]
    freshness = [float(item["_post_freshness_score"]) for item in adjusted]
    penalties = [float(item["_post_exposure_penalty"]) for item in adjusted]
    positives = sum(1 for delta in deltas if delta > 0)
    negatives = sum(1 for delta in deltas if delta < 0)
    neutral = len(deltas) - positives - negatives
    return {
        "sample_size": len(adjusted),
        "delta": _stats(deltas),
        "effective_exposure": _stats(exposures),
        "freshness": _stats(freshness),
        "exposure_penalty": _stats(penalties),
        "delta_direction_counts": {
            "positive": positives,
            "negative": negatives,
            "neutral": neutral,
        },
        "top_positive_delta": [
            {
                "title": item["song"]["title"],
                "delta": item["_post_recall_delta"],
                "freshness": item["_post_freshness_score"],
                "longtail": item["_post_longtail_score"],
                "exposure_penalty": item["_post_exposure_penalty"],
            }
            for item in sorted(adjusted, key=lambda item: item["_post_recall_delta"], reverse=True)[:5]
        ],
        "top_negative_delta": [
            {
                "title": item["song"]["title"],
                "delta": item["_post_recall_delta"],
                "freshness": item["_post_freshness_score"],
                "longtail": item["_post_longtail_score"],
                "exposure_penalty": item["_post_exposure_penalty"],
            }
            for item in sorted(adjusted, key=lambda item: item["_post_recall_delta"])[:5]
        ],
    }


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate post-recall adjustments")
    parser.add_argument("--live", action="store_true", help="Also sample Neo4j catalog metadata")
    parser.add_argument("--limit", type=int, default=200, help="Live catalog sample size")
    parser.add_argument("--output-dir", default="tests/eval/results")
    args = parser.parse_args()

    report: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "synthetic": evaluate_synthetic(),
    }
    if args.live:
        try:
            report["live_catalog"] = evaluate_live_catalog(args.limit)
        except Exception as exc:
            report["live_catalog_error"] = f"{type(exc).__name__}: {exc}"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"post_recall_adjustments_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    synthetic = report["synthetic"]
    print("Post-recall Adjustment Eval")
    print(f"git_sha: {report['git_sha']}")
    print(f"max_abs_delta: {synthetic['max_abs_delta']:.6f}")
    print(f"before_rank: {synthetic['before_rank']}")
    print(f"after_rank:  {synthetic['after_rank']}")
    print(f"strong_gap_flips: {len(synthetic['safety']['strong_gap_flips'])}")
    print(f"near_tie_flips: {len(synthetic['safety']['near_tie_flips'])}")
    print(f"direction_checks: {synthetic['direction_checks']}")
    if "live_catalog" in report:
        live = report["live_catalog"]
        print(f"live_sample_size: {live['sample_size']}")
        print(f"live_delta: {live['delta']}")
        print(f"live_direction_counts: {live['delta_direction_counts']}")
    elif "live_catalog_error" in report:
        print(f"live_catalog_error: {report['live_catalog_error']}")
    print(f"report: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
