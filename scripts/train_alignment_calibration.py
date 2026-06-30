"""Train a lightweight text-audio gap calibration for dense retrieval.

This script intentionally trains a transparent bias calibration, not a neural
adapter.  It uses frozen attribute queries, splits them into train/validation,
and writes a JSON file that production only consumes when
MUSIC_ALIGNMENT_CALIBRATION_PATH is explicitly set.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _norm(vector: list[float]) -> float:
    return math.sqrt(sum(float(value) * float(value) for value in vector))


def _normalize(vector: list[float]) -> list[float]:
    norm = _norm(vector)
    return [float(value) / norm for value in vector] if norm > 0 else vector


def split_calibration_queries(queries: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deterministic 2:1 train/validation split, preserving mixed attributes."""
    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    for index, query in enumerate(queries):
        (validation if index % 3 == 0 else train).append(query)
    return train, validation


def _apply_spec(vector: list[float], spec: dict[str, Any]) -> list[float]:
    bias = spec.get("bias")
    if not isinstance(bias, list) or len(bias) != len(vector):
        return vector
    scale = float(spec.get("scale", 1.0))
    adjusted = [float(value) * scale + float(delta) for value, delta in zip(vector, bias)]
    return _normalize(adjusted) if spec.get("normalize", True) else adjusted


def _scaled_spec(spec: dict[str, Any], shrink: float) -> dict[str, Any]:
    adjusted = dict(spec)
    bias = spec.get("bias")
    adjusted["bias"] = [float(value) * float(shrink) for value in bias] if isinstance(bias, list) else []
    adjusted["shrink"] = float(shrink)
    return adjusted


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _centroid_for_target(
    ids: list[str],
    labels_by_id: dict[str, dict[str, Any]],
    matrix: Any,
    target: dict[str, Any],
) -> list[float] | None:
    import numpy as np
    from tests.eval.evaluate_alignment_attribute import label_matches

    indexes = [
        index
        for index, music_id in enumerate(ids)
        if label_matches(labels_by_id.get(music_id, {}), target)
    ]
    if not indexes:
        return None
    centroid = np.asarray(matrix[indexes], dtype=np.float32).mean(axis=0)
    norm = np.linalg.norm(centroid)
    if norm > 0:
        centroid = centroid / norm
    return [float(value) for value in centroid.tolist()]


def _encode_text(backend: str, text: str) -> list[float]:
    if backend == "muq":
        from retrieval.muq_embedder import encode_text_to_muq

        return [float(value) for value in encode_text_to_muq(text)]
    if backend == "m2d":
        from retrieval.audio_embedder import encode_text_to_embedding

        return [float(value) for value in encode_text_to_embedding(text)]
    raise ValueError(f"unsupported backend: {backend}")


def _evaluate_backend(
    *,
    backend: str,
    queries: list[dict[str, Any]],
    ids: list[str],
    labels_by_id: dict[str, dict[str, Any]],
    matrix: Any,
    spec: dict[str, Any] | None,
    k: int,
) -> dict[str, Any]:
    from tests.eval.evaluate_alignment_attribute import _rank_ids, precision_at_k

    rows = []
    for query in queries:
        vector = _encode_text(backend, query["text"])
        ranked = _rank_ids(_apply_spec(vector, spec or {}), matrix, ids)
        rows.append({
            "id": query["id"],
            "query_language": query["query_language"],
            "precision_at_k": precision_at_k(ranked, labels_by_id, query["target"], k),
        })
    return {
        "count": len(rows),
        "mean_precision_at_k": _mean([row["precision_at_k"] for row in rows]),
        "by_query_language": {
            lang: _mean([row["precision_at_k"] for row in rows if row["query_language"] == lang])
            for lang in sorted({row["query_language"] for row in rows})
        },
        "rows": rows,
    }


def train_backend_calibration(
    *,
    backend: str,
    train_queries: list[dict[str, Any]],
    validation_queries: list[dict[str, Any]],
    ids: list[str],
    labels_by_id: dict[str, dict[str, Any]],
    matrix: Any,
    k: int,
) -> dict[str, Any]:
    from retrieval.alignment_calibration import build_bias_calibration

    pairs: list[tuple[list[float], list[float]]] = []
    skipped = []
    for query in train_queries:
        centroid = _centroid_for_target(ids, labels_by_id, matrix, query["target"])
        if centroid is None:
            skipped.append(query["id"])
            continue
        pairs.append((_normalize(_encode_text(backend, query["text"])), centroid))

    base_spec = build_bias_calibration(pairs, scale=1.0, normalize=True)
    shrink_candidates = [0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]
    candidate_reports = []
    for shrink in shrink_candidates:
        candidate_spec = _scaled_spec(base_spec, shrink)
        score = _evaluate_backend(
            backend=backend,
            queries=train_queries,
            ids=ids,
            labels_by_id=labels_by_id,
            matrix=matrix,
            spec=candidate_spec,
            k=k,
        )["mean_precision_at_k"]
        candidate_reports.append({"shrink": shrink, "train_precision_at_k": score})
    best = max(candidate_reports, key=lambda row: (row["train_precision_at_k"], -row["shrink"]))
    spec = _scaled_spec(base_spec, float(best["shrink"]))
    return {
        "backend": backend,
        "spec": spec,
        "shrink_search": candidate_reports,
        "skipped_train_queries": skipped,
        "train": {
            "before": _evaluate_backend(
                backend=backend,
                queries=train_queries,
                ids=ids,
                labels_by_id=labels_by_id,
                matrix=matrix,
                spec=None,
                k=k,
            ),
            "after": _evaluate_backend(
                backend=backend,
                queries=train_queries,
                ids=ids,
                labels_by_id=labels_by_id,
                matrix=matrix,
                spec=spec,
                k=k,
            ),
        },
        "validation": {
            "before": _evaluate_backend(
                backend=backend,
                queries=validation_queries,
                ids=ids,
                labels_by_id=labels_by_id,
                matrix=matrix,
                spec=None,
                k=k,
            ),
            "after": _evaluate_backend(
                backend=backend,
                queries=validation_queries,
                ids=ids,
                labels_by_id=labels_by_id,
                matrix=matrix,
                spec=spec,
                k=k,
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train auditable alignment calibration JSON")
    parser.add_argument("--backend", choices=["muq", "m2d", "both"], default="both")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--output",
        default="data/alignment_calibration.json",
        help="Calibration JSON path. data/ is gitignored for local artifacts.",
    )
    parser.add_argument("--report-output", default="", help="Optional full training report JSON")
    args = parser.parse_args()

    from tests.eval.evaluate_alignment_attribute import (
        FROZEN_ATTRIBUTE_QUERIES,
        _fetch_common_corpus,
        _git_info,
    )

    ids, labels_by_id, m2d_matrix, muq_matrix = _fetch_common_corpus()
    train_queries, validation_queries = split_calibration_queries(FROZEN_ATTRIBUTE_QUERIES)
    backends = ["muq", "m2d"] if args.backend == "both" else [args.backend]

    backend_reports = {}
    calibration_payload: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "method": "attribute_centroid_bias_v1",
        "k": args.k,
        "git": _git_info(),
        "corpus": {"songs": len(ids)},
        "split": {
            "train_ids": [query["id"] for query in train_queries],
            "validation_ids": [query["id"] for query in validation_queries],
        },
    }
    for backend in backends:
        matrix = muq_matrix if backend == "muq" else m2d_matrix
        report = train_backend_calibration(
            backend=backend,
            train_queries=train_queries,
            validation_queries=validation_queries,
            ids=ids,
            labels_by_id=labels_by_id,
            matrix=matrix,
            k=args.k,
        )
        calibration_payload[backend] = report["spec"]
        backend_reports[backend] = report

    output = REPO_ROOT / args.output if not Path(args.output).is_absolute() else Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(calibration_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    full_report = {
        **{key: calibration_payload[key] for key in ("created_at", "method", "k", "git", "corpus", "split")},
        "calibration_path": str(output),
        "backends": backend_reports,
    }
    if args.report_output:
        report_path = REPO_ROOT / args.report_output if not Path(args.report_output).is_absolute() else Path(args.report_output)
    else:
        report_path = REPO_ROOT / "tests" / "eval" / "results" / (
            "alignment_calibration_train_"
            + calibration_payload["git"]["sha"]
            + "_"
            + datetime.now().strftime("%Y%m%d_%H%M%S")
            + ".json"
        )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(full_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("=" * 72)
    print("Alignment Calibration Training")
    print("=" * 72)
    print(f"Calibration: {output}")
    print(f"Report: {report_path}")
    print(f"Corpus songs: {len(ids)} | train={len(train_queries)} validation={len(validation_queries)} | P@{args.k}")
    for backend, report in backend_reports.items():
        train_before = report["train"]["before"]["mean_precision_at_k"]
        train_after = report["train"]["after"]["mean_precision_at_k"]
        val_before = report["validation"]["before"]["mean_precision_at_k"]
        val_after = report["validation"]["after"]["mean_precision_at_k"]
        print(
            f"{backend}: train {train_before:.3f}->{train_after:.3f} | "
            f"validation {val_before:.3f}->{val_after:.3f} | pairs={report['spec']['num_pairs']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
