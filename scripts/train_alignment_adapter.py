"""Train a reversible text-side alignment adapter for dense retrieval.

The adapter maps text embeddings toward the local catalog audio embedding
distribution while leaving stored audio vectors unchanged. Runtime only uses the
adapter when MUSIC_ALIGNMENT_ADAPTER_PATH points to the generated JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_GOLD = REPO_ROOT / "tests" / "eval" / "alignment_gold_captions.json"


def _normalise_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _normalise_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector if norm == 0 else vector / norm


def train_linear_adapter(
    source_vectors: list[list[float]],
    target_vectors: list[list[float]],
    *,
    alpha: float = 1.0,
) -> dict[str, Any]:
    """Fit a ridge linear map from source vectors to target vectors."""
    if not source_vectors or len(source_vectors) != len(target_vectors):
        raise ValueError("source_vectors and target_vectors must have the same non-zero length")
    source = _normalise_rows(np.asarray(source_vectors, dtype=np.float32))
    target = _normalise_rows(np.asarray(target_vectors, dtype=np.float32))
    if source.shape[1] != target.shape[1]:
        raise ValueError("source and target dimensions must match")

    ones = np.ones((source.shape[0], 1), dtype=np.float32)
    design = np.concatenate([source, ones], axis=1)
    regularizer = np.eye(design.shape[1], dtype=np.float32) * float(alpha)
    regularizer[-1, -1] = 0.0
    left = design.T @ design + regularizer
    right = design.T @ target
    try:
        weights = np.linalg.solve(left, right)
    except np.linalg.LinAlgError:
        weights = np.linalg.pinv(left) @ right
    matrix = weights[:-1, :].T
    bias = weights[-1, :]
    return {
        "type": "linear",
        "input_dim": int(source.shape[1]),
        "output_dim": int(target.shape[1]),
        "matrix": matrix.astype(float).tolist(),
        "bias": bias.astype(float).tolist(),
        "normalize": True,
        "alpha": float(alpha),
        "num_pairs": int(source.shape[0]),
    }


def apply_adapter_np(vector: list[float] | np.ndarray, spec: dict[str, Any], *, mix: float = 1.0) -> np.ndarray:
    source = np.asarray(vector, dtype=np.float32)
    matrix = np.asarray(spec["matrix"], dtype=np.float32)
    bias = np.asarray(spec.get("bias", []), dtype=np.float32)
    mapped = matrix @ source
    if bias.size:
        mapped = mapped + bias
    mix = max(0.0, min(1.0, float(mix)))
    if mapped.shape == source.shape and mix < 1.0:
        mapped = (1.0 - mix) * source + mix * mapped
    return _normalise_vector(mapped)


def split_gold_items(items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deterministic 4:1 train/validation split."""
    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        (validation if index % 5 == 0 else train).append(item)
    return train, validation


def _load_gold(path: Path, max_items: int = 0) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = list(payload.get("items") or [])
    if max_items > 0:
        items = items[:max_items]
    return items


def caption_for_item(item: dict[str, Any], caption_style: str = "metadata") -> str:
    """Return the caption used for adapter training.

    `metadata` preserves the original frozen A4.1 caption. `acoustic` builds a
    deterministic MusicCaps-like proxy from the same frozen metadata so adapter
    v2 can learn from audio-text wording closer to timbre, dynamics, and scene.
    """

    if caption_style == "metadata":
        return str(item.get("caption") or "").strip()
    if caption_style != "acoustic":
        raise ValueError(f"unsupported caption_style: {caption_style}")
    explicit = str(item.get("acoustic_caption") or "").strip()
    if explicit:
        return explicit
    from tests.eval.build_alignment_gold import acoustic_caption_from_song

    metadata = dict(item.get("metadata") or {})
    return acoustic_caption_from_song(
        {
            "language": metadata.get("language") or "",
            "region": metadata.get("region") or "",
            "vibe": metadata.get("vibe") or "",
            "genres": metadata.get("genres") or [],
            "moods": metadata.get("moods") or [],
            "themes": metadata.get("themes") or [],
            "scenarios": metadata.get("scenarios") or [],
        }
    )


def _encode_text(backend: str, text: str) -> list[float]:
    if backend == "muq":
        from retrieval.muq_embedder import encode_text_to_muq

        return [float(value) for value in encode_text_to_muq(text)]
    if backend == "m2d":
        from retrieval.audio_embedder import encode_text_to_embedding

        return [float(value) for value in encode_text_to_embedding(text)]
    raise ValueError(f"unsupported backend: {backend}")


def _fetch_corpus_for_backend(backend: str) -> tuple[list[str], np.ndarray]:
    from tests.eval.evaluate_alignment_attribute import _fetch_common_corpus

    _ids, _labels_by_id, corpora = _fetch_common_corpus()
    corpus = corpora[backend]
    return list(corpus["ids"]), np.asarray(corpus["matrix"], dtype=np.float32)


def _pairs_for_items(
    *,
    backend: str,
    items: list[dict[str, Any]],
    ids: list[str],
    matrix: np.ndarray,
    caption_style: str,
) -> tuple[list[list[float]], list[list[float]], list[str]]:
    id_to_index = {music_id: index for index, music_id in enumerate(ids)}
    sources: list[list[float]] = []
    targets: list[list[float]] = []
    skipped: list[str] = []
    for item in items:
        music_id = str(item.get("music_id") or "")
        caption = caption_for_item(item, caption_style)
        index = id_to_index.get(music_id)
        if index is None or not caption:
            skipped.append(music_id)
            continue
        sources.append(_encode_text(backend, caption))
        targets.append([float(value) for value in matrix[index].tolist()])
    return sources, targets, skipped


def _rank_metrics(
    *,
    backend: str,
    items: list[dict[str, Any]],
    ids: list[str],
    matrix: np.ndarray,
    spec: dict[str, Any] | None,
    mix: float,
    caption_style: str,
) -> dict[str, Any]:
    ranks: list[int | None] = []
    for item in items:
        music_id = str(item.get("music_id") or "")
        caption = caption_for_item(item, caption_style)
        if not caption or music_id not in ids:
            ranks.append(None)
            continue
        vector = np.asarray(_encode_text(backend, caption), dtype=np.float32)
        query = apply_adapter_np(vector, spec, mix=mix) if spec else _normalise_vector(vector)
        scores = matrix @ query
        order = np.argsort(-scores, kind="mergesort")
        ranked_ids = [ids[int(index)] for index in order]
        ranks.append(ranked_ids.index(music_id) + 1 if music_id in ranked_ids else None)

    valid = [rank for rank in ranks if rank is not None]

    def recall_at(k: int) -> float:
        return float(sum(1 for rank in valid if rank <= k) / len(valid)) if valid else 0.0

    return {
        "count": len(valid),
        "recall_at_1": recall_at(1),
        "recall_at_5": recall_at(5),
        "recall_at_10": recall_at(10),
        "mrr": float(sum(1.0 / rank for rank in valid) / len(valid)) if valid else 0.0,
    }


def _choose_mix(
    *,
    backend: str,
    validation_items: list[dict[str, Any]],
    ids: list[str],
    matrix: np.ndarray,
    spec: dict[str, Any],
    candidates: list[float],
    caption_style: str,
) -> tuple[float, list[dict[str, Any]]]:
    reports = []
    for mix in candidates:
        metrics = _rank_metrics(
            backend=backend,
            items=validation_items,
            ids=ids,
            matrix=matrix,
            spec=spec if mix > 0 else None,
            mix=mix,
            caption_style=caption_style,
        )
        reports.append({"mix": float(mix), **metrics})
    best = max(reports, key=lambda row: (row["mrr"], row["recall_at_10"], -row["mix"]))
    return float(best["mix"]), reports


def _git_info() -> dict[str, Any]:
    from tests.eval.evaluate_alignment_attribute import _git_info as git_info

    return git_info()


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a local text-side alignment adapter")
    parser.add_argument("--backend", choices=["muq", "m2d"], default="muq")
    parser.add_argument("--gold", default=str(DEFAULT_GOLD), help="Frozen caption gold JSON")
    parser.add_argument(
        "--caption-style",
        choices=["metadata", "acoustic"],
        default="metadata",
        help="Caption wording used for adapter training.",
    )
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=2.0, help="Ridge regularization")
    parser.add_argument(
        "--mix-candidates",
        default="0,0.25,0.5,0.75,1.0",
        help="Comma-separated adapter mix values validated on held-out captions",
    )
    parser.add_argument("--output", default="data/alignment_adapter.json")
    parser.add_argument("--report-output", default="")
    parser.add_argument("--skip-attribute-eval", action="store_true")
    args = parser.parse_args()

    gold_path = Path(args.gold)
    if not gold_path.is_absolute():
        gold_path = REPO_ROOT / gold_path
    items = _load_gold(gold_path, args.max_items)
    train_items, validation_items = split_gold_items(items)
    ids, matrix = _fetch_corpus_for_backend(args.backend)
    sources, targets, skipped = _pairs_for_items(
        backend=args.backend,
        items=train_items,
        ids=ids,
        matrix=matrix,
        caption_style=args.caption_style,
    )
    spec = train_linear_adapter(sources, targets, alpha=args.alpha)
    mix_candidates = [float(value.strip()) for value in args.mix_candidates.split(",") if value.strip()]
    best_mix, mix_reports = _choose_mix(
        backend=args.backend,
        validation_items=validation_items,
        ids=ids,
        matrix=matrix,
        spec=spec,
        candidates=mix_candidates,
        caption_style=args.caption_style,
    )
    spec["mix"] = best_mix

    payload: dict[str, Any] = {
        "schema_version": 1,
        "method": "caption_ridge_linear_adapter_v1",
        "caption_style": args.caption_style,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "git": _git_info(),
        "gold_path": str(gold_path),
        "split": {
            "train_count": len(train_items),
            "validation_count": len(validation_items),
            "skipped_train_ids": skipped,
        },
        "backends": {args.backend: spec},
    }
    output = Path(args.output)
    if not output.is_absolute():
        output = REPO_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    before_train = _rank_metrics(
        backend=args.backend,
        items=train_items,
        ids=ids,
        matrix=matrix,
        spec=None,
        mix=0,
        caption_style=args.caption_style,
    )
    after_train = _rank_metrics(
        backend=args.backend,
        items=train_items,
        ids=ids,
        matrix=matrix,
        spec=spec if best_mix > 0 else None,
        mix=best_mix,
        caption_style=args.caption_style,
    )
    before_val = _rank_metrics(
        backend=args.backend,
        items=validation_items,
        ids=ids,
        matrix=matrix,
        spec=None,
        mix=0,
        caption_style=args.caption_style,
    )
    after_val = _rank_metrics(
        backend=args.backend,
        items=validation_items,
        ids=ids,
        matrix=matrix,
        spec=spec if best_mix > 0 else None,
        mix=best_mix,
        caption_style=args.caption_style,
    )

    report: dict[str, Any] = {
        "adapter_path": str(output),
        "backend": args.backend,
        "caption_style": args.caption_style,
        "best_mix": best_mix,
        "mix_search": mix_reports,
        "exact_caption_retrieval": {
            "train_before": before_train,
            "train_after": after_train,
            "validation_before": before_val,
            "validation_after": after_val,
        },
    }
    if not args.skip_attribute_eval:
        from tests.eval.evaluate_alignment_attribute import evaluate_attribute_alignment

        report["attribute_eval"] = evaluate_attribute_alignment(k=10, adapter_path=str(output))

    if args.report_output:
        report_path = Path(args.report_output)
        if not report_path.is_absolute():
            report_path = REPO_ROOT / report_path
    else:
        report_path = REPO_ROOT / "tests" / "eval" / "results" / (
            "alignment_adapter_train_"
            + payload["git"]["sha"]
            + "_"
            + datetime.now().strftime("%Y%m%d_%H%M%S")
            + ".json"
        )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("=" * 72)
    print("Alignment Adapter Training")
    print("=" * 72)
    print(f"Adapter: {output}")
    print(f"Report: {report_path}")
    print(
        f"Backend: {args.backend} | caption_style={args.caption_style} | "
        f"train={len(train_items)} validation={len(validation_items)}"
    )
    print(f"Best mix: {best_mix:.2f} | pairs={spec['num_pairs']} | skipped_train={len(skipped)}")
    print(
        "Validation exact caption R@10: "
        f"{before_val['recall_at_10']:.3f}->{after_val['recall_at_10']:.3f} | "
        f"MRR {before_val['mrr']:.3f}->{after_val['mrr']:.3f}"
    )
    if "attribute_eval" in report:
        metrics = report["attribute_eval"]["metrics"]["overall"]
        print("Attribute P@10: " + " | ".join(f"{key.upper()}={value:.3f}" for key, value in metrics.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
