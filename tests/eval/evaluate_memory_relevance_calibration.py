"""Calibrate open memory relevance scores without loading a model by default."""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import math
import sys
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_FIXTURE = Path(__file__).parent / "fixtures" / "memory_relevance_calibration.json"
LEVELS = ("L1", "L2")
DEFAULT_TOP_KS = (1, 2, 3)
DEFAULT_MARGINS = (0.0, 0.03, 0.05, 0.10)
REQUIRED_EXAMPLE_FIELDS = {
    "id",
    "group_id",
    "level",
    "language",
    "category",
    "query",
    "memory",
    "label",
    "tags",
}


def _json_rows(path: Path, keys: Sequence[str]) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Input file is empty: {path}")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        try:
            payload = [json.loads(line) for line in text.splitlines() if line.strip()]
        except json.JSONDecodeError as exc:
            raise ValueError(f"Input must be JSON or JSONL: {path}") from exc

    if isinstance(payload, dict):
        for key in keys:
            if key in payload:
                payload = payload[key]
                break
    if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
        raise ValueError(f"Expected a list of row objects in {path}")
    return [dict(row) for row in payload]


def validate_examples(examples: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Validate the open fixture schema and return detached row dictionaries."""

    rows = [dict(example) for example in examples]
    if not rows:
        raise ValueError("Calibration examples must not be empty")

    seen_ids: set[str] = set()
    group_signatures: dict[str, tuple[str, str, str, str]] = {}
    for row in rows:
        missing = REQUIRED_EXAMPLE_FIELDS - row.keys()
        if missing:
            raise ValueError(f"Example is missing fields {sorted(missing)}: {row.get('id', '<unknown>')}")

        example_id = row["id"]
        group_id = row["group_id"]
        if not isinstance(example_id, str) or not example_id.strip():
            raise ValueError("Every example must have a non-empty string id")
        if example_id in seen_ids:
            raise ValueError(f"Duplicate example id: {example_id}")
        seen_ids.add(example_id)
        if not isinstance(group_id, str) or not group_id.strip():
            raise ValueError(f"Example {example_id} must have a non-empty group_id")
        if row["level"] not in LEVELS:
            raise ValueError(f"Example {example_id} has unsupported level: {row['level']}")
        if row["label"] not in (0, 1) or isinstance(row["label"], bool):
            raise ValueError(f"Example {example_id} label must be integer 0 or 1")
        for field in ("language", "category", "query", "memory"):
            if not isinstance(row[field], str) or not row[field].strip():
                raise ValueError(f"Example {example_id} field {field} must be a non-empty string")
        if not isinstance(row["tags"], list) or not all(
            isinstance(tag, str) and tag.strip() for tag in row["tags"]
        ):
            raise ValueError(f"Example {example_id} tags must be a list of non-empty strings")

        signature = (row["level"], row["language"], row["category"], row["query"])
        existing = group_signatures.setdefault(group_id, signature)
        if existing != signature:
            raise ValueError(f"Candidates in group {group_id} must share level, language, category, and query")
    return rows


def load_examples(path: Path = DEFAULT_FIXTURE) -> list[dict[str, Any]]:
    return validate_examples(_json_rows(path, ("examples",)))


def _validated_score(value: Any, *, example_id: str) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Score for {example_id} must be numeric") from exc
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"Score for {example_id} must be finite and in [0, 1]")
    return score


def attach_scores(
    examples: Iterable[Mapping[str, Any]],
    score_rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Merge compact ``{"id": ..., "score": ...}`` rows into examples."""

    validated_examples = validate_examples(examples)
    scores: dict[str, float] = {}
    for raw_row in score_rows:
        row = dict(raw_row)
        example_id = row.get("id")
        if not isinstance(example_id, str) or not example_id:
            raise ValueError("Every score row must contain a non-empty string id")
        if example_id in scores:
            raise ValueError(f"Duplicate score row: {example_id}")
        if "score" not in row:
            raise ValueError(f"Score row is missing score: {example_id}")
        scores[example_id] = _validated_score(row["score"], example_id=example_id)

    expected_ids = {row["id"] for row in validated_examples}
    unknown = scores.keys() - expected_ids
    missing = expected_ids - scores.keys()
    if unknown:
        raise ValueError(f"Scores contain unknown example ids: {sorted(unknown)}")
    if missing:
        raise ValueError(f"Scores are missing example ids: {sorted(missing)}")
    return [{**row, "score": scores[row["id"]]} for row in validated_examples]


def load_score_rows(path: Path) -> list[dict[str, Any]]:
    return _json_rows(path, ("scores", "rows"))


def score_examples(
    examples: Iterable[Mapping[str, Any]],
    scorer: Any,
) -> list[dict[str, Any]]:
    """Score examples using a batch ``.score`` object or a pairwise callable.

    Batch scorers receive ``(query, list_of_memories)``. Pairwise callables
    receive ``(query, memory)``. This function deliberately imports no model.
    """

    validated_examples = validate_examples(examples)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in validated_examples:
        grouped[row["group_id"]].append(row)

    scores: dict[str, float] = {}
    batch_method = getattr(scorer, "score", None)
    if batch_method is None and not callable(scorer):
        raise TypeError("Scorer must expose score(query, documents) or be callable(query, memory)")

    for candidates in grouped.values():
        query = candidates[0]["query"]
        if callable(batch_method):
            raw_scores = batch_method(query, [row["memory"] for row in candidates])
            if isinstance(raw_scores, (str, bytes)):
                raise ValueError("Batch scorer must return one numeric score per memory")
            try:
                candidate_scores = list(raw_scores)
            except TypeError as exc:
                raise ValueError("Batch scorer must return one numeric score per memory") from exc
            if len(candidate_scores) != len(candidates):
                raise ValueError(
                    f"Batch scorer returned {len(candidate_scores)} scores for {len(candidates)} memories"
                )
        else:
            candidate_scores = [scorer(query, row["memory"]) for row in candidates]

        for row, raw_score in zip(candidates, candidate_scores, strict=True):
            scores[row["id"]] = _validated_score(raw_score, example_id=row["id"])
    return [{**row, "score": scores[row["id"]]} for row in validated_examples]


def validate_scored_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    scored_rows = validate_examples(rows)
    for row in scored_rows:
        if "score" not in row:
            raise ValueError(f"Scored row is missing score: {row['id']}")
        row["score"] = _validated_score(row["score"], example_id=row["id"])
    return scored_rows


def classification_metrics(
    rows: Iterable[Mapping[str, Any]],
    selected_ids: Iterable[str],
) -> dict[str, int | float]:
    """Return candidate-level precision, recall, and false-positive metrics."""

    materialized = list(rows)
    selected = set(selected_ids)
    known_ids = {str(row["id"]) for row in materialized}
    unknown = selected - known_ids
    if unknown:
        raise ValueError(f"Selected ids are absent from scored rows: {sorted(unknown)}")

    true_positives = sum(row["id"] in selected and row["label"] == 1 for row in materialized)
    false_positives = sum(row["id"] in selected and row["label"] == 0 for row in materialized)
    false_negatives = sum(row["id"] not in selected and row["label"] == 1 for row in materialized)
    true_negatives = sum(row["id"] not in selected and row["label"] == 0 for row in materialized)
    predicted_positive = true_positives + false_positives
    actual_positive = true_positives + false_negatives
    actual_negative = false_positives + true_negatives
    return {
        "total": len(materialized),
        "predicted_positive": predicted_positive,
        "actual_positive": actual_positive,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "true_negatives": true_negatives,
        "false_negatives": false_negatives,
        "precision": true_positives / predicted_positive if predicted_positive else 1.0,
        "recall": true_positives / actual_positive if actual_positive else 1.0,
        "false_positive_rate": false_positives / actual_negative if actual_negative else 0.0,
    }


def metrics_at_threshold(rows: Iterable[Mapping[str, Any]], threshold: float) -> dict[str, int | float]:
    materialized = list(rows)
    selected = [row["id"] for row in materialized if float(row["score"]) >= threshold]
    return classification_metrics(materialized, selected)


def choose_threshold(
    rows: Iterable[Mapping[str, Any]],
    *,
    target_precision: float = 0.85,
) -> dict[str, Any]:
    """Maximize recall subject to observed precision, then prefer safer ties."""

    materialized = list(rows)
    if not materialized:
        raise ValueError("Cannot calibrate a threshold without scored rows")
    if not 0.0 <= target_precision <= 1.0:
        raise ValueError("target_precision must be in [0, 1]")
    if not any(row["label"] == 1 for row in materialized):
        raise ValueError("Threshold calibration requires at least one positive example")

    candidates: list[tuple[float, dict[str, int | float]]] = []
    for threshold in sorted({float(row["score"]) for row in materialized}, reverse=True):
        metrics = metrics_at_threshold(materialized, threshold)
        if metrics["predicted_positive"]:
            candidates.append((threshold, metrics))

    passing = [item for item in candidates if item[1]["precision"] >= target_precision]
    pool = passing or candidates
    if passing:
        threshold, metrics = max(
            pool,
            key=lambda item: (
                item[1]["recall"],
                item[1]["precision"],
                -item[1]["false_positives"],
                item[0],
            ),
        )
    else:
        threshold, metrics = max(
            pool,
            key=lambda item: (
                item[1]["precision"],
                item[1]["recall"],
                -item[1]["false_positives"],
                item[0],
            ),
        )
    return {
        "threshold": threshold,
        "target_precision": target_precision,
        "target_met": bool(passing),
        "metrics": metrics,
    }


def calibrate_thresholds(
    rows: Iterable[Mapping[str, Any]],
    *,
    target_precision: float = 0.85,
) -> dict[str, dict[str, Any]]:
    """Choose independent L1 and L2 thresholds."""

    materialized = list(rows)
    thresholds: dict[str, dict[str, Any]] = {}
    for level in LEVELS:
        level_rows = [row for row in materialized if row["level"] == level]
        if not level_rows:
            raise ValueError(f"Calibration rows are missing level {level}")
        thresholds[level] = choose_threshold(level_rows, target_precision=target_precision)
    return thresholds


def _threshold_value(thresholds: Mapping[str, Any], level: str) -> float:
    value = thresholds[level]
    if isinstance(value, Mapping):
        value = value["threshold"]
    return float(value)


def select_with_rank_policy(
    rows: Iterable[Mapping[str, Any]],
    thresholds: Mapping[str, Any],
    *,
    top_k: int,
    margin: float,
) -> set[str]:
    """Select thresholded candidates in top-k and close enough to the group best.

    ``margin`` is the maximum permitted score drop from the group's best
    threshold-passing candidate. A smaller value is more conservative.
    """

    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if margin < 0.0:
        raise ValueError("margin must be non-negative")

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["group_id"])].append(row)

    selected: set[str] = set()
    for group_id, candidates in grouped.items():
        levels = {str(row["level"]) for row in candidates}
        if len(levels) != 1:
            raise ValueError(f"Candidates in group {group_id} have mixed levels")
        level = next(iter(levels))
        if level not in thresholds:
            raise ValueError(f"Missing threshold for level {level}")
        threshold = _threshold_value(thresholds, level)
        eligible = sorted(
            (row for row in candidates if float(row["score"]) >= threshold),
            key=lambda row: (-float(row["score"]), str(row["id"])),
        )
        if not eligible:
            continue
        best_score = float(eligible[0]["score"])
        for row in eligible[:top_k]:
            if best_score - float(row["score"]) <= margin + 1e-12:
                selected.add(str(row["id"]))
    return selected


def choose_rank_policy(
    rows: Iterable[Mapping[str, Any]],
    thresholds: Mapping[str, Any],
    *,
    target_precision: float = 0.85,
    top_ks: Sequence[int] = DEFAULT_TOP_KS,
    margins: Sequence[float] = DEFAULT_MARGINS,
) -> dict[str, Any]:
    """Search top-k/margin settings with the same precision-first constraint."""

    materialized = list(rows)
    if not materialized:
        raise ValueError("Cannot calibrate rank policy without scored rows")
    if not top_ks or not margins:
        raise ValueError("top_ks and margins must not be empty")

    candidates: list[dict[str, Any]] = []
    for top_k in sorted(set(int(value) for value in top_ks)):
        for margin in sorted(set(float(value) for value in margins)):
            selected = select_with_rank_policy(
                materialized,
                thresholds,
                top_k=top_k,
                margin=margin,
            )
            metrics = classification_metrics(materialized, selected)
            candidates.append(
                {
                    "top_k": top_k,
                    "margin": margin,
                    "selected_ids": selected,
                    "metrics": metrics,
                }
            )

    passing = [
        item
        for item in candidates
        if item["metrics"]["predicted_positive"]
        and item["metrics"]["precision"] >= target_precision
    ]
    pool = passing or candidates
    if passing:
        best = max(
            pool,
            key=lambda item: (
                item["metrics"]["recall"],
                item["metrics"]["precision"],
                -item["metrics"]["false_positives"],
                -item["top_k"],
                -item["margin"],
            ),
        )
    else:
        best = max(
            pool,
            key=lambda item: (
                item["metrics"]["precision"],
                item["metrics"]["recall"],
                -item["metrics"]["false_positives"],
                -item["top_k"],
                -item["margin"],
            ),
        )

    selected = best["selected_ids"]
    by_level = {}
    for level in LEVELS:
        level_rows = [row for row in materialized if row["level"] == level]
        level_ids = {row["id"] for row in level_rows}
        by_level[level] = classification_metrics(level_rows, selected & level_ids)
    return {
        "top_k": best["top_k"],
        "margin": best["margin"],
        "target_precision": target_precision,
        "target_met": bool(passing),
        "metrics": best["metrics"],
        "by_level": by_level,
    }


def calibrate(
    rows: Iterable[Mapping[str, Any]],
    *,
    target_precision: float = 0.85,
    top_ks: Sequence[int] = DEFAULT_TOP_KS,
    margins: Sequence[float] = DEFAULT_MARGINS,
) -> dict[str, Any]:
    """Build the complete threshold and rank-policy calibration report."""

    scored_rows = validate_scored_rows(rows)
    thresholds = calibrate_thresholds(scored_rows, target_precision=target_precision)
    policy = choose_rank_policy(
        scored_rows,
        thresholds,
        target_precision=target_precision,
        top_ks=top_ks,
        margins=margins,
    )
    threshold_goal_met = all(result["target_met"] for result in thresholds.values())
    return {
        "example_count": len(scored_rows),
        "group_count": len({row["group_id"] for row in scored_rows}),
        "positive_count": sum(row["label"] == 1 for row in scored_rows),
        "target_precision": target_precision,
        "thresholds": thresholds,
        "rank_policy": policy,
        "precision_goal_met": threshold_goal_met and policy["target_met"],
    }


def _metric_line(metrics: Mapping[str, Any]) -> str:
    return (
        f"precision={metrics['precision']:.3f} recall={metrics['recall']:.3f} "
        f"false_positives={metrics['false_positives']} "
        f"false_positive_rate={metrics['false_positive_rate']:.3f} "
        f"selected={metrics['predicted_positive']}"
    )


def print_report(report: Mapping[str, Any]) -> None:
    print("Memory relevance/applicability calibration")
    print(
        f"rows={report['example_count']} groups={report['group_count']} "
        f"positives={report['positive_count']} target_precision={report['target_precision']:.2f}"
    )
    print("\nLevel thresholds (maximum recall at the precision target; safer ties win):")
    for level in LEVELS:
        result = report["thresholds"][level]
        print(
            f"  {level}: threshold={result['threshold']:.6f} "
            f"target_met={str(result['target_met']).lower()} {_metric_line(result['metrics'])}"
        )

    policy = report["rank_policy"]
    print("\nTop-k/margin policy (margin is maximum score drop from group best):")
    print(
        f"  top_k={policy['top_k']} margin={policy['margin']:.6f} "
        f"target_met={str(policy['target_met']).lower()} {_metric_line(policy['metrics'])}"
    )
    for level in LEVELS:
        print(f"  {level} final: {_metric_line(policy['by_level'][level])}")
    print(f"\nPrecision goal met: {str(report['precision_goal_met']).lower()}")


def _load_injected_scorer(spec: str) -> Any:
    try:
        module_name, attribute = spec.rsplit(":", 1)
    except ValueError as exc:
        raise ValueError("Scorer must use module:attribute syntax") from exc
    target = getattr(importlib.import_module(module_name), attribute)
    return target() if inspect.isclass(target) else target


def _parse_int_grid(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated integers") from exc
    if not values or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError("top-k values must be positive integers")
    return values


def _parse_float_grid(value: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated numbers") from exc
    if not values or any(not math.isfinite(item) or item < 0.0 for item in values):
        raise argparse.ArgumentTypeError("margin values must be finite and non-negative")
    return values


def main(argv: Sequence[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--scores", type=Path, help="JSON/JSONL rows containing id and score")
    source.add_argument("--scorer", help="Injected scorer in module:attribute syntax")
    parser.add_argument("--target-precision", type=float, default=0.85)
    parser.add_argument("--top-k-grid", type=_parse_int_grid, default=DEFAULT_TOP_KS)
    parser.add_argument("--margin-grid", type=_parse_float_grid, default=DEFAULT_MARGINS)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    parser.add_argument("--output", type=Path, help="Also write the JSON report to this path")
    args = parser.parse_args(argv)

    examples = load_examples(args.fixture)
    if args.scores:
        rows = attach_scores(examples, load_score_rows(args.scores))
    else:
        rows = score_examples(examples, _load_injected_scorer(args.scorer))
    report = calibrate(
        rows,
        target_precision=args.target_precision,
        top_ks=args.top_k_grid,
        margins=args.margin_grid,
    )
    serialized = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    if args.json:
        print(serialized)
    else:
        print_report(report)
    return 0 if report["precision_goal_met"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
