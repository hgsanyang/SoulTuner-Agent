"""Evaluate the memory injection / silence decision (memory v3 MV3-5).

"When Should Memory Stay Silent" (arXiv 2606.06055) treats memory use as a
decision, not just a ranking: it matters as much when memory should be
withheld as when it should be injected. This evaluator scores the retriever's
inject-vs-silent decision against a fixture of (query, candidate memories,
gold should-inject subset), isolating the DECISION logic from BGE quality by
feeding each memory's simulated relevance through an injected scorer.

Usage:
    python -m tests.eval.evaluate_memory_silence
    python -m tests.eval.evaluate_memory_silence --cases <path> --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from services.memory_models import MemoryLayer, MemoryRecord  # noqa: E402
from services.memory_retriever import (  # noqa: E402
    DEFAULT_LAYER_THRESHOLDS,
    MemoryRelevanceRetriever,
)

DEFAULT_CASES = Path(__file__).parent / "fixtures" / "memory_silence_cases.json"

_LAYER = {"L1": MemoryLayer.EXPLICIT, "L2": MemoryLayer.INFERRED, "L3": MemoryLayer.EPISODIC}


class _FixtureScorer:
    """Returns each document's fixture-declared relevance (isolates BGE)."""

    name = "fixture-relevance"

    def __init__(self, relevance_by_text: dict[str, float]):
        self._by_text = relevance_by_text

    def score(self, query: str, documents: list[str]) -> list[float]:
        return [float(self._by_text.get(doc, 0.0)) for doc in documents]


def _build_records(case: dict[str, Any], now_ms: int) -> tuple[list[MemoryRecord], dict[str, float]]:
    records: list[MemoryRecord] = []
    relevance_by_text: dict[str, float] = {}
    for spec in case.get("memories", []):
        layer = _LAYER[str(spec["layer"])]
        payload: dict[str, Any] = {
            "field": spec.get("field", ""),
            "value": spec.get("value", ""),
            "canonical_memory_id": spec["id"],
        }
        for key in ("scope", "scene", "occurred_at", "valid_until", "links", "retrieval_cues"):
            if key in spec:
                payload[key] = spec[key]
        created_at = int(spec.get("created_at", now_ms - 86_400_000))
        record = MemoryRecord(
            record_id=spec["id"], user_id="silence_eval", layer=layer, kind="preference",
            source="fixture", evidence_id="e", confidence=float(spec.get("confidence", 0.9)),
            created_at=created_at, valid_from=created_at, payload=payload,
        )
        records.append(record)
        # the retriever scores `_record_text(record)`; mirror its text assembly
        from services.memory_retriever import _record_text

        relevance_by_text[_record_text(record)] = float(spec.get("relevance", 0.0))
    return records, relevance_by_text


def evaluate(cases_path: Path) -> dict[str, Any]:
    data = json.loads(cases_path.read_text(encoding="utf-8"))
    default_now = int(data.get("now_ms") or 1_800_000_000_000)
    tp = fp = fn = 0
    silent_cases = silent_correct = 0
    rows: list[dict[str, Any]] = []

    for case in data.get("cases", []):
        now_ms = int(case.get("now_ms") or default_now)
        records, relevance_by_text = _build_records(case, now_ms)
        retriever = MemoryRelevanceRetriever(
            semantic_scorer=_FixtureScorer(relevance_by_text),
            layer_thresholds=DEFAULT_LAYER_THRESHOLDS,
            max_per_layer=5,
        )
        trace: dict[str, Any] = {}
        selected = retriever.retrieve(
            query=str(case.get("query") or ""),
            records=records,
            include_episodic=bool(case.get("include_episodic")),
            now_ms=now_ms,
            scene=str(case.get("scene") or ""),
            trace=trace,
        )
        selected_ids = {
            str(item.record.payload.get("canonical_memory_id") or item.record.record_id)
            for item in selected
        }
        gold = set(case.get("gold_inject_ids") or [])
        case_tp = len(selected_ids & gold)
        case_fp = len(selected_ids - gold)
        case_fn = len(gold - selected_ids)
        tp += case_tp
        fp += case_fp
        fn += case_fn
        if not gold:
            silent_cases += 1
            if not selected_ids:
                silent_correct += 1
        rows.append({
            "id": case.get("id"),
            "selected": sorted(selected_ids),
            "gold": sorted(gold),
            "ok": selected_ids == gold,
            "silence_decision": trace,
        })

    total_candidates = tp + fp + fn
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    over_injection = fp / total_candidates if total_candidates else 0.0
    silence_appropriateness = silent_correct / silent_cases if silent_cases else 1.0
    summary = {
        "cases": len(rows),
        "passed": sum(1 for r in rows if r["ok"]),
        "injection_precision": round(precision, 4),
        "injection_recall": round(recall, 4),
        "over_injection_rate": round(over_injection, 4),
        "silence_appropriateness": round(silence_appropriateness, 4),
    }
    return {"summary": summary, "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--json", action="store_true", help="emit full JSON")
    args = parser.parse_args()
    report = evaluate(args.cases)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        for row in report["rows"]:
            print(f"{'PASS' if row['ok'] else 'FAIL'}  {row['id']}  selected={row['selected']} gold={row['gold']}")
        print(json.dumps({"summary": report["summary"]}, ensure_ascii=False))
    s = report["summary"]
    # DoD draft: over-injection <= 0.05, injection precision >= 0.90
    ok = s["over_injection_rate"] <= 0.05 and s["injection_precision"] >= 0.90
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
