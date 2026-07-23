"""Score a trained student's predictions against the teacher eval gold (Phase B).

Loop: `swift infer` produces predictions for eval_v2_chatml.jsonl, then this
scores them WITHOUT any LLM — deterministic checks only:

  - schema_valid : prediction parses as strict PlannerDecisionV2
  - compilable   : compile_to_query_plan(pred) succeeds (executable)
  - intent_acc   : pred.intent == gold.intent
  - lane_f1      : tool lanes (graph/dense/web) F1 vs gold
  - hard_match   : hard.language / instrumental / artist-set match rate
  - hyde_present : dense-lane predictions that carry acoustic_queries

HyDE *quality* (not just presence) is measured separately by the text→audio
ruler (tests/eval/evaluate_alignment_attribute.py) on the student's acoustic
queries — this script only checks structure.

Predictions file: jsonl, each line {"messages":[...], "prediction": "<json str>"}
(the ms-swift infer output). Gold is read from the same eval file's assistant.

Usage:
    python -m data.sft.score_student --eval data/sft/eval_v2_chatml.jsonl \
        --pred output/eval_predictions.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _lanes(decision) -> set[str]:
    from schemas.planner_decision import _normalized_tool_lanes

    return _normalized_tool_lanes(decision.tool_names)


def score(eval_path: Path, pred_path: Path) -> dict:
    from schemas.planner_decision import PlannerDecisionV2, compile_to_query_plan

    golds = [json.loads(line) for line in eval_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    preds = [json.loads(line) for line in pred_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    n = min(len(golds), len(preds))

    tallies = {
        "n": n, "schema_valid": 0, "compilable": 0, "intent_acc": 0,
        "hard_lang_match": 0, "hard_instr_match": 0, "artist_match": 0,
        "hyde_present_when_dense": 0, "dense_cases": 0,
    }
    lane_tp = lane_fp = lane_fn = 0

    for g, p in zip(golds[:n], preds[:n]):
        gold = PlannerDecisionV2.model_validate(json.loads(g["messages"][-1]["content"]))
        raw = p.get("prediction") or p.get("response") or ""
        try:
            pred = PlannerDecisionV2.model_validate(json.loads(raw))
            tallies["schema_valid"] += 1
        except Exception:
            continue  # invalid schema -> everything else counts as miss
        try:
            compile_to_query_plan(pred)
            tallies["compilable"] += 1
        except Exception:
            pass
        if pred.intent == gold.intent:
            tallies["intent_acc"] += 1
        if (pred.hard.language or None) == (gold.hard.language or None):
            tallies["hard_lang_match"] += 1
        if pred.hard.instrumental == gold.hard.instrumental:
            tallies["hard_instr_match"] += 1
        if {a.casefold() for a in pred.hard.artist} == {a.casefold() for a in gold.hard.artist}:
            tallies["artist_match"] += 1
        gl, pl = _lanes(gold), _lanes(pred)
        lane_tp += len(gl & pl); lane_fp += len(pl - gl); lane_fn += len(gl - pl)
        if "dense" in gl:
            tallies["dense_cases"] += 1
            if pred.acoustic_queries:
                tallies["hyde_present_when_dense"] += 1

    def rate(k: str, d: int | None = None) -> float:
        denom = d if d is not None else n
        return round(tallies[k] / denom, 4) if denom else 0.0

    prec = lane_tp / (lane_tp + lane_fp) if (lane_tp + lane_fp) else 0.0
    rec = lane_tp / (lane_tp + lane_fn) if (lane_tp + lane_fn) else 0.0
    lane_f1 = round(2 * prec * rec / (prec + rec), 4) if (prec + rec) else 0.0

    return {
        "n": n,
        "schema_valid": rate("schema_valid"),
        "compilable": rate("compilable"),
        "intent_acc": rate("intent_acc"),
        "hard_language_match": rate("hard_lang_match"),
        "hard_instrumental_match": rate("hard_instr_match"),
        "artist_set_match": rate("artist_match"),
        "lane_f1": lane_f1,
        "hyde_present_when_dense": rate("hyde_present_when_dense", tallies["dense_cases"]),
        "dense_cases": tallies["dense_cases"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval", type=Path, required=True)
    parser.add_argument("--pred", type=Path, required=True)
    args = parser.parse_args()
    report = score(args.eval, args.pred)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
