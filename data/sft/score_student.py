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


def _user_key(row: dict) -> str:
    """Stable alignment key = hash of the user message (uniquely identifies the
    sample); prefer explicit meta.episode_id#turn_id when the infer output kept it."""
    meta = row.get("meta") or {}
    if meta.get("episode_id") is not None and meta.get("turn_id") is not None:
        return f"{meta['episode_id']}#{meta['turn_id']}"
    for m in row.get("messages") or []:
        if m.get("role") == "user":
            return "u:" + " ".join(str(m.get("content") or "").split())
    return ""


def score(eval_path: Path, pred_path: Path) -> dict:
    from schemas.planner_decision import PlannerDecisionV2, compile_to_query_plan

    golds = [json.loads(line) for line in eval_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    preds = [json.loads(line) for line in pred_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    gold_by_key = {_user_key(g): g for g in golds}
    pred_by_key: dict[str, dict] = {}
    duplicate = 0
    for p in preds:
        k = _user_key(p)
        if k in pred_by_key:
            duplicate += 1
        else:
            pred_by_key[k] = p
    matched = sorted(set(gold_by_key) & set(pred_by_key))
    missing = sorted(set(gold_by_key) - set(pred_by_key))
    extra = sorted(set(pred_by_key) - set(gold_by_key))
    coverage = {
        "gold": len(gold_by_key),
        "pred": len(pred_by_key),
        "matched": len(matched),
        "missing": len(missing),
        "duplicate": duplicate,
        "extra": len(extra),
        "coverage": round(len(matched) / len(gold_by_key), 4) if gold_by_key else 0.0,
        "complete": len(missing) == 0 and duplicate == 0 and len(extra) == 0,
    }

    n = len(gold_by_key)  # denominator is ALWAYS the full gold set — misses count as failures
    t = {k: 0 for k in ("schema_valid", "compilable", "intent_acc", "hard_lang", "hard_instr",
                         "artist", "song", "region", "metadata", "hyde_dense", "dense",
                         "clar_tp", "clar_fp", "clar_fn")}
    lane_tp = lane_fp = lane_fn = 0

    for k in matched:
        gold = PlannerDecisionV2.model_validate(json.loads(gold_by_key[k]["messages"][-1]["content"]))
        raw = pred_by_key[k].get("prediction") or pred_by_key[k].get("response") or ""
        gold_clar = gold.intent == "clarification"
        try:
            pred = PlannerDecisionV2.model_validate(json.loads(raw))
            t["schema_valid"] += 1
        except Exception:
            if gold_clar:
                t["clar_fn"] += 1
            continue
        try:
            compile_to_query_plan(pred); t["compilable"] += 1
        except Exception:
            pass
        pred_clar = pred.intent == "clarification"
        if pred.intent == gold.intent:
            t["intent_acc"] += 1
        # clarification precision/recall (over-clarification = fp)
        if gold_clar and pred_clar:
            t["clar_tp"] += 1
        elif pred_clar and not gold_clar:
            t["clar_fp"] += 1
        elif gold_clar and not pred_clar:
            t["clar_fn"] += 1
        if (pred.hard.language or None) == (gold.hard.language or None):
            t["hard_lang"] += 1
        if pred.hard.instrumental == gold.hard.instrumental:
            t["hard_instr"] += 1
        if {a.casefold() for a in pred.hard.artist} == {a.casefold() for a in gold.hard.artist}:
            t["artist"] += 1
        if {s.casefold() for s in pred.hard.song} == {s.casefold() for s in gold.hard.song}:
            t["song"] += 1
        if (pred.hard.region or None) == (gold.hard.region or None):
            t["region"] += 1
        if pred.metadata.model_dump() == gold.metadata.model_dump():
            t["metadata"] += 1
        gl, pl = _lanes(gold), _lanes(pred)
        lane_tp += len(gl & pl); lane_fp += len(pl - gl); lane_fn += len(gl - pl)
        if "dense" in gl:
            t["dense"] += 1
            if pred.acoustic_queries:
                t["hyde_dense"] += 1

    def rate(k: str, d: int | None = None) -> float:
        denom = d if d is not None else n
        return round(t[k] / denom, 4) if denom else 0.0

    lp = t["clar_tp"] / (t["clar_tp"] + t["clar_fp"]) if (t["clar_tp"] + t["clar_fp"]) else 0.0
    lr = t["clar_tp"] / (t["clar_tp"] + t["clar_fn"]) if (t["clar_tp"] + t["clar_fn"]) else 0.0
    prec = lane_tp / (lane_tp + lane_fp) if (lane_tp + lane_fp) else 0.0
    rec = lane_tp / (lane_tp + lane_fn) if (lane_tp + lane_fn) else 0.0
    return {
        "coverage": coverage,
        "note": "denominator = full gold set; missing predictions count as failures",
        "schema_valid": rate("schema_valid"),
        "compilable": rate("compilable"),
        "intent_acc": rate("intent_acc"),
        "hard_language_match": rate("hard_lang"),
        "hard_instrumental_match": rate("hard_instr"),
        "artist_set_match": rate("artist"),
        "song_set_match": rate("song"),
        "region_match": rate("region"),
        "metadata_exact_match": rate("metadata"),
        "lane_f1": round(2 * prec * rec / (prec + rec), 4) if (prec + rec) else 0.0,
        "clarification_precision": round(lp, 4),
        "clarification_recall": round(lr, 4),
        "over_clarification_rate": round(t["clar_fp"] / n, 4) if n else 0.0,
        "hyde_present_when_dense": rate("hyde_dense", t["dense"]),
        "dense_cases": t["dense"],
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
