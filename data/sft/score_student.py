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
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _lanes(decision) -> set[str]:
    if hasattr(decision, "request_kind"):
        return set(decision.tool_names)
    from schemas.planner_decision import _normalized_tool_lanes

    return _normalized_tool_lanes(decision.tool_names)


def _gold_lanes(raw: dict) -> set[str]:
    """The lane set the gold asks for, read straight from the raw decision.

    Deliberately does not validate the whole model: whether a category asks for
    any lanes describes the evaluation set, so it must hold even for a row whose
    prediction never parsed, and must not depend on the prediction at all.
    """
    names = [str(x) for x in (raw.get("tool_names") or [])]
    if "request_kind" in raw:
        return set(names)
    from schemas.planner_decision import _normalized_tool_lanes

    return _normalized_tool_lanes(names)


def _decision_contract(raw: dict):
    if "request_kind" in raw:
        from schemas.planner_decision_v3 import (
            PlannerDecisionV3,
            compile_v3_to_query_plan,
        )

        return PlannerDecisionV3, compile_v3_to_query_plan, "request_kind"

    from schemas.planner_decision import PlannerDecisionV2, compile_to_query_plan

    return PlannerDecisionV2, compile_to_query_plan, "intent"


#: Status for a metric that has no meaning for the rows it was asked about,
#: as distinct from a metric that was computed and came out badly. Consumers —
#: check_planner_release in particular — must never coerce this to 0.0.
NOT_APPLICABLE = "not_applicable"

#: Below this many supporting rows a metric is reported but not enforced.
#:
#: This is an operational floor, not a statistical one: it is the point below
#: which a percentage stops being worth acting on, and it does NOT license any
#: claim about resolving a 3pp difference with confidence.
MIN_OPERATIONAL_SUPPORT = 20


def _row_keys(row: dict) -> list[str]:
    """Every identifier this row can be matched by, most specific first.

    One key per row is not enough, because the two sides do not carry the same
    fields. Gold rows come from the frozen split and carry
    ``meta.episode_id``/``turn_id``; ``swift infer --result_path`` writes
    ``{dataset, labels, logprobs, messages, response}`` and drops ``meta``
    entirely. Keying each side by "meta if present, else user text" therefore
    keys gold by episode and predictions by text, and the two never meet.

    That failure is silent, which is what makes it dangerous: nothing raises,
    every gold row simply reports as missing, and each metric divides a zero
    numerator by the full gold count. The report reads 0.0 across the board —
    indistinguishable from a model that learned nothing. Measured on a 20-row
    smoke test: coverage 0.0, matched 0/20, every score 0.0, on an adapter whose
    predictions were in fact well-formed.

    Returning the full key set and matching on any intersection removes the
    dependency on both sides having chosen the same scheme.
    """
    keys: list[str] = []
    meta = row.get("meta") or {}
    if meta.get("episode_id") is not None and meta.get("turn_id") is not None:
        keys.append(f"{meta['episode_id']}#{meta['turn_id']}")
    for m in row.get("messages") or []:
        if m.get("role") == "user":
            keys.append("u:" + " ".join(str(m.get("content") or "").split()))
            break
    return keys


#: A chat template that injects an empty ``<think></think>`` pair produces a
#: response that is not JSON, even though the model emitted no reasoning at all.
#: Stripping the wrapper is not the same as ignoring thinking: the payload is
#: parsed, and the block's contents are reported so a real leak stays visible.
_THINK_BLOCK = re.compile(
    r"^\s*<\s*think(?:ing)?\s*>(.*?)<\s*/\s*think(?:ing)?\s*>\s*", re.IGNORECASE | re.DOTALL
)


def _split_thinking(raw: str) -> tuple[str, str]:
    """Return (payload, thinking_content). Absent a block, payload is unchanged."""
    match = _THINK_BLOCK.match(raw or "")
    if not match:
        return raw or "", ""
    return raw[match.end():], match.group(1)


def score(eval_path: Path, pred_path: Path) -> dict:
    golds = [json.loads(line) for line in eval_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    preds = [json.loads(line) for line in pred_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    # Gold rows are held as a list and indexed by every identifier they carry, so
    # one row indexed under two keys is still one row in the denominator.
    gold_rows = [g for g in golds if _row_keys(g)]
    gold_empty_key = len(golds) - len(gold_rows)
    gold_index: dict[str, int] = {}
    gold_duplicate = 0
    for position, g in enumerate(gold_rows):
        # Counted per row, not per key: a row that collides on both its episode id
        # and its user text is one unidentifiable row, not two problems.
        clashed = False
        for k in _row_keys(g):
            prior = gold_index.get(k)
            if prior is not None and prior != position:
                clashed = True
            else:
                gold_index[k] = position
        if clashed:
            gold_duplicate += 1

    pred_for: dict[int, dict] = {}
    duplicate = 0
    extra = 0
    for p in preds:
        position = next((gold_index[k] for k in _row_keys(p) if k in gold_index), None)
        if position is None:
            extra += 1
        elif position in pred_for:
            duplicate += 1
        else:
            pred_for[position] = p

    matched = sorted(pred_for)
    coverage = {
        "gold": len(gold_rows),
        "pred": len(preds),
        "matched": len(matched),
        "missing": len(gold_rows) - len(matched),
        "duplicate": duplicate,
        "extra": extra,
        "gold_duplicate": gold_duplicate,
        "gold_empty_key": gold_empty_key,
        "coverage": round(len(matched) / len(gold_rows), 4) if gold_rows else 0.0,
        "complete": (len(matched) == len(gold_rows) and duplicate == 0 and extra == 0
                     and gold_duplicate == 0 and gold_empty_key == 0),
    }

    n = len(gold_rows)  # denominator is ALWAYS the full gold set — misses count as failures
    by_kind: dict[str, dict[str, int]] = {}
    for row in gold_rows:
        raw = json.loads(row["messages"][-1]["content"])
        kind = str(raw.get("request_kind") or raw.get("intent") or "unknown")
        # Counted from the gold, before any prediction is looked at: whether this
        # category asks for lanes is a property of the evaluation set. Counting it
        # inside the scoring loop made it disappear whenever a prediction failed
        # to parse, which is exactly when you most want to know what was asked.
        gold_lanes = _gold_lanes(raw)
        by_kind.setdefault(
            kind,
            {
                "rows": 0,
                "schema_valid": 0,
                "request_kind_correct": 0,
                "lane_tp": 0,
                "lane_fp": 0,
                "lane_fn": 0,
                # A positive-lane F1 is only defined where the gold actually asks
                # for lanes. Counting the rows and the labels separately keeps
                # "this kind has no gold lanes at all" distinguishable from
                # "this kind has gold lanes and the model missed them".
                "lane_gold_positive_rows": 0,
                "lane_gold_positive_labels": 0,
                "tool_set_exact": 0,
            },
        )["rows"] += 1
        if gold_lanes:
            by_kind[kind]["lane_gold_positive_rows"] += 1
            by_kind[kind]["lane_gold_positive_labels"] += len(gold_lanes)
    t = {k: 0 for k in ("schema_valid", "compilable", "intent_acc", "hard_lang", "hard_instr",
                         "artist", "song", "region", "metadata", "hyde_dense", "dense",
                         "clar_tp", "clar_fp", "clar_fn",
                         "thinking_wrapped", "thinking_nonempty",
                         "tool_set_exact")}
    lane_tp = lane_fp = lane_fn = 0

    for position in matched:
        gold_raw = json.loads(gold_rows[position]["messages"][-1]["content"])
        decision_model, compiler, decision_field = _decision_contract(gold_raw)
        gold = decision_model.model_validate(gold_raw)
        gold_kind = str(getattr(gold, decision_field))
        kind_stats = by_kind[gold_kind]
        prediction = pred_for[position]
        emitted = prediction.get("prediction") or prediction.get("response") or ""
        raw, thinking = _split_thinking(emitted)
        if raw != emitted:
            t["thinking_wrapped"] += 1
            if thinking.strip():
                t["thinking_nonempty"] += 1
        gold_clar = (
            gold.response_mode == "clarify"
            if hasattr(gold, "response_mode")
            else gold.intent == "clarification"
        )
        try:
            pred = decision_model.model_validate(json.loads(raw))
            t["schema_valid"] += 1
            kind_stats["schema_valid"] += 1
        except Exception:
            if gold_clar:
                t["clar_fn"] += 1
            continue
        try:
            compiler(pred)
            t["compilable"] += 1
        except Exception:
            pass
        pred_clar = (
            pred.response_mode == "clarify"
            if hasattr(pred, "response_mode")
            else pred.intent == "clarification"
        )
        if getattr(pred, decision_field) == getattr(gold, decision_field):
            t["intent_acc"] += 1
            kind_stats["request_kind_correct"] += 1
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
        lane_tp += len(gl & pl)
        lane_fp += len(pl - gl)
        lane_fn += len(gl - pl)
        kind_stats["lane_tp"] += len(gl & pl)
        kind_stats["lane_fp"] += len(pl - gl)
        kind_stats["lane_fn"] += len(gl - pl)
        # Two empty sets agreeing is a correct answer, but it is not a true
        # positive: counting it as one would make F1 depend on how many rows
        # happen to want no tools. It is recorded as an exact match instead.
        # (gold-positive counts live in the pre-pass over the gold, above.)
        if gl == pl:
            t["tool_set_exact"] += 1
            kind_stats["tool_set_exact"] += 1
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
    by_request_kind = {}
    for kind, values in sorted(by_kind.items()):
        rows = values["rows"]
        kind_precision_denominator = values["lane_tp"] + values["lane_fp"]
        kind_recall_denominator = values["lane_tp"] + values["lane_fn"]
        kind_precision = (
            values["lane_tp"] / kind_precision_denominator
            if kind_precision_denominator
            else 1.0
        )
        kind_recall = (
            values["lane_tp"] / kind_recall_denominator
            if kind_recall_denominator
            else 1.0
        )
        # A positive-lane F1 needs at least one gold lane to be positive about.
        # `conversation` has none — every gold row correctly asks for no tools —
        # so tp/fp/fn are all zero for the rows that are right, and the whole
        # category's F1 ends up decided by whichever single row is wrong. The
        # measured value for that category was 0.0 while 100 of its 101 rows
        # matched exactly. Reporting null with an explicit status says "this
        # metric does not apply here", which is different from "it scored zero".
        lane_measurable = values["lane_gold_positive_labels"] > 0
        lane_f1 = (
            round(2 * kind_precision * kind_recall / (kind_precision + kind_recall), 4)
            if lane_measurable and (kind_precision + kind_recall)
            else (None if not lane_measurable else 0.0)
        )
        by_request_kind[kind] = {
            "rows": rows,
            "schema_valid": round(values["schema_valid"] / rows, 4) if rows else 0.0,
            "request_kind_acc": round(values["request_kind_correct"] / rows, 4) if rows else 0.0,
            "lane_f1": lane_f1,
            "lane_f1_status": "measured" if lane_measurable else NOT_APPLICABLE,
            "lane_gold_positive_rows": values["lane_gold_positive_rows"],
            "lane_gold_positive_labels": values["lane_gold_positive_labels"],
            # Defined for every category, including the ones where the right
            # answer is an empty tool set. Denominator is the category's full
            # gold count, so an unparseable prediction counts as a miss.
            "tool_set_exact_match": round(values["tool_set_exact"] / rows, 4) if rows else 0.0,
            "tool_set_exact_match_numerator": values["tool_set_exact"],
            "tool_set_exact_match_denominator": rows,
        }

    return {
        "coverage": coverage,
        "note": "denominator = full gold set; missing predictions count as failures",
        "schema_valid": rate("schema_valid"),
        "compilable": rate("compilable"),
        "intent_acc": rate("intent_acc"),
        "request_kind_acc": (
            rate("intent_acc")
            if golds
            and "request_kind"
            in json.loads(golds[0]["messages"][-1]["content"])
            else None
        ),
        "hard_language_match": rate("hard_lang"),
        "hard_instrumental_match": rate("hard_instr"),
        "artist_set_match": rate("artist"),
        "song_set_match": rate("song"),
        "region_match": rate("region"),
        "metadata_exact_match": rate("metadata"),
        "lane_f1": round(2 * prec * rec / (prec + rec), 4) if (prec + rec) else 0.0,
        "lane_gold_positive_rows": sum(
            v["lane_gold_positive_rows"] for v in by_kind.values()
        ),
        "lane_gold_positive_labels": sum(
            v["lane_gold_positive_labels"] for v in by_kind.values()
        ),
        "tool_set_exact_match": rate("tool_set_exact"),
        "tool_set_exact_match_numerator": t["tool_set_exact"],
        "tool_set_exact_match_denominator": n,
        "clarification_precision": round(lp, 4),
        "clarification_recall": round(lr, 4),
        "clarification_gold_cases": t["clar_tp"] + t["clar_fn"],
        "clarification_predicted_cases": t["clar_tp"] + t["clar_fp"],
        # Precision and recall do not share a denominator, so they do not share
        # a support either. Judging both by the gold count let a precision built
        # on 5 predictions be enforced as though it rested on 3 gold rows.
        "clarification_precision_support": t["clar_tp"] + t["clar_fp"],
        "clarification_recall_support": t["clar_tp"] + t["clar_fn"],
        "over_clarification_rate": round(t["clar_fp"] / n, 4) if n else 0.0,
        "lane_authority_violations": n - t["compilable"],
        "hyde_present_when_dense": rate("hyde_dense", t["dense"]),
        "dense_cases": t["dense"],
        # Reported, never silently absorbed. An empty wrapper is a chat-template
        # artifact; a non-empty one means the model really did emit reasoning,
        # and that is a finding about the run rather than a parsing detail.
        "thinking_wrapped": t["thinking_wrapped"],
        "thinking_nonempty": t["thinking_nonempty"],
        "by_request_kind": by_request_kind,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval", type=Path, required=True)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--json", dest="json_out", type=Path)
    parser.add_argument("--no-strict", dest="strict", action="store_false",
                        help="Do not fail the pipeline on incomplete coverage / invalid schema")
    args = parser.parse_args()
    report = score(args.eval, args.pred)
    cov = report["coverage"]
    gate_fail = (
        not cov["complete"]
        or cov["missing"] > 0
        or cov["duplicate"] > 0
        or cov["extra"] > 0
        or report["schema_valid"] < 1.0
    )
    report["gate"] = {"passed": not gate_fail, "strict": args.strict}
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if gate_fail and args.strict:
        print("GATE FAIL: incomplete coverage or invalid schema — pipeline must not proceed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
