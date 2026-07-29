#!/usr/bin/env python
"""Repeatable gate for a planner SFT split.

Freezes the checks from the 2026-07-29 independent audit so they can be re-run
against any future dataset instead of being a one-off analysis.

Two tiers, and the split between them is the whole design:

``hard``   schema validity and the executable contract. A violation means the
           agent cannot run the decision, so it exits non-zero.
``report`` coverage, class skew, cross-split entity overlap. These are real
           problems for *training*, but they do not stop a 50-step environment
           preflight, so they never block. Making them blocking would have
           stopped the preflight the V3 data is genuinely fine for.

Never rewrites the input. Never prints sample text — findings carry
``episode_id:turn_id`` and machine facts (lane names, counts) only, so the
output can be pasted into a report without leaking private queries.

    python scripts/validate_sft_dataset.py --train data/sft/train_v3_chatml.jsonl \
                                           --eval  data/sft/eval_v3_chatml.jsonl
    python scripts/validate_sft_dataset.py ... --json out.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from schemas.planner_decision_v3 import PlannerDecisionV3  # noqa: E402

CJK = re.compile(r"[一-鿿]")
_CURRENT_INPUT = re.compile(r"\[当前输入\]\s*(.*)", re.S)
_ENTITY_NOISE = re.compile(r"[「」《》“”\"'（）()]")
_MEMORY_BLOCK = re.compile(r"\[(长期)?记忆|\[画像")
FAILURE_MARKERS = ("失败", "超时", "没有找到", "没找到", "无结果", "报错", "error", "timeout")


@dataclass
class Finding:
    check: str
    split: str
    sample_id: str
    fact: str          # machine facts only — never user text


@dataclass
class Report:
    hard: list[Finding] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.hard


# ---------------------------------------------------------------- loading ----

def load_rows(path: str | Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sample_id(row: dict[str, Any], index: int) -> str:
    meta = row.get("meta") or {}
    episode = meta.get("episode_id")
    if episode:
        return f"{episode}:{meta.get('turn_id', 0)}"
    return f"#{index}"


def messages(row: dict[str, Any]) -> dict[str, str]:
    return {m.get("role"): (m.get("content") or "") for m in row.get("messages", [])}


def current_input(user: str) -> str:
    match = _CURRENT_INPUT.search(user)
    return (match.group(1) if match else user).strip()


def normalise(text: str) -> str:
    """Entity-blind form: digits and quoted names collapse, so two prompts that
    differ only by which artist is named land on the same key."""
    return re.sub(r"\s+", "", re.sub(r"\d+", "#", _ENTITY_NOISE.sub("", text)))


def shingles(text: str, n: int = 5) -> set[str]:
    t = normalise(text)
    return {t[i : i + n] for i in range(max(len(t) - n + 1, 1))}


# ------------------------------------------------------- hard: the contract ---

def _schema_fact(exc: Exception) -> str:
    """Describe a validation failure without echoing the offending value.

    pydantic's ``loc``/``type``/``msg`` are authored strings and schema
    vocabulary; ``input`` is the user's data and is deliberately never read.
    """
    errors = getattr(exc, "errors", None)
    if not callable(errors):
        return type(exc).__name__
    parts = []
    for err in errors()[:3]:
        loc = ".".join(str(x) for x in (err.get("loc") or ())) or "(model)"
        parts.append(f"{loc}[{err.get('type')}] {err.get('msg')}")
    return "; ".join(parts) or type(exc).__name__


def check_contract(rows: list[dict[str, Any]], split: str) -> tuple[list[Finding], list]:
    """Schema plus the executable rules the schema does *not* already enforce.

    ``PlannerDecisionV3._enforce_invariants`` already rejects: clarify without
    text, clarify with lanes, conversation with lanes, no lane at all, a lane
    outside ``KIND_LANES``, and a missing ``KIND_REQUIRED_ANY`` lane. Re-checking
    those here would be unreachable code that reads like coverage — anything the
    model rejects surfaces as ``schema_invalid`` with the specific invariant in
    ``fact``. Only the rules below survive ``model_validate`` and so need their
    own check.
    """
    findings: list[Finding] = []
    decisions: list[PlannerDecisionV3 | None] = []

    for index, row in enumerate(rows):
        sid = sample_id(row, index)
        raw = messages(row).get("assistant", "")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            findings.append(Finding("assistant_not_json", split, sid, type(exc).__name__))
            decisions.append(None)
            continue
        try:
            decision = PlannerDecisionV3.model_validate(payload)
        except Exception as exc:
            findings.append(Finding("schema_invalid", split, sid, _schema_fact(exc)))
            decisions.append(None)
            continue
        decisions.append(decision)

        lanes = set(decision.tool_names)
        lane_fact = f"kind={decision.request_kind} lanes={sorted(lanes)}"

        # The dense lane is a vector search: selected with no query, it runs on
        # nothing. NOTE the inverse is NOT a defect — a named-song request that
        # goes graph-only and writes no acoustic query is correct behaviour, and
        # flagging it was a real measurement error in the first audit pass.
        if "dense" in lanes and not decision.acoustic_queries:
            findings.append(Finding("dense_lane_without_query", split, sid, lane_fact))
        if decision.acoustic_queries and "dense" not in lanes:
            findings.append(Finding("query_with_no_dense_lane", split, sid, lane_fact))
        if any(CJK.search(q) for q in decision.acoustic_queries):
            # MuQ-MuLan's text tower is English-trained; a Chinese acoustic query
            # degrades text-to-music recall rather than merely reading oddly.
            findings.append(Finding("acoustic_query_not_english", split, sid,
                                    f"n_queries={len(decision.acoustic_queries)}"))

    return findings, decisions


# ------------------------------------------------- report: coverage & leak ----

def measure_coverage(rows: list[dict[str, Any]], decisions: list) -> dict[str, Any]:
    valid = [d for d in decisions if d is not None]
    kinds = Counter(d.request_kind for d in valid)
    total = sum(kinds.values()) or 1
    users = [messages(r).get("user", "") for r in rows]
    n = max(len(rows), 1)
    return {
        "rows": len(rows),
        "request_kind": dict(kinds.most_common()),
        "request_kind_share": {k: round(v / total, 4) for k, v in kinds.most_common()},
        "recommendation_share": round(kinds.get("recommendation", 0) / total, 4),
        # what one sample is worth: a class with 1 sample cannot measure a 3pp gate
        "pp_per_sample": {k: round(100.0 / v, 2) for k, v in kinds.items()},
        "clarify": sum(1 for d in valid if d.response_mode == "clarify"),
        "lane_use": dict(Counter(lane for d in valid for lane in d.tool_names).most_common()),
        "multi_turn_share": round(
            sum(1 for r in rows if int((r.get("meta") or {}).get("turn_id") or 0) > 0) / n, 4),
        "turn_ids": sorted({int((r.get("meta") or {}).get("turn_id") or 0) for r in rows}),
        "history_share": round(sum(1 for u in users if "[对话历史]" in u) / n, 4),
        "prev_plan_share": round(sum(1 for u in users if "[上轮检索计划]" in u) / n, 4),
        "memory_share": round(sum(1 for u in users if _MEMORY_BLOCK.search(u)) / n, 4),
        "failure_recovery_rows": sum(
            1 for u in users if any(m in u.lower() for m in FAILURE_MARKERS)),
        "rows_with_execution_trace": sum(
            1 for r in rows
            if any(k in (r.get("meta") or {}) for k in ("trace_id", "run_id", "observation"))),
    }


def _entities(decisions: Iterable) -> tuple[Counter, Counter]:
    artists: Counter = Counter()
    songs: Counter = Counter()
    for decision in decisions:
        if decision is None:
            continue
        hard = decision.hard.model_dump()
        for name in hard.get("artist") or []:
            artists[str(name).strip().casefold()] += 1
        for name in hard.get("song") or []:
            songs[str(name).strip().casefold()] += 1
    return artists, songs


def measure_overlap(train: list, eval_rows: list, dtrain: list, deval: list) -> dict[str, Any]:
    tr_user = [messages(r).get("user", "") for r in train]
    ev_user = [messages(r).get("user", "") for r in eval_rows]
    tr_cur = [current_input(u) for u in tr_user]
    ev_cur = [current_input(u) for u in ev_user]

    tr_ep = {(r.get("meta") or {}).get("episode_id") for r in train}
    ev_ep = {(r.get("meta") or {}).get("episode_id") for r in eval_rows}

    tr_norm = {normalise(u) for u in tr_cur}
    ev_norm = [normalise(u) for u in ev_cur]

    tr_shingles = [shingles(u) for u in tr_cur]
    best_jaccard = []
    for text in ev_cur:
        s = shingles(text)
        best = 0.0
        for other in tr_shingles:
            inter = len(s & other)
            if inter:
                best = max(best, inter / len(s | other))
        best_jaccard.append(best)

    tra, trs = _entities(dtrain)
    eva, evs = _entities(deval)

    def entity_block(train_c: Counter, eval_c: Counter) -> dict[str, Any]:
        shared = set(train_c) & set(eval_c)
        mentions = sum(eval_c.values())
        return {
            "train_distinct": len(train_c),
            "eval_distinct": len(eval_c),
            "shared_distinct": len(shared),
            "eval_mentions": mentions,
            "eval_mentions_seen_in_train": sum(eval_c[x] for x in shared),
            "share_seen": round(sum(eval_c[x] for x in shared) / max(mentions, 1), 4),
        }

    return {
        "episode_shared": len({x for x in (tr_ep & ev_ep) if x}),
        "exact_user_shared": len(set(tr_user) & set(ev_user)),
        "exact_current_input_shared": len(set(tr_cur) & set(ev_cur)),
        "template_shared_forms": len(tr_norm & set(ev_norm)),
        "near_dupe_ge_0_9": sum(1 for j in best_jaccard if j >= 0.9),
        "near_dupe_ge_0_8": sum(1 for j in best_jaccard if j >= 0.8),
        "near_dupe_median": round(sorted(best_jaccard)[len(best_jaccard) // 2], 4)
        if best_jaccard else 0.0,
        # episode_id cannot see this one: the same artist under new phrasing
        "artists": entity_block(tra, eva),
        "songs": entity_block(trs, evs),
    }


# -------------------------------------------------------------------- run ----

def validate(train_path: str | Path, eval_path: str | Path) -> Report:
    train, eval_rows = load_rows(train_path), load_rows(eval_path)
    report = Report()

    tr_findings, dtrain = check_contract(train, "train")
    ev_findings, deval = check_contract(eval_rows, "eval")
    report.hard = tr_findings + ev_findings

    systems = {messages(r).get("system", "") for r in train + eval_rows}
    report.stats = {
        "system_prompt_variants": len(systems),
        "coverage": {
            "train": measure_coverage(train, dtrain),
            "eval": measure_coverage(eval_rows, deval),
        },
        "overlap": measure_overlap(train, eval_rows, dtrain, deval),
        "hard_findings_by_check": dict(Counter(f.check for f in report.hard).most_common()),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", dest="eval_path", required=True)
    parser.add_argument("--json", dest="json_out", help="write the full report here")
    parser.add_argument("--max-findings", type=int, default=25)
    args = parser.parse_args()

    report = validate(args.train, args.eval_path)

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps({"hard": [asdict(f) for f in report.hard], "stats": report.stats},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"report -> {args.json_out}")

    print(f"system prompt variants: {report.stats['system_prompt_variants']}")
    for split in ("train", "eval"):
        cov = report.stats["coverage"][split]
        print(f"{split}: rows={cov['rows']} "
              f"recommendation={cov['recommendation_share']:.1%} "
              f"clarify={cov['clarify']} "
              f"failure_rows={cov['failure_recovery_rows']} "
              f"traced={cov['rows_with_execution_trace']}")
    ov = report.stats["overlap"]
    print(f"overlap: episode={ov['episode_shared']} exact={ov['exact_user_shared']} "
          f"template={ov['template_shared_forms']} near>=0.8={ov['near_dupe_ge_0_8']} "
          f"eval artists seen in train={ov['artists']['share_seen']:.1%}")

    if report.ok:
        print("CONTRACT OK — no schema or executability violation")
        return 0
    print(f"CONTRACT FAIL — {len(report.hard)} violation(s):")
    for finding in report.hard[: args.max_findings]:
        print(f"  [{finding.check}] {finding.split} {finding.sample_id} :: {finding.fact}")
    if len(report.hard) > args.max_findings:
        print(f"  ... and {len(report.hard) - args.max_findings} more")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
