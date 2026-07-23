"""Deterministic verifier chain for collected teacher episodes (Phase A-3).

Runs offline, deterministic checks over a collected episode jsonl (the output of
``collect_episodes``). No LLM is called. Hard failures block the batch from
entering training; warnings are surfaced for human pilot review.

Checks
------
HARD (batch fails if any occur):
  V01 schema        teacher_decision validates as strict PlannerDecisionV2
  V02 compile       compile_to_query_plan(decision) succeeds (executable)
  V03 leakage       current_query not present in any frozen eval/holdout/blind case
  V04 dedup         no exact-duplicate current_query across the batch
  V05 instrumental  explicit instrumental request -> hard.instrumental=True
                    (skipped for clarification/general_chat)
  V06 clarify_sync  intent==clarification <=> clarification text present
  V07 hyde_present  dense lane (hybrid/vector) -> acoustic_queries non-empty
  V08 alignment     the COMPILED target is ToolPlan-consistent — decision is
                    compiled (compile_to_query_plan + compile_legacy_tool_plan)
                    and re-checked, so raw-plan quirks the deterministic
                    compiler heals (era gap-check, clarification retrieval) do
                    NOT disqualify a sample; we measure what the student produces.
  V09 tool_intent   tool_names (authoritative lane choice) must not contradict
                    intent: dense-intent has dense, graph has graph, web has web,
                    clarification/general_chat carry no lanes.

WARN (reported, non-blocking):
  W01 language_hint explicit language request but hard.language unset
  W02 token_budget  target_token_estimate outside [30, 400]
  W03 mem_silence   no memory injected but retrieve_memory tool requested
  W04 contradiction instrumental demand + vocal demand co-occur (pilot review)
  W05 raw_healed    teacher raw-plan alignment quirk the compiler healed (info)

Usage:
    python -m data.sft.verify_episodes --in data/teacher/private/episodes.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Audit heuristics (verifier-only; NOT a production keyword state machine).
INSTRUMENTAL_MARKERS = ("纯音乐", "器乐", "无人声", "没有人声", "无歌词", "instrumental", "no vocals", "without vocals")
# Explicit demand for vocals; co-occurring with an instrumental marker is a
# likely contradiction the LLM must judge (clarify or charitably satisfy). We
# only FLAG these for pilot human review — never hard-fail (LLM judges).
VOCAL_DEMAND_MARKERS = ("说唱", "rap", "要有人声", "突出人声", "有人声", "女声", "男声", "vocal", "演唱")
LANGUAGE_MARKERS: dict[str, tuple[str, ...]] = {
    "chinese": ("中文", "华语", "国语", "普通话", "汉语", "mandarin", "chinese"),
    "cantonese": ("粤语", "广东话", "港乐", "cantonese", "cantopop"),
    "english": ("英文", "英语", "欧美", "english"),
    "japanese": ("日文", "日语", "japanese", "j-pop", "jpop"),
    "korean": ("韩文", "韩语", "korean", "k-pop", "kpop"),
}
DENSE_INTENTS = {"hybrid_search", "vector_search"}
NO_CONSTRAINT_INTENTS = {"clarification", "general_chat"}
_INSTR_NEGATIONS = ("不", "别", "没", "無", "莫", "厌", "no ", "not ", "without")


def _affirmative_instrumental(low: str) -> bool:
    """True only if an instrumental request is AFFIRMATIVE — a mention like
    "不想听纯音乐" (negated) must not trip the instrumental hard-filter check."""
    for marker in INSTRUMENTAL_MARKERS:
        idx = low.find(marker)
        while idx != -1:
            prefix = low[max(0, idx - 4):idx]
            if not any(neg in prefix for neg in _INSTR_NEGATIONS):
                return True
            idx = low.find(marker, idx + 1)
    return False


def _eval_exclusions() -> set[str]:
    from data.sft.seeds_from_legacy import _eval_case_exclusions

    return _eval_case_exclusions()


def _load_records(path: Path) -> list[dict]:
    records: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def verify(path: Path, write_clean: Path | None = None) -> dict:
    from data.sft.augment_queries import _normalize
    from schemas.planner_decision import PlannerDecisionV2, _normalized_tool_lanes, compile_to_query_plan
    from schemas.tool_plan import compile_legacy_tool_plan, tool_plan_alignment_issues

    excluded = _eval_exclusions()
    records = _load_records(path)

    hard_fail: dict[str, list[str]] = defaultdict(list)
    warn: dict[str, list[str]] = defaultdict(list)
    seen_queries: dict[str, str] = {}

    for rec in records:
        tag = f"{rec.get('episode_id')}#t{rec.get('turn_id')}"
        query = str(rec.get("current_query") or "")
        low = query.casefold()
        decision_raw = rec.get("teacher_decision") or {}

        # V01 schema
        try:
            decision = PlannerDecisionV2.model_validate(decision_raw)
        except Exception as exc:
            hard_fail["V01_schema"].append(f"{tag}: {type(exc).__name__}")
            continue  # nothing else is trustworthy without a valid decision

        # V02 compile + V08 alignment on the COMPILED target (not the teacher's
        # throwaway verbose plan). The deterministic downstream compiler adds
        # inspect_catalog_gap / search_graph and drops retrieval for
        # clarification, so raw-plan issues that the compiler heals must NOT
        # disqualify a target. We measure what the student actually produces.
        compiled_issues: list[str] = []
        try:
            plan = compile_to_query_plan(decision)
            plan.tool_plan = compile_legacy_tool_plan(plan)
            compiled_issues = tool_plan_alignment_issues(plan)
        except Exception as exc:
            hard_fail["V02_compile"].append(f"{tag}: {type(exc).__name__}: {exc}")
            continue
        if compiled_issues:
            hard_fail["V08_alignment"].append(f"{tag}: compiled {compiled_issues}")

        # V03 leakage
        if _normalize(query) in excluded:
            hard_fail["V03_leakage"].append(f"{tag}: {query[:40]}")

        # V04 dedup (cross-episode exact duplicate)
        key = _normalize(query)
        if key:
            if key in seen_queries:
                hard_fail["V04_dedup"].append(f"{tag}: dup of {seen_queries[key]}")
            else:
                seen_queries[key] = tag

        is_clarify = decision.intent == "clarification"

        # V05 instrumental hard-filter (skip clarification/general_chat; ignore
        # negated mentions like "不想听纯音乐")
        if decision.intent not in NO_CONSTRAINT_INTENTS:
            if _affirmative_instrumental(low) and not decision.hard.instrumental:
                hard_fail["V05_instrumental"].append(f"{tag}: {query[:40]}")

        # V06 clarify sync
        has_text = bool((decision.clarification or "").strip())
        if is_clarify != has_text:
            hard_fail["V06_clarify_sync"].append(
                f"{tag}: intent={decision.intent} clarification_text={has_text}"
            )

        # V07 HyDE present for dense lane
        if decision.intent in DENSE_INTENTS and not decision.acoustic_queries:
            hard_fail["V07_hyde_present"].append(f"{tag}: {decision.intent} no acoustic_queries")

        # V09 tool_names <-> intent invariant (tool_names is the authoritative lane
        # choice; when present it must not contradict the intent).
        lanes = _normalized_tool_lanes(decision.tool_names)
        if lanes:
            if decision.intent in NO_CONSTRAINT_INTENTS:
                hard_fail["V09_tool_intent"].append(f"{tag}: {decision.intent} carries lanes {sorted(lanes)}")
            else:
                required = {"graph_search": "graph", "vector_search": "dense",
                            "hybrid_search": "dense", "web_search": "web"}.get(decision.intent)
                if required and required not in lanes:
                    hard_fail["V09_tool_intent"].append(f"{tag}: {decision.intent} lanes {sorted(lanes)} missing '{required}'")

        # V10 recall-class TARGET must carry explicit CANONICAL tool_names.
        # (schema allows empty for inference fallback, but a training target for
        # a recall intent must name graph/dense/web — no empty, no aliases.)
        if decision.intent in {"graph_search", "vector_search", "hybrid_search", "web_search"}:
            if not decision.tool_names:
                hard_fail["V10_recall_tools"].append(f"{tag}: {decision.intent} empty tool_names")
            else:
                noncanon = [t for t in decision.tool_names if str(t).strip().casefold() not in {"graph", "dense", "web"}]
                if noncanon:
                    hard_fail["V10_recall_tools"].append(f"{tag}: non-canonical tool_names {noncanon}")

        # W05 raw-plan alignment quirks the compiler heals (informational only).
        raw_issues = rec.get("alignment_issues") or []
        if raw_issues and not compiled_issues:
            warn["W05_raw_alignment_healed"].append(f"{tag}: {raw_issues}")

        # W01 explicit language but hard.language unset (skip constraint-free intents)
        if decision.intent not in NO_CONSTRAINT_INTENTS and not decision.hard.language:
            for lang, markers in LANGUAGE_MARKERS.items():
                if any(m in low for m in markers):
                    warn["W01_language_hint"].append(f"{tag}: '{lang}' cue, hard.language unset")
                    break

        # W02 token budget
        tok = rec.get("target_token_estimate")
        if isinstance(tok, (int, float)) and not (30 <= tok <= 400):
            warn["W02_token_budget"].append(f"{tag}: ~{tok} tok")

        # W03 memory silence proxy
        mems = rec.get("retrieved_memories") or []
        tools = rec.get("available_tools") or []
        if not mems and "retrieve_memory" in tools:
            warn["W03_mem_silence"].append(f"{tag}: recall requested with no memory injected")

        # W04 possible contradiction (instrumental demand + vocal demand co-occur).
        # Flagged for pilot human review; the LLM legitimately may clarify OR
        # charitably satisfy, so this is never a hard fail.
        if (
            not is_clarify
            and any(m in low for m in INSTRUMENTAL_MARKERS)
            and any(v in low for v in VOCAL_DEMAND_MARKERS)
        ):
            warn["W04_possible_contradiction"].append(f"{tag}: {query[:44]} -> {decision.intent}")

    hard_total = sum(len(v) for v in hard_fail.values())
    warn_total = sum(len(v) for v in warn.values())
    failed_tags = {entry.split(": ", 1)[0] for entries in hard_fail.values() for entry in entries}
    clean = [r for r in records if f"{r.get('episode_id')}#t{r.get('turn_id')}" not in failed_tags]
    if write_clean is not None:
        write_clean.parent.mkdir(parents=True, exist_ok=True)
        with write_clean.open("w", encoding="utf-8") as sink:
            for r in clean:
                sink.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {
        "input": str(path),
        "records": len(records),
        "unique_queries": len(seen_queries),
        "clean_records": len(clean),
        "quarantined": len(records) - len(clean),
        "hard_fail_total": hard_total,
        "warn_total": warn_total,
        "hard_fail": {k: {"count": len(v), "examples": v[:8]} for k, v in sorted(hard_fail.items())},
        "warn": {k: {"count": len(v), "examples": v[:8]} for k, v in sorted(warn.items())},
        "clean_out": str(write_clean) if write_clean is not None else "",
        "green": hard_total == 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="inp", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--write-clean", dest="write_clean", type=Path, default=None,
                        help="Write only records with zero hard failures to this path (gate output)")
    args = parser.parse_args()

    report = verify(args.inp, write_clean=args.write_clean)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("=" * 64)
    print(f"Episode verifier: {report['records']} records | {report['unique_queries']} unique")
    print(f"Clean: {report['clean_records']} | Quarantined: {report['quarantined']}")
    print(f"HARD fails: {report['hard_fail_total']} | WARNs: {report['warn_total']}")
    for check, info in report["hard_fail"].items():
        print(f"  [FAIL] {check}: {info['count']}  e.g. {info['examples'][:2]}")
    for check, info in report["warn"].items():
        print(f"  [warn] {check}: {info['count']}")
    print("GREEN" if report["green"] else "RED — hard failures present")
    return 0 if report["green"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
