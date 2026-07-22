"""Collect strong-model planner trajectories for Phase 4 distillation.

Runs the production IntentPlanner (qwen3.7-plus) over a query set and records
each full trajectory — intent, retrieval plan, tool plan, reasoning — with
provenance (model, prompt version, tool-schema version, timestamp). Output is
schema-valid by construction (the planner returns a validated MusicQueryPlan),
deduped by normalized query, and written as JSONL for SFT formatting.

Side effects are disabled during collection: no Neo4j / memory / feedback
writes, so a large run is safe and reproducible.

Usage:
    # validation batch (small, cheap) from the curated seed queries
    python -m data.sft.collect_teacher_trajectories --limit 20 --out data/teacher/collected_smoke.jsonl

    # full run from a custom query file (jsonl {"query": ...} or one per line)
    python -m data.sft.collect_teacher_trajectories --queries data/sft/queries.jsonl \
        --out data/teacher/collected.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("EVAL_DISABLE_SIDE_EFFECTS", "1")
os.environ.setdefault("TEACHER_LOG", "0")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _normalize_query(text: str) -> str:
    return " ".join(str(text or "").split()).strip().casefold()


def load_queries(path: Path | None, limit: int | None) -> list[str]:
    """Load queries from a file, or fall back to the curated SFT seed inputs."""
    queries: list[str] = []
    if path is not None:
        raw = path.read_text(encoding="utf-8")
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                try:
                    obj = json.loads(line)
                    q = str(obj.get("query") or obj.get("input") or "").strip()
                except json.JSONDecodeError:
                    q = ""
            else:
                q = line
            if q:
                queries.append(q)
    else:
        from data.sft.generate_planner_sft import data as seed_data

        queries = [str(item.get("input") or "").strip() for item in seed_data if item.get("input")]

    # dedup preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for q in queries:
        key = _normalize_query(q)
        if key and key not in seen:
            seen.add(key)
            unique.append(q)
    if limit:
        unique = unique[: max(1, int(limit))]
    return unique


async def collect(queries: list[str], out_path: Path, case_timeout: float) -> dict:
    from agent.intent.planner import UNIFIED_PLANNER_PROMPT_VERSION, IntentPlanner
    from llms.multi_llm import get_intent_chat_model
    from schemas.tool_plan import TOOL_PLAN_VERSION, tool_plan_alignment_issues
    from config.settings import settings

    planner = IntentPlanner(get_intent_chat_model)
    model_name = settings.intent_llm_model or settings.llm_default_model
    provider = settings.intent_llm_provider or settings.llm_default_provider

    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    failed = 0
    started = time.time()
    with out_path.open("w", encoding="utf-8") as sink:
        for index, query in enumerate(queries, 1):
            case_started = time.perf_counter()
            try:
                plan = await asyncio.wait_for(
                    planner.plan(
                        user_input=query,
                        user_preferences="无",
                        chat_history="",
                        previous_plan="",
                        graphzep_facts="",
                        user_id="__teacher_collect__",
                    ),
                    timeout=case_timeout if case_timeout > 0 else None,
                )
            except Exception as exc:
                failed += 1
                print(f"[{index}/{len(queries)}] FAIL {query[:40]}: {type(exc).__name__}")
                continue
            tool_plan = plan.tool_plan
            record = {
                "query": query,
                "output": plan.model_dump(mode="json"),
                "tools": sorted({call.name.value for call in tool_plan.tool_calls}) if tool_plan else [],
                "alignment_issues": tool_plan_alignment_issues(plan),
                "metadata": {
                    "provider": provider,
                    "model": model_name,
                    "planner_prompt_version": UNIFIED_PLANNER_PROMPT_VERSION,
                    "tool_schema_version": TOOL_PLAN_VERSION,
                    "collected_at": int(time.time()),
                    "latency_ms": round((time.perf_counter() - case_started) * 1000, 1),
                    "quality_mode": settings.planner_quality_mode,
                },
            }
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
            sink.flush()
            written += 1
            print(f"[{index}/{len(queries)}] OK {plan.intent_type} tools={record['tools']} {query[:36]}")

    return {
        "queries": len(queries),
        "written": written,
        "failed": failed,
        "elapsed_s": round(time.time() - started, 1),
        "out": str(out_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, default=None, help="query file (jsonl or txt); default = curated seed")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "data" / "teacher" / "collected.jsonl")
    parser.add_argument("--limit", type=int, default=0, help="cap number of queries (0 = all)")
    parser.add_argument("--case-timeout", type=float, default=120.0)
    args = parser.parse_args()

    queries = load_queries(args.queries, args.limit or None)
    if not queries:
        print("no queries loaded")
        return 1
    print(f"collecting {len(queries)} trajectories -> {args.out}")
    report = asyncio.run(collect(queries, args.out, args.case_timeout))
    print(json.dumps({"summary": report}, ensure_ascii=False))
    return 0 if report["written"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
