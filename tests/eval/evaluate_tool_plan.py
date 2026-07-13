"""Evaluate ToolPlan v1 selection without executing recommendation tools."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import time

from dotenv import load_dotenv

load_dotenv()
os.environ["EVAL_DISABLE_SIDE_EFFECTS"] = "1"
os.environ["TEACHER_LOG"] = "0"

from agent.intent.planner import IntentPlanner  # noqa: E402
from llms.multi_llm import get_intent_chat_model  # noqa: E402
from schemas.tool_plan import tool_plan_alignment_issues  # noqa: E402


DEFAULT_CASES = Path(__file__).parent / "cases" / "tool_plan_dev.json"


async def evaluate(cases_path: Path, *, user_id: str, case_ids: set[str]) -> int:
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    if case_ids:
        cases = [case for case in cases if case["id"] in case_ids]
    planner = IntentPlanner(get_intent_chat_model)
    passed = 0
    rows = []
    for case in cases:
        started = time.perf_counter()
        try:
            plan = await planner.plan(
                user_input=case["query"],
                user_preferences="无",
                chat_history="",
                previous_plan="",
                graphzep_facts="",
                user_id=user_id,
            )
        except Exception as exc:
            row = {
                "id": case["id"],
                "passed": False,
                "error": f"{type(exc).__name__}: {str(exc)[:300]}",
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            }
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False))
            continue
        tool_plan = plan.tool_plan
        actual = {call.name.value for call in tool_plan.tool_calls}
        required = set(case.get("required_tools") or [])
        forbidden = set(case.get("forbidden_tools") or [])
        issues = tool_plan_alignment_issues(plan)
        checks = {
            "required": required.issubset(actual),
            "forbidden": not bool(forbidden & actual),
            "alignment": not issues,
            "clarification": (
                tool_plan.needs_clarification
                if case.get("expect_clarification")
                else True
            ),
        }
        ok = all(checks.values())
        passed += int(ok)
        row = {
            "id": case["id"],
            "passed": ok,
            "intent": plan.intent_type,
            "origin": tool_plan.origin,
            "tools": sorted(actual),
            "issues": issues,
            "checks": checks,
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))

    summary = {
        "passed": passed,
        "total": len(cases),
        "pass_rate": round(passed / len(cases), 4) if cases else 0.0,
        "direct_planner_rate": round(
            sum(row["origin"] == "planner" for row in rows) / len(rows), 4
        ) if rows else 0.0,
    }
    print(json.dumps({"summary": summary}, ensure_ascii=False))
    return 0 if passed == len(cases) else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--user-id", default="__tool_plan_eval__")
    parser.add_argument("--case-id", action="append", default=[])
    args = parser.parse_args()
    return asyncio.run(evaluate(args.cases, user_id=args.user_id, case_ids=set(args.case_id)))


if __name__ == "__main__":
    raise SystemExit(main())
