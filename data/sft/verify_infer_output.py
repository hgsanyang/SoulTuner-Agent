#!/usr/bin/env python
"""Prove that a real ``swift infer`` run came back without a thinking block.

Turning thinking off is a claim about the runtime, and a claim about the runtime
can only be settled by the runtime. The training script used to settle it with
``grep -q '<think>'``, which misses every other way a thinking-capable Qwen
build can leak its scratchpad: a closing tag with no opening tag when the
opening one was consumed by the template, a ``reasoning_content`` field beside
the answer, or the fullwidth ``◁think▷`` delimiters some builds emit.

This reads the JSONL that ``swift infer --result_path`` writes and fails if any
of those appear. It also reports how many responses parse as the target schema,
because "the model emitted its reasoning" and "the model emitted invalid JSON"
have the same downstream symptom and must not be confused for each other.

Exit codes: 0 clean, 4 unusable input, 9 a thinking block survived.

    python -m data.sft.verify_infer_output --pred output/.../eval_predictions.jsonl
    python -m data.sft.verify_infer_output --pred ... --schema tool_plan --json out.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

EXIT_OK = 0
EXIT_UNUSABLE = 4
EXIT_THINKING = 9

# Opening and closing forms are listed separately on purpose: a template that
# injects the opening tag itself leaves only the closing tag in the response,
# and matching just "<think>" would call that clean.
THINKING_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("angle_open", re.compile(r"<\s*think(ing)?\s*>", re.I)),
    ("angle_close", re.compile(r"<\s*/\s*think(ing)?\s*>", re.I)),
    ("fullwidth", re.compile(r"◁\s*/?\s*think\s*▷")),
    ("pipe_tag", re.compile(r"<\|\s*/?\s*think(ing)?\s*\|>", re.I)),
    ("bracket_tag", re.compile(r"\[\s*/?\s*(thinking|thought)\s*\]", re.I)),
)

# Fields a chat backend uses to return the scratchpad beside the answer.
REASONING_FIELDS = ("reasoning_content", "reasoning", "thinking", "thought")

RESPONSE_FIELDS = ("response", "prediction", "generated_text", "output", "content")


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        rows.append(payload if isinstance(payload, dict) else {"response": payload})
    return rows


def extract_response(row: dict[str, Any]) -> str:
    for field in RESPONSE_FIELDS:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value
    messages = row.get("messages")
    if isinstance(messages, list):
        for message in reversed(messages):
            if isinstance(message, dict) and message.get("role") == "assistant":
                content = message.get("content")
                if isinstance(content, str):
                    return content
    return ""


def _walk_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [text for item in value.values() for text in _walk_strings(item)]
    if isinstance(value, list):
        return [text for item in value for text in _walk_strings(item)]
    return []


def _reasoning_hits(row: dict[str, Any]) -> list[str]:
    hits = []
    stack: list[Any] = [row]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            for key, value in node.items():
                if key in REASONING_FIELDS and value not in (None, "", [], {}):
                    hits.append(f"field:{key}")
                stack.append(value)
        elif isinstance(node, list):
            stack.extend(node)
    return hits


def scan_row(row: dict[str, Any]) -> list[str]:
    """Names of every thinking signal present anywhere in the row."""
    hits = list(_reasoning_hits(row))
    for text in _walk_strings(row):
        for name, pattern in THINKING_PATTERNS:
            if pattern.search(text):
                hits.append(name)
    return sorted(set(hits))


def _parses_as(text: str, schema: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return False
    if schema == "none":
        return True
    try:
        if schema == "tool_plan":
            from schemas.tool_plan import ToolPlan

            ToolPlan.model_validate(payload)
        else:
            from schemas.planner_decision_v3 import PlannerDecisionV3

            PlannerDecisionV3.model_validate(payload)
    except Exception:
        return False
    return True


def verify(path: Path, *, schema: str = "planner_v3") -> tuple[int, dict[str, Any]]:
    report: dict[str, Any] = {
        "path": str(path),
        "rows": 0,
        "rows_with_thinking": 0,
        "hits_by_kind": {},
        "offending_row_indexes": [],
        "rows_parsing_as_schema": 0,
        "schema": schema,
        "problems": [],
    }
    if not path.is_file():
        report["problems"].append(f"prediction file not found: {path}")
        return EXIT_UNUSABLE, report
    try:
        rows = load_rows(path)
    except (OSError, json.JSONDecodeError) as exc:
        report["problems"].append(f"prediction file is not readable JSONL: {exc}")
        return EXIT_UNUSABLE, report
    if not rows:
        report["problems"].append("prediction file is empty; there is nothing to verify")
        return EXIT_UNUSABLE, report

    report["rows"] = len(rows)
    counts: dict[str, int] = {}
    for index, row in enumerate(rows):
        hits = scan_row(row)
        if hits:
            report["rows_with_thinking"] += 1
            if len(report["offending_row_indexes"]) < 20:
                report["offending_row_indexes"].append(index)
            for hit in hits:
                counts[hit] = counts.get(hit, 0) + 1
        if _parses_as(extract_response(row), schema):
            report["rows_parsing_as_schema"] += 1
    report["hits_by_kind"] = dict(sorted(counts.items()))
    report["schema_parse_rate"] = round(
        report["rows_parsing_as_schema"] / len(rows), 4
    )

    if report["rows_with_thinking"]:
        report["problems"].append(
            f"{report['rows_with_thinking']}/{len(rows)} responses still carry a "
            f"thinking block: {report['hits_by_kind']}"
        )
        return EXIT_THINKING, report
    return EXIT_OK, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument(
        "--schema",
        choices=("planner_v3", "tool_plan", "none"),
        default="planner_v3",
    )
    parser.add_argument("--json", dest="json_out", type=Path)
    args = parser.parse_args(argv)

    code, report = verify(args.pred, schema=args.schema)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    if code == EXIT_OK:
        print(
            f"INFER OK: rows={report['rows']} no thinking block; "
            f"{report['schema']} parse rate={report['schema_parse_rate']:.1%}"
        )
        return EXIT_OK
    print("INFER FAIL:")
    for problem in report["problems"]:
        print(f"  - {problem}")
    if report.get("offending_row_indexes"):
        print(f"  first offending row indexes: {report['offending_row_indexes']}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
