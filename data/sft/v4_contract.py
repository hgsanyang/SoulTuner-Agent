"""Executable row contract for the private Planner V4 corpus."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from schemas.planner_decision_v3 import PlannerDecisionV3, compile_v3_to_tool_plan


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_SCHEMA = PROJECT_ROOT / "data" / "sft" / "v4" / "MANIFEST.schema.json"


def _assistant_payload(row: dict[str, Any]) -> dict[str, Any]:
    messages = row.get("messages") or []
    assistant = [message for message in messages if message.get("role") == "assistant"]
    if not assistant:
        raise ValueError("row has no assistant message")
    return json.loads(str(assistant[-1].get("content") or ""))


def provenance_errors(meta: dict[str, Any], *, schema_path: Path = MANIFEST_SCHEMA) -> list[str]:
    """Validate the actual ``meta`` object against ``sampleProvenance``.

    Manifest definitions are not active merely because they exist under
    ``$defs``.  This explicit entry point is the missing link between the
    documented contract and real JSONL rows.
    """
    try:
        from jsonschema import Draft202012Validator
    except ImportError:
        return ["jsonschema is not installed"]
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    row_schema = {
        **schema["$defs"]["sampleProvenance"],
        "$defs": schema["$defs"],
    }
    validator = Draft202012Validator(row_schema)
    return [
        f"{'/'.join(str(part) for part in error.path) or '(meta)'}: {error.message}"
        for error in sorted(validator.iter_errors(meta), key=lambda item: list(item.path))
    ]


def row_contract_errors(row: dict[str, Any]) -> list[str]:
    errors = provenance_errors(dict(row.get("meta") or {}))
    try:
        decision = PlannerDecisionV3.model_validate(_assistant_payload(row))
        compile_v3_to_tool_plan(decision)
    except Exception as exc:
        errors.append(f"decision: {type(exc).__name__}: {exc}")
    return errors


def validate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    findings = []
    for index, row in enumerate(rows):
        errors = row_contract_errors(row)
        if errors:
            meta = row.get("meta") or {}
            findings.append(
                {
                    "row": index,
                    "sample": f"{meta.get('episode_id', '?')}#{meta.get('turn_id', '?')}",
                    "errors": errors,
                }
            )
    return {"rows": len(rows), "invalid_rows": len(findings), "findings": findings}

