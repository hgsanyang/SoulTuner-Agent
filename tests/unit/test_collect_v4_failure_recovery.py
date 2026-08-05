from __future__ import annotations

import asyncio
import json

from data.sft.collect_v4_failure_recovery import (
    SCENARIOS,
    build_rows,
    execute_scenario,
    write_rows,
)
from schemas.tool_plan import ToolPlan


def test_every_failure_family_executes_and_produces_a_valid_recovery():
    for index, scenario in enumerate(SCENARIOS):
        row = asyncio.run(execute_scenario(scenario, index))
        assert row["meta"]["observation_origin"] == "harness_execution"
        assert row["meta"]["execution_environment"] == "controlled_harness"
        assert row["meta"]["trace_id"].startswith("harness-")
        assert row["messages"][3]["role"] == "tool"
        observations = json.loads(row["messages"][3]["content"])
        assert len(observations) == 1
        assert observations[0]["status"] in {
            "empty",
            "error",
            "timeout",
            "success",
        }
        ToolPlan.model_validate_json(row["messages"][-1]["content"])


def test_build_rows_is_balanced_and_trace_ids_are_unique(tmp_path):
    rows = asyncio.run(build_rows(20))
    assert len(rows) == 20
    assert len({row["meta"]["trace_id"] for row in rows}) == 20
    counts = {}
    for row in rows:
        key = row["lineage"]["fault_scenario"]
        counts[key] = counts.get(key, 0) + 1
    assert set(counts) == {scenario.name for scenario in SCENARIOS}
    assert set(counts.values()) == {2}

    output = tmp_path / "failure.jsonl"
    summary = write_rows(output, rows)
    assert summary["rows"] == 20
    assert summary["trace_ids"] == 20
    assert len(output.read_text(encoding="utf-8").splitlines()) == 20


def test_too_small_collection_is_rejected():
    try:
        asyncio.run(build_rows(len(SCENARIOS) - 1))
    except ValueError as exc:
        assert "at least" in str(exc)
    else:
        raise AssertionError("small collection should fail")
