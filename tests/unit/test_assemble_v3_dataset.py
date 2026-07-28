import json
from pathlib import Path

import pytest

from data.sft.assemble_v3_dataset import assemble
from data.sft.build_sft_chatml import STUDENT_SYSTEM_PROMPT_V3, to_chatml


def _row(episode_id: str, request_kind: str = "recommendation") -> dict:
    lanes = {
        "recommendation": ["graph"],
        "conversation": [],
        "library": ["library"],
        "acquisition": ["web", "ingest"],
    }[request_kind]
    return {
        "episode_id": episode_id,
        "turn_id": 0,
        "current_query": episode_id,
        "teacher_decision_v3": {
            "request_kind": request_kind,
            "response_mode": "answer",
            "tool_names": lanes,
        },
    }


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_assemble_v3_dataset_is_unique_valid_and_manifested(tmp_path):
    base = tmp_path / "base.jsonl"
    recollected = tmp_path / "recollected.jsonl"
    _write(base, [_row("base")])
    _write(recollected, [_row("fixed")])
    output = tmp_path / "out.jsonl"
    manifest_path = tmp_path / "manifest.json"

    manifest = assemble(
        [("base", base), ("strong_teacher_recollection", recollected)],
        output,
        manifest_path,
        expected_recollected=1,
    )

    assert manifest["records"] == 2
    assert manifest["sample_keys_unique"] is True
    assert len(manifest["sha256"]) == 64
    assert all(
        row["training_governance"]["training_eligible"]
        for row in (
            json.loads(line)
            for line in output.read_text(encoding="utf-8").splitlines()
        )
    )


def test_assemble_v3_dataset_rejects_duplicate_keys(tmp_path):
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    _write(first, [_row("same")])
    _write(second, [_row("same")])
    with pytest.raises(ValueError, match="duplicate V3 sample key"):
        assemble(
            [("first", first), ("second", second)],
            tmp_path / "out.jsonl",
            tmp_path / "manifest.json",
            expected_recollected=0,
        )


def test_v3_chatml_uses_v3_prompt_and_target():
    chatml = to_chatml(_row("library", request_kind="library"))
    assert chatml["messages"][0]["content"] == STUDENT_SYSTEM_PROMPT_V3
    assert json.loads(chatml["messages"][-1]["content"])["tool_names"] == ["library"]
    assert chatml["meta"]["decision_schema"] == "planner_decision_v3"
