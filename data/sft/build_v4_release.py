"""Assemble, validate and fingerprint the private Planner V4 release."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any

from data.sft.build_v4_contract_curriculum import build_rows as build_contract_rows
from data.sft.build_v4_recommendation_curriculum import build_rows as build_recommendation_rows
from data.sft.build_v4_sealed_seeds import load_jsonl
from data.sft.v4_contract import validate_rows
from schemas.planner_decision_v3 import PlannerDecisionV3
from schemas.tool_plan import ToolName, ToolPlan
from scripts.validate_sft_dataset import current_input, messages, validate


PROJECT_ROOT = Path(__file__).resolve().parents[2]
KINDS = ("recommendation", "information", "acquisition", "library", "conversation")
TRAJECTORIES = {
    "single_turn",
    "multi_turn_inheritance",
    "memory_vs_current_request",
    "clarification_positive",
    "clarification_negative",
    "failure_recovery",
    "library_state",
    "acquisition",
    "conversation",
}


def _decision(row: dict[str, Any]) -> PlannerDecisionV3:
    assistant = [message for message in row.get("messages") or [] if message.get("role") == "assistant"]
    return PlannerDecisionV3.model_validate_json(str(assistant[-1]["content"]))


def _normalise_legacy(row: dict[str, Any], *, split: str) -> dict[str, Any]:
    output = json.loads(json.dumps(row, ensure_ascii=False))
    decision = _decision(output)
    meta = dict(output.get("meta") or {})
    turn_id = int(meta.get("turn_id") or 0)
    user = messages(output).get("user", "")
    if decision.response_mode == "clarify":
        trajectory = "clarification_positive"
    elif "[长期记忆]" in user or "[画像]" in user:
        trajectory = "memory_vs_current_request"
    elif turn_id > 0:
        trajectory = "multi_turn_inheritance"
    else:
        trajectory = "single_turn"
    meta.update(
        {
            "seed_source": "curated_seed",
            "episode_id": str(meta.get("episode_id") or f"{split}_legacy_missing"),
            "turn_id": turn_id,
            "request_kind": decision.request_kind,
            "trajectory_kind": trajectory,
            "observation_origin": "none",
            "teacher": {"model": "qwen3.7-plus", "version": "2026-07", "vendor": "dashscope"},
            "reviewer": {"model": "v3-contract-audit", "version": "2026-07-29", "vendor": "SoulTuner"},
            "reviewer_verdict": "accept",
        }
    )
    output["meta"] = meta
    return output


def _normalise_failure_recovery(row: dict[str, Any], *, system_prompt: str) -> dict[str, Any]:
    """Convert an executed ToolPlan recovery trace into one V3 target.

    The harness records an initial ToolPlan, a real observation and a replanned
    ToolPlan. Feeding that ChatML directly to the planner student would teach
    two incompatible assistant JSON contracts. Keep the executed evidence in
    the user context and make the only assistant target PlannerDecisionV3.
    """
    output = json.loads(json.dumps(row, ensure_ascii=False))
    original_messages = output.get("messages") or []
    assistants = [message for message in original_messages if message.get("role") == "assistant"]
    tools = [message for message in original_messages if message.get("role") == "tool"]
    users = [message for message in original_messages if message.get("role") == "user"]
    if len(assistants) < 2 or not tools or not users:
        raise ValueError("failure recovery row must contain initial plan, observation and final plan")

    initial = ToolPlan.model_validate_json(str(assistants[0]["content"]))
    recovery = ToolPlan.model_validate_json(str(assistants[-1]["content"]))
    meta = dict(output.get("meta") or {})
    request_kind = str(meta.get("request_kind") or recovery.request_mode)

    graph_arguments: dict[str, Any] = {}
    audio_arguments: dict[str, Any] = {}
    external_arguments: dict[str, Any] = {}
    lanes: list[str] = []
    # Entity constraints survive a failed lane, but acoustic queries and web
    # requirements belong only to the selected recovery lane. Otherwise a
    # timed-out dense query leaks into a graph-only target.
    for call in [*initial.tool_calls, *recovery.tool_calls]:
        if call.name == ToolName.SEARCH_GRAPH:
            graph_arguments.update(call.arguments)
    for call in recovery.tool_calls:
        if call.name == ToolName.SEARCH_AUDIO:
            audio_arguments.update(call.arguments)
        elif call.name == ToolName.SEARCH_EXTERNAL_MUSIC:
            external_arguments.update(call.arguments)
    for call in recovery.tool_calls:
        lane = {
            ToolName.SEARCH_GRAPH: "graph",
            ToolName.SEARCH_AUDIO: "dense",
            ToolName.SEARCH_EXTERNAL_MUSIC: "web",
            ToolName.READ_LIBRARY: "library",
            ToolName.STAGE_INGEST: "ingest",
        }.get(call.name)
        if lane and lane not in lanes:
            lanes.append(lane)

    if recovery.needs_clarification:
        decision = PlannerDecisionV3(
            request_kind=request_kind,
            response_mode="clarify",
            clarification=recovery.clarification_question,
            decision_summary=recovery.decision_summary[:200],
        )
    else:
        if request_kind == "acquisition":
            # V3 describes the complete reversible proposal rather than only
            # the harness's individual external-discovery retry.
            lanes = list(dict.fromkeys([*lanes, "ingest"]))
        decision = PlannerDecisionV3.model_validate(
            {
                "request_kind": request_kind,
                "response_mode": "answer",
                "tool_names": lanes,
                "hard": {
                    "artist": list(graph_arguments.get("artist_entities") or []),
                    "song": list(graph_arguments.get("song_entities") or []),
                    "language": graph_arguments.get("language"),
                    "region": graph_arguments.get("region"),
                    "instrumental": bool(graph_arguments.get("instrumental", False)),
                },
                "soft": {
                    "goal": str(external_arguments.get("requirements") or ""),
                    "trajectory": "recover after an observed tool failure",
                    "vibe": [],
                    "avoid": list(audio_arguments.get("negative_targets") or []),
                },
                "hints": {
                    "mood": list(graph_arguments.get("moods") or []),
                    "scenario": list(graph_arguments.get("scenarios") or []),
                    "genre": list(graph_arguments.get("genres") or []),
                },
                "metadata": {
                    "era": graph_arguments.get("era"),
                    "release_year_from": graph_arguments.get("release_year_from"),
                    "release_year_to": graph_arguments.get("release_year_to"),
                    "recency_required": False,
                    "external_knowledge_required": "web" in lanes,
                },
                "acoustic_queries": list(audio_arguments.get("acoustic_queries") or []),
                "decision_summary": recovery.decision_summary[:200],
            }
        )

    context = "\n".join(
        (
            "[CURRENT REQUEST]",
            str(users[-1].get("content") or ""),
            "[PREVIOUS PLAN]",
            str(assistants[0].get("content") or ""),
            "[OBSERVED TOOL RESULT]",
            str(tools[-1].get("content") or ""),
            "[TASK] Re-plan from the observed result. Do not claim an unobserved success.",
        )
    )
    output["messages"] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": context},
        {"role": "assistant", "content": decision.model_dump_json(exclude_none=True)},
    ]
    meta.update(
        {
            "request_kind": decision.request_kind,
            "trajectory_kind": "failure_recovery",
            "observation_origin": "harness_execution",
        }
    )
    output["meta"] = meta
    return output


def _sample_key(row: dict[str, Any]) -> str:
    meta = row.get("meta") or {}
    return f"{meta.get('episode_id')}#{meta.get('turn_id')}"


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _template(row: dict[str, Any]) -> str:
    text = current_input(messages(row).get("user", "")).casefold()
    decision = _decision(row)
    entities = list(decision.hard.artist) + list(decision.hard.song)
    for entity in sorted(entities, key=len, reverse=True):
        if entity:
            text = re.sub(re.escape(entity.casefold()), "<entity>", text)
    text = re.sub(r"\d+", "#", text)
    return re.sub(r"[\W_]+", "", text, flags=re.UNICODE)


def _entity_sets(rows: list[dict[str, Any]]) -> tuple[set[str], set[str]]:
    artists, songs = set(), set()
    for row in rows:
        decision = _decision(row)
        artists.update(value.strip().casefold() for value in decision.hard.artist if value.strip())
        songs.update(value.strip().casefold() for value in decision.hard.song if value.strip())
    return artists, songs


def _shingles(text: str, size: int = 5) -> set[str]:
    if not text:
        return set()
    return {text[index:index + size] for index in range(max(1, len(text) - size + 1))}


def _overlap(train: list[dict[str, Any]], sealed: list[dict[str, Any]]) -> dict[str, Any]:
    train_artists, train_songs = _entity_sets(train)
    sealed_artists, sealed_songs = _entity_sets(sealed)
    train_episodes = {str(row["meta"]["episode_id"]) for row in train}
    sealed_episodes = {str(row["meta"]["episode_id"]) for row in sealed}
    train_templates = {_template(row) for row in train}
    sealed_templates = {_template(row) for row in sealed}
    train_inputs = {current_input(messages(row).get("user", "")).strip().casefold() for row in train}
    sealed_inputs = {current_input(messages(row).get("user", "")).strip().casefold() for row in sealed}
    train_shingles = [_shingles(value) for value in train_templates if value]
    max_jaccard = 0.0
    for template in sealed_templates:
        candidate = _shingles(template)
        for reference in train_shingles:
            union = candidate | reference
            if union:
                max_jaccard = max(max_jaccard, len(candidate & reference) / len(union))
    return {
        "shared_episodes": len(train_episodes & sealed_episodes),
        "shared_artists": len(train_artists & sealed_artists),
        "shared_songs": len(train_songs & sealed_songs),
        "shared_templates": len(train_templates & sealed_templates),
        "shared_exact_inputs": len(train_inputs & sealed_inputs),
        "max_near_dupe_jaccard": round(max_jaccard, 4),
    }


def _split(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    kinds = Counter(row["meta"]["request_kind"] for row in rows)
    trajectories = Counter(row["meta"]["trajectory_kind"] for row in rows)
    return {
        "path": str(path.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "sha256": _sha(path),
        "rows": len(rows),
        "counts_by_request_kind": {kind: kinds.get(kind, 0) for kind in KINDS},
        "counts_by_trajectory": {kind: trajectories.get(kind, 0) for kind in sorted(TRAJECTORIES) if trajectories.get(kind, 0)},
        "min_rows_per_request_kind": min(kinds.get(kind, 0) for kind in KINDS),
    }


def _git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()


def build_release(base_train: Path, regression: Path, failure: Path, sealed_reviewed: Path, output_dir: Path, dataset_version: str) -> dict[str, Any]:
    base_rows = [_normalise_legacy(row, split="train") for row in load_jsonl(base_train)]
    regression_rows = [_normalise_legacy(row, split="regression") for row in load_jsonl(regression)]
    system_prompt = str(base_rows[0]["messages"][0]["content"])
    failure_rows = [
        _normalise_failure_recovery(row, system_prompt=system_prompt)
        for row in load_jsonl(failure)
    ]
    contract_rows = build_contract_rows(base_rows, counts={kind: 800 for kind in ("conversation", "library", "acquisition", "information")})
    recommendation_rows = build_recommendation_rows(base_rows)
    train_rows = base_rows + failure_rows + contract_rows + recommendation_rows
    sealed_rows = load_jsonl(sealed_reviewed)

    if len(train_rows) != 8000:
        raise ValueError(f"V4 train must contain exactly 8000 rows, got {len(train_rows)}")
    if len(sealed_rows) != 500:
        raise ValueError(f"V4 sealed must contain exactly 500 rows, got {len(sealed_rows)}")
    for name, rows in (("train", train_rows), ("regression", regression_rows), ("sealed", sealed_rows)):
        keys = [_sample_key(row) for row in rows]
        if len(keys) != len(set(keys)):
            raise ValueError(f"{name} contains duplicate episode/turn keys")
        contract = validate_rows(rows)
        if contract["invalid_rows"]:
            raise ValueError(f"{name} has invalid V4 rows: {contract['findings'][:3]}")

    overlap = _overlap(train_rows, sealed_rows)
    if any(overlap[key] for key in ("shared_episodes", "shared_artists", "shared_songs", "shared_templates", "shared_exact_inputs")):
        raise ValueError(f"sealed split is not disjoint: {overlap}")

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train_v4_chatml.jsonl"
    regression_path = output_dir / "regression_v4_chatml.jsonl"
    sealed_path = output_dir / "sealed_v4_chatml.jsonl"
    _write_jsonl(train_path, train_rows)
    _write_jsonl(regression_path, regression_rows)
    _write_jsonl(sealed_path, sealed_rows)

    gate_path = output_dir / "dataset_gate.json"
    gate = validate(train_path, sealed_path)
    gate_path.write_text(json.dumps({"hard": [finding.__dict__ for finding in gate.hard], "stats": gate.stats}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not gate.ok:
        raise ValueError(f"dataset execution gate has {len(gate.hard)} hard findings")

    counts = Counter(row["meta"]["request_kind"] for row in train_rows)
    recommendation_share = counts["recommendation"] / len(train_rows)
    multi_turn_share = sum(int(row["meta"]["turn_id"]) > 0 for row in train_rows) / len(train_rows)
    memory_share = sum(row["meta"]["trajectory_kind"] == "memory_vs_current_request" for row in train_rows) / len(train_rows)
    if not 0.55 <= recommendation_share <= 0.65:
        raise ValueError(f"recommendation share out of range: {recommendation_share:.4f}")
    if multi_turn_share < 0.30 or memory_share < 0.15:
        raise ValueError(f"context coverage too low: multi_turn={multi_turn_share:.4f} memory={memory_share:.4f}")

    manifest = {
        "manifest_version": "1.0",
        "dataset_version": dataset_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "generator_commit": _git_commit(),
        "notes": "Private planner corpus; transfer separately and never publish with source code.",
        "splits": {
            "train": _split(train_path, train_rows),
            "regression": _split(regression_path, regression_rows),
            "sealed": _split(sealed_path, sealed_rows),
        },
        "sealed_policy": {
            "entity_disjoint": True,
            "template_disjoint": True,
            "episode_namespace": "sealed_v4",
            "seed_pool": "neo4j_unseen_entities_2026_08_blind_review",
            "measured": {key: value for key, value in overlap.items() if key != "shared_exact_inputs"},
            "release_gates": {
                "overall_vs_teacher_pp": -3.0,
                "per_kind_max_regression_pp": 3.0,
                "schema_validity": 1.0,
                "lane_authority_violations": 0,
                "sealed_vs_regression_max_gap_pp": 8.0,
            },
        },
        "validator": {
            "tool": "scripts/validate_sft_dataset.py",
            "commit": _git_commit(),
            "hard_findings": 0,
            "report_path": str(gate_path.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/"),
        },
    }
    manifest_path = output_dir / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"manifest": str(manifest_path), "train_rows": len(train_rows), "sealed_rows": len(sealed_rows), "recommendation_share": round(recommendation_share, 4), "multi_turn_share": round(multi_turn_share, 4), "memory_share": round(memory_share, 4), "counts": dict(counts), "overlap": overlap}


def main() -> int:
    parser = argparse.ArgumentParser()
    root = Path("data/teacher/private/v4")
    parser.add_argument("--base-train", type=Path, default=root / "train_v3_repaired.jsonl")
    parser.add_argument("--regression", type=Path, default=root / "regression_v3_repaired.jsonl")
    parser.add_argument("--failure", type=Path, default=root / "failure_recovery_harness.jsonl")
    parser.add_argument("--sealed-reviewed", type=Path, default=root / "sealed_v4_reviewed.jsonl")
    parser.add_argument("--output-dir", type=Path, default=root / "frozen-v4.0.0")
    parser.add_argument("--dataset-version", default="v4.0.0")
    args = parser.parse_args()
    for path in (args.base_train, args.regression, args.failure, args.sealed_reviewed, args.output_dir):
        if "private" not in {part.casefold() for part in path.parts}:
            raise ValueError("V4 release inputs and outputs must stay private")
    report = build_release(args.base_train, args.regression, args.failure, args.sealed_reviewed, args.output_dir, args.dataset_version)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
