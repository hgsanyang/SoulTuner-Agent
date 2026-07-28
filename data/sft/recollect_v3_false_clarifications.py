"""Recollect the seven false-clarification V3 targets with the strong teacher.

Only schema-valid ``PlannerDecisionV3`` answers are written. Invalid responses,
API failures, and decisions that still clarify stay quarantined.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable, Iterable, Mapping

import httpx
from dotenv import load_dotenv
from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.sft.review_v3_ambiguous import FALSE_CLARIFICATION_REASONS  # noqa: E402
from schemas.planner_decision_v3 import (  # noqa: E402
    PLANNER_DECISION_V3_VERSION,
    PlannerDecisionV3,
)


DASHSCOPE_MODEL = "qwen3.7-plus"
DASHSCOPE_CHAT_URL = (
    "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
)
RECOLLECTION_VERSION = "false_clarification_recollection_2026_07_28"
ModelCall = Callable[[Mapping[str, Any]], Mapping[str, Any]]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def select_quarantined_rows(
    rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Select exactly the seven frozen false-clarification samples."""

    selected: dict[str, dict[str, Any]] = {}
    for raw in rows:
        episode_id = str(raw.get("episode_id") or "")
        if episode_id not in FALSE_CLARIFICATION_REASONS:
            continue
        if episode_id in selected:
            raise ValueError(f"duplicate quarantined episode_id: {episode_id}")
        decision = raw.get("teacher_decision_v3") or {}
        if decision.get("response_mode") != "clarify":
            raise ValueError(
                f"quarantined source is no longer a clarification: {episode_id}"
            )
        selected[episode_id] = dict(raw)

    expected = set(FALSE_CLARIFICATION_REASONS)
    actual = set(selected)
    if actual != expected:
        raise ValueError(
            f"false-clarification source mismatch; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    return [selected[episode_id] for episode_id in sorted(selected)]


def build_messages(row: Mapping[str, Any]) -> list[dict[str, str]]:
    schema = json.dumps(
        PlannerDecisionV3.model_json_schema(),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    context = {
        "current_query": row.get("current_query") or "",
        "chat_history": row.get("chat_history") or "",
        "previous_plan": row.get("previous_plan") or "",
        "profile_snapshot": row.get("profile_snapshot") or "",
        "retrieved_memories": row.get("retrieved_memories") or [],
    }
    system = (
        "你是 SoulTuner 的强教师 Planner。该样本此前被错误地反问；"
        "请直接产出一个可执行的 PlannerDecisionV3。"
        "必须保持用户原始语义，不得编造歌手、歌曲、历史或偏好。"
        "response_mode 必须为 answer，clarification 字段不得出现。"
        "这些样本是推荐请求，tool_names 只使用 graph/dense/web，并与请求自洽。"
        "推荐请求不能只用 web；必须包含 graph 或 dense，web 只能作为补充。"
        "只输出 JSON，不要解释、Markdown 或思维过程。\n"
        f"严格 JSON Schema：{schema}"
    )
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": json.dumps(
                context,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        },
    ]


def build_request_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "model": DASHSCOPE_MODEL,
        "messages": build_messages(row),
        "temperature": 0.0,
        "max_tokens": 1200,
        "enable_thinking": False,
        "response_format": {"type": "json_object"},
    }


def parse_answer_decision(content: Any) -> PlannerDecisionV3:
    if not isinstance(content, str) or not content.strip():
        raise ValueError("teacher returned empty content")
    try:
        decision = PlannerDecisionV3.model_validate_json(content)
    except ValidationError as exc:
        raise ValueError(
            "teacher output failed PlannerDecisionV3 validation"
        ) from exc
    if decision.response_mode != "answer":
        raise ValueError("teacher still requested clarification")
    if decision.clarification is not None:
        raise ValueError("answer decision must not include clarification")
    return decision


def call_dashscope(
    row: Mapping[str, Any],
    *,
    api_key: str,
    timeout_seconds: float,
    client: httpx.Client | None = None,
) -> Mapping[str, Any]:
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY is not configured")
    owns_client = client is None
    # Windows environments commonly carry ``NO_PROXY=...,::1``. Some httpx
    # versions parse that bare IPv6 token as an invalid ``:1`` port before the
    # request is sent. Teacher collection is a fixed HTTPS endpoint, so ignore
    # ambient proxy variables unless a caller injects its own client.
    http_client = client or httpx.Client(
        timeout=timeout_seconds,
        trust_env=False,
    )
    try:
        response = http_client.post(
            DASHSCOPE_CHAT_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=build_request_payload(row),
        )
        response.raise_for_status()
        payload = response.json()
        return {
            "content": payload["choices"][0]["message"]["content"],
            "usage": payload.get("usage") or {},
        }
    finally:
        if owns_client:
            http_client.close()


def recollect_rows(
    rows: Iterable[Mapping[str, Any]],
    model_call: ModelCall,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Accept only strict answer decisions and retain provenance."""

    selected = select_quarantined_rows(rows)
    accepted: list[dict[str, Any]] = []
    failure_types: Counter[str] = Counter()
    for row in selected:
        started = time.perf_counter()
        try:
            response = model_call(row)
            decision = parse_answer_decision(response.get("content"))
        except Exception as exc:
            failure_types[type(exc).__name__] += 1
            continue

        original_provenance = dict(row.get("provenance") or {})
        accepted.append(
            {
                **{
                    key: value
                    for key, value in row.items()
                    if key not in {"teacher_decision_v3", "migration"}
                },
                "teacher_decision_v3": decision.model_dump(
                    mode="json",
                    exclude_none=True,
                ),
                "provenance": {
                    **original_provenance,
                    "source_type": "strong_teacher_recollection",
                    "parent_source_type": original_provenance.get("source_type"),
                    "teacher_provider": "dashscope",
                    "teacher_model": DASHSCOPE_MODEL,
                    "thinking_enabled": False,
                    "decision_schema_version": PLANNER_DECISION_V3_VERSION,
                    "recollection_version": RECOLLECTION_VERSION,
                    "collected_at": int(time.time()),
                    "latency_ms": round(
                        (time.perf_counter() - started) * 1000,
                        1,
                    ),
                },
                "training_governance": {
                    "training_eligible": True,
                    "data_purpose": "planner_v3_sft",
                    "source_type": "strong_teacher_recollection",
                    "recollection_version": RECOLLECTION_VERSION,
                    "source_episode_id": row.get("episode_id"),
                },
                "recollection": {
                    "original_issue": FALSE_CLARIFICATION_REASONS[
                        str(row.get("episode_id"))
                    ],
                    "accepted_condition": "schema_valid_and_response_mode_answer",
                },
            }
        )

    summary = {
        "requested": len(selected),
        "accepted": len(accepted),
        "still_quarantined": len(selected) - len(accepted),
        "failure_types": dict(sorted(failure_types.items())),
        "provider": "dashscope",
        "model": DASHSCOPE_MODEL,
        "thinking_enabled": False,
        "schema": PLANNER_DECISION_V3_VERSION,
    }
    return accepted, summary


def ensure_private_output(path: Path) -> None:
    if "private" not in {part.casefold() for part in path.parts}:
        raise ValueError(
            "recollection output must be inside a directory named private"
        )
    if path.suffix.casefold() != ".jsonl":
        raise ValueError("recollection output must be a .jsonl file")


def write_private_jsonl(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
) -> None:
    ensure_private_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(
            json.dumps(dict(row), ensure_ascii=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--env-file",
        type=Path,
        default=PROJECT_ROOT / ".env",
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    ensure_private_output(args.out)
    load_dotenv(args.env_file, override=True)
    api_key = os.getenv("DASHSCOPE_API_KEY", "")
    if not api_key:
        print(json.dumps({"error": "DASHSCOPE_API_KEY is not configured"}))
        return 2

    rows = _load_jsonl(args.source)
    accepted, summary = recollect_rows(
        rows,
        lambda row: call_dashscope(
            row,
            api_key=api_key,
            timeout_seconds=max(1.0, args.timeout),
        ),
    )
    write_private_jsonl(args.out, accepted)
    print(json.dumps({"summary": summary}, ensure_ascii=False))
    return 0 if summary["still_quarantined"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
