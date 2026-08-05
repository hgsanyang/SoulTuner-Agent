"""Annotate entity-disjoint V4 seeds with the strong planner teacher.

The output remains a *candidate* sealed set. Rows intentionally carry no
reviewer verdict; a separate blind reviewer must add one before the manifest
schema can describe the build as frozen.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import httpx
from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from schemas.planner_decision_v3 import PlannerDecisionV3  # noqa: E402


MODEL = "qwen3.7-plus"
URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
VERSION = "v4_sealed_teacher_2026_07_29"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_api_key() -> str:
    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env", override=True)
    except ImportError:
        pass
    return str(os.getenv("DASHSCOPE_API_KEY") or "").strip()


def collection_messages(seeds: list[dict[str, Any]]) -> list[dict[str, str]]:
    schema = json.dumps(
        PlannerDecisionV3.model_json_schema(),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    contexts = [
        {
            "seed_id": seed["episode_id"],
            "current_query": seed["current_query"],
            "chat_history": seed.get("chat_history") or "",
            "previous_plan": seed.get("previous_plan") or "",
            "profile_snapshot": seed.get("profile_snapshot") or "",
            "retrieved_memories": seed.get("retrieved_memories") or [],
        }
        for seed in seeds
    ]
    system = (
        "你是 SoulTuner 的强教师 Planner。以下实体没有出现在训练集，但必须"
        "按普通用户请求理解，不能因为名字陌生而反问。推荐请求至少用 graph "
        "或 dense；明确歌曲/歌手实体应保留在 hard；外部资料或本地可能缺失时"
        "可以补 web。graph 是本地歌曲/歌手/知识卡，library 只表示用户个人"
        "点赞收藏，二者不可混淆。听感描述必须是英文 acoustic_queries。"
        "只输出 JSON 对象 {\"items\":[{\"seed_id\":\"...\","
        "\"decision\":{...}}]}，每个输入恰好一次，不要解释或思维过程。\n"
        f"PlannerDecisionV3 schema: {schema}"
    )
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": json.dumps(
                {"items": contexts},
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        },
    ]


def training_messages(seed: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "你是 SoulTuner 的 Planner。根据上下文输出严格的 "
                "PlannerDecisionV3 JSON，不要解释、Markdown 或思维过程。"
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "current_query": seed["current_query"],
                    "chat_history": seed.get("chat_history") or "",
                    "previous_plan": seed.get("previous_plan") or "",
                    "profile_snapshot": seed.get("profile_snapshot") or "",
                    "retrieved_memories": seed.get("retrieved_memories") or [],
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        },
    ]


def request_payload(seeds: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "model": MODEL,
        "messages": collection_messages(seeds),
        "temperature": 0.0,
        "max_tokens": max(1600, 900 * len(seeds)),
        "enable_thinking": False,
        "response_format": {"type": "json_object"},
    }


def parse_response(
    content: Any,
    seeds: list[dict[str, Any]],
) -> tuple[list[tuple[dict[str, Any], PlannerDecisionV3]], Counter]:
    failures: Counter[str] = Counter()
    try:
        payload = json.loads(str(content or ""))
    except json.JSONDecodeError:
        return [], Counter({"invalid_json": len(seeds)})
    items = payload.get("items") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        return [], Counter({"missing_items": len(seeds)})
    returned = {
        str(item.get("seed_id")): item.get("decision")
        for item in items
        if isinstance(item, dict)
    }
    accepted = []
    for seed in seeds:
        raw = returned.get(str(seed["episode_id"]))
        if raw is None:
            failures["missing_seed"] += 1
            continue
        try:
            decision = PlannerDecisionV3.model_validate(raw)
        except ValidationError:
            failures["schema_invalid"] += 1
            continue
        if decision.response_mode == "clarify":
            failures["false_clarification"] += 1
            continue
        accepted.append((seed, decision))
    return accepted, failures


def call_batch(
    seeds: list[dict[str, Any]],
    *,
    api_key: str,
    timeout: float,
) -> tuple[list[tuple[dict[str, Any], PlannerDecisionV3]], Counter, dict[str, Any]]:
    started = time.perf_counter()
    with httpx.Client(timeout=timeout, trust_env=False) as client:
        response = client.post(
            URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=request_payload(seeds),
        )
        response.raise_for_status()
        payload = response.json()
    accepted, failures = parse_response(
        payload["choices"][0]["message"]["content"],
        seeds,
    )
    return accepted, failures, {
        "usage": payload.get("usage") or {},
        "latency_ms": round((time.perf_counter() - started) * 1000, 1),
    }


def collect(
    seeds: list[dict[str, Any]],
    *,
    api_key: str,
    batch_size: int = 5,
    workers: int = 4,
    timeout: float = 180.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not api_key:
        raise RuntimeError("project-local DASHSCOPE_API_KEY is not configured")
    batches = [
        seeds[index : index + batch_size]
        for index in range(0, len(seeds), batch_size)
    ]
    accepted_rows = []
    failures: Counter[str] = Counter()
    usage: Counter[str] = Counter()
    latencies = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {
            pool.submit(call_batch, batch, api_key=api_key, timeout=timeout): batch
            for batch in batches
        }
        for future in as_completed(futures):
            batch = futures[future]
            try:
                accepted, batch_failures, diagnostics = future.result()
            except Exception as exc:
                failures[f"request_{type(exc).__name__}"] += len(batch)
                continue
            failures.update(batch_failures)
            for key, value in (diagnostics["usage"] or {}).items():
                if isinstance(value, (int, float)):
                    usage[key] += int(value)
            latencies.append(diagnostics["latency_ms"])
            for seed, decision in accepted:
                accepted_rows.append(
                    {
                        "messages": [
                            *training_messages(seed),
                            {
                                "role": "assistant",
                                "content": decision.model_dump_json(exclude_none=True),
                            },
                        ],
                        "meta": {
                            "seed_source": "curated_seed",
                            "episode_id": seed["episode_id"],
                            "turn_id": int(seed.get("turn_id") or 0),
                            "request_kind": decision.request_kind,
                            "trajectory_kind": "single_turn",
                            "observation_origin": "none",
                            "teacher": {
                                "model": MODEL,
                                "version": "2026-07",
                                "vendor": "dashscope",
                            },
                            "reviewer": {
                                "model": "pending-independent-review",
                                "version": "0",
                                "vendor": "pending",
                            },
                        },
                        "lineage": {
                            "collector": "collect_v4_sealed_teacher",
                            "collector_version": VERSION,
                            "entity": seed.get("entity") or {},
                            "seed_kind": seed.get("seed_kind"),
                        },
                    }
                )
    accepted_rows.sort(key=lambda row: row["meta"]["episode_id"])
    summary = {
        "requested": len(seeds),
        "accepted": len(accepted_rows),
        "acceptance_rate": round(len(accepted_rows) / max(1, len(seeds)), 4),
        "failures": dict(sorted(failures.items())),
        "usage": dict(sorted(usage.items())),
        "batches": len(batches),
        "latency_ms": {
            "mean": round(sum(latencies) / len(latencies), 1) if latencies else None,
            "max": max(latencies) if latencies else None,
        },
        "review_status": "pending_independent_review",
        "teacher_model": MODEL,
    }
    return accepted_rows, summary


def write_candidate(
    output: Path,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    if "private" not in {part.casefold() for part in output.parts}:
        raise ValueError("sealed teacher output must stay private")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds",
        type=Path,
        default=Path("data/teacher/private/v4/sealed_v4_pending.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/teacher/private/v4/sealed_v4_teacher_candidate.jsonl"),
    )
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()
    rows, summary = collect(
        load_jsonl(args.seeds),
        api_key=_load_api_key(),
        batch_size=args.batch_size,
        workers=args.workers,
        timeout=args.timeout,
    )
    write_candidate(args.output, rows, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["acceptance_rate"] >= 0.95 else 2


if __name__ == "__main__":
    raise SystemExit(main())
