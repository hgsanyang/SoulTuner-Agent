"""Collect balanced V4 planner trajectories from the strong teacher.

This is deliberately a pilot-first collector. It batches several contexts per
DashScope call, validates every returned PlannerDecisionV3, and only writes
accepted rows to the private teacher directory. Scaling is allowed only after
the acceptance and class-balance summary has been reviewed.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterable

import httpx
from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from schemas.planner_decision_v3 import (  # noqa: E402
    PLANNER_DECISION_V3_VERSION,
    PlannerDecisionV3,
)


DASHSCOPE_MODEL = "qwen3.7-plus"
DASHSCOPE_CHAT_URL = (
    "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
)
COLLECTOR_VERSION = "v4_teacher_pilot_2026_07_29"


@dataclass(frozen=True)
class Seed:
    seed_id: str
    trajectory_kind: str
    expected_kind: str
    expected_mode: str
    current_query: str
    chat_history: str = ""
    previous_plan: str = ""
    profile_snapshot: str = ""
    retrieved_memories: tuple[str, ...] = ()

    def context(self) -> dict[str, Any]:
        return {
            "seed_id": self.seed_id,
            "trajectory_kind": self.trajectory_kind,
            "current_query": self.current_query,
            "chat_history": self.chat_history,
            "previous_plan": self.previous_plan,
            "profile_snapshot": self.profile_snapshot,
            "retrieved_memories": list(self.retrieved_memories),
        }


ARTISTS = (
    "椅子乐团",
    "The Cure",
    "Vaundy",
    "朴树",
    "Massive Attack",
    "Slowdive",
    "坂本龙一",
    "张震岳",
    "Sonic Youth",
    "Lamp",
    "Corn Wave",
    "The Blue Nile",
)
COLLECTIONS = ("点赞", "收藏", "最近播放", "不喜欢")
VIBES = (
    "安静柔软",
    "克制忧郁",
    "明亮有节奏",
    "深夜氛围",
    "缓慢推进",
    "温暖但不要太甜",
)


def build_seeds(per_family: int = 10) -> list[Seed]:
    if per_family < 2:
        raise ValueError("per_family must be at least 2")
    seeds: list[Seed] = []

    for index in range(per_family):
        artist = ARTISTS[index % len(ARTISTS)]
        vibe = VIBES[index % len(VIBES)]
        collection = COLLECTIONS[index % len(COLLECTIONS)]
        seeds.extend(
            [
                Seed(
                    f"v4_conversation_{index:04d}",
                    "conversation",
                    "conversation",
                    "answer",
                    (
                        "今天只想聊聊为什么某些旋律会让人记住一段经历，"
                        "先不用推荐歌。"
                        if index % 2 == 0
                        else "谢谢你刚才陪我听歌，我们先聊聊音乐和记忆吧。"
                    ),
                ),
                Seed(
                    f"v4_library_{index:04d}",
                    "library_state",
                    "library",
                    "answer",
                    f"在我的{collection}里找和 {artist} 有关的歌，只查看，不要新增。",
                ),
                Seed(
                    f"v4_acquisition_{index:04d}",
                    "acquisition",
                    "acquisition",
                    "answer",
                    f"找到 {artist} 的正式录音版本，先放进待入库让我确认，不要直接长期保存音源。",
                ),
                Seed(
                    f"v4_information_{index:04d}",
                    "single_turn",
                    "information",
                    "answer",
                    f"{artist} 的主要风格和活跃年代是什么？本地有资料就先用本地，没有再联网。",
                ),
                Seed(
                    f"v4_multi_{index:04d}",
                    "multi_turn_inheritance",
                    "recommendation",
                    "answer",
                    "再收一点，鼓不要那么密，保留刚才的场景。",
                    chat_history=(
                        f"用户: 下雨天想听一些 {artist} 那种有空间感的歌。\n"
                        "助手: 我会从本地曲库和声学向量里找。"
                    ),
                    previous_plan=(
                        f"artist={artist}; scenario=rainy evening; "
                        "vibe=spacious and introspective"
                    ),
                ),
                Seed(
                    f"v4_memory_{index:04d}",
                    "memory_vs_current_request",
                    "recommendation",
                    "answer",
                    f"今天就想听{vibe}的，先别按我平时爱听热闹摇滚的习惯来。",
                    profile_snapshot="长期偏好：高能量摇滚、现场感、强鼓点。",
                    retrieved_memories=("最近一周常在通勤时听高能量歌曲。",),
                ),
                Seed(
                    f"v4_clarify_positive_{index:04d}",
                    "clarification_positive",
                    "acquisition" if index % 2 else "recommendation",
                    "clarify",
                    (
                        "把刚才那个版本存下来。"
                        if index % 2
                        else "帮我放那个。"
                    ),
                    chat_history=(
                        "助手刚才同时给出了两个同名版本：录音室版和现场版，"
                        "用户没有说明指哪一个。"
                    ),
                ),
            ]
        )

    # Clarification negatives are intentionally 1.5x positives. These are
    # answerable requests that a timid planner often over-clarifies.
    for index in range(per_family + per_family // 2):
        artist = ARTISTS[(index + 3) % len(ARTISTS)]
        vibe = VIBES[(index + 2) % len(VIBES)]
        variants = (
            Seed(
                f"v4_clarify_negative_{index:04d}",
                "clarification_negative",
                "recommendation",
                "answer",
                f"来点{vibe}的，具体曲目你先按默认推荐。",
            ),
            Seed(
                f"v4_clarify_negative_{index:04d}",
                "clarification_negative",
                "recommendation",
                "answer",
                "再来点这种，但少一点人声。",
                chat_history=f"用户: 想听 {artist} 那种慢慢铺开的氛围。\n助手: 已给出一组推荐。",
                previous_plan=f"artist_reference={artist}; trajectory=slow-building",
            ),
            Seed(
                f"v4_clarify_negative_{index:04d}",
                "clarification_negative",
                "recommendation",
                "answer",
                "今天想听我平时不常听的电子乐，可以陌生一点。",
                profile_snapshot="长期偏好：民谣和独立摇滚，较少主动选择电子乐。",
            ),
        )
        seeds.append(variants[index % len(variants)])

    ids = [seed.seed_id for seed in seeds]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate seed ids")
    return seeds


def build_messages(seeds: Iterable[Seed]) -> list[dict[str, str]]:
    schema = json.dumps(
        PlannerDecisionV3.model_json_schema(),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    contexts = [seed.context() for seed in seeds]
    system = (
        "你是 SoulTuner 的强教师 Planner。请独立理解每个上下文，输出严格、"
        "可执行的 PlannerDecisionV3。不要把简单含糊一律反问：能从历史解析的"
        "指代要继承，当前请求与长期画像冲突时当前请求优先，可排序的张力直接"
        "回答。只有指代确实有多个不同候选、关键对象不可确定或约束无法安全"
        "默认时才 clarify。推荐至少使用 graph 或 dense；听感需求使用英文"
        " acoustic_queries。工具职责必须严格区分：graph 查询本地歌曲、歌手、"
        "知识卡和结构化元数据；library 只查询当前用户的点赞、收藏、不喜欢和"
        "最近播放；information 只允许 graph/web；acquisition 必须含 ingest，"
        "通常是 web+ingest。不要编造历史、实体或用户偏好。只输出 JSON 对象，"
        "格式为 {\"items\":[{\"seed_id\":\"...\",\"decision\":{...}}]}，"
        "每个输入恰好返回一次，不要解释或输出思维过程。\n"
        f"PlannerDecisionV3 JSON Schema: {schema}"
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


def build_request_payload(seeds: list[Seed]) -> dict[str, Any]:
    return {
        "model": DASHSCOPE_MODEL,
        "messages": build_messages(seeds),
        "temperature": 0.0,
        "max_tokens": max(1600, 900 * len(seeds)),
        "enable_thinking": False,
        "response_format": {"type": "json_object"},
    }


def build_training_messages(seed: Seed) -> list[dict[str, str]]:
    system = (
        "你是 SoulTuner 的 Planner。根据当前输入、对话历史、上一轮计划、"
        "画像和相关记忆，输出一个严格的 PlannerDecisionV3 JSON。当前请求"
        "优先于长期画像；能从上下文解析的指代直接继承；只有无法安全确定"
        "关键对象时才反问。不要输出解释、Markdown 或思维过程。"
    )
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": json.dumps(
                seed.context(),
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        },
    ]


def parse_batch(content: Any, seeds: list[Seed]) -> tuple[list[tuple[Seed, PlannerDecisionV3]], Counter]:
    failures: Counter[str] = Counter()
    if not isinstance(content, str) or not content.strip():
        return [], Counter({"empty_content": len(seeds)})
    try:
        payload = json.loads(content)
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
        raw = returned.get(seed.seed_id)
        if raw is None:
            failures["missing_seed"] += 1
            continue
        try:
            decision = PlannerDecisionV3.model_validate(raw)
        except ValidationError:
            failures["schema_invalid"] += 1
            continue
        if decision.response_mode != seed.expected_mode:
            failures["mode_mismatch"] += 1
            continue
        if (
            decision.response_mode == "answer"
            and decision.request_kind != seed.expected_kind
        ):
            failures["kind_mismatch"] += 1
            continue
        accepted.append((seed, decision))
    return accepted, failures


def _load_api_key() -> str:
    # Project-local .env is the explicit source of truth. It intentionally
    # overrides a stale machine-wide variable and is never printed.
    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env", override=True)
    except ImportError:
        pass
    return str(os.getenv("DASHSCOPE_API_KEY") or "").strip()


def call_batch(
    seeds: list[Seed],
    *,
    api_key: str,
    timeout_seconds: float,
) -> tuple[list[tuple[Seed, PlannerDecisionV3]], Counter, dict[str, Any]]:
    started = time.perf_counter()
    with httpx.Client(timeout=timeout_seconds, trust_env=False) as client:
        response = client.post(
            DASHSCOPE_CHAT_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=build_request_payload(seeds),
        )
        response.raise_for_status()
        payload = response.json()
    accepted, failures = parse_batch(
        payload["choices"][0]["message"]["content"],
        seeds,
    )
    return accepted, failures, {
        "usage": payload.get("usage") or {},
        "latency_ms": round((time.perf_counter() - started) * 1000, 1),
    }


def collect(
    seeds: list[Seed],
    *,
    api_key: str,
    batch_size: int = 5,
    workers: int = 3,
    timeout_seconds: float = 120.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not api_key:
        raise RuntimeError("project-local DASHSCOPE_API_KEY is not configured")
    batches = [
        seeds[index : index + batch_size]
        for index in range(0, len(seeds), batch_size)
    ]
    accepted_rows: list[dict[str, Any]] = []
    failure_types: Counter[str] = Counter()
    usage: Counter[str] = Counter()
    latencies = []

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        future_batches = {
            pool.submit(
                call_batch,
                batch,
                api_key=api_key,
                timeout_seconds=timeout_seconds,
            ): batch
            for batch in batches
        }
        for future in as_completed(future_batches):
            batch = future_batches[future]
            try:
                accepted, failures, diagnostics = future.result()
            except Exception as exc:
                failure_types[f"request_{type(exc).__name__}"] += len(batch)
                continue
            failure_types.update(failures)
            for key, value in (diagnostics.get("usage") or {}).items():
                if isinstance(value, (int, float)):
                    usage[key] += int(value)
            latencies.append(float(diagnostics["latency_ms"]))
            for seed, decision in accepted:
                accepted_rows.append(
                    {
                        "messages": [
                            *build_training_messages(seed),
                            {
                                "role": "assistant",
                                "content": decision.model_dump_json(exclude_none=True),
                            },
                        ],
                        "meta": {
                            "seed_source": "curated_seed",
                            "episode_id": seed.seed_id,
                            "turn_id": 0 if not seed.chat_history else 1,
                            "request_kind": decision.request_kind,
                            "trajectory_kind": seed.trajectory_kind,
                            "observation_origin": "none",
                            "teacher": {
                                "model": DASHSCOPE_MODEL,
                                "version": "2026-07",
                                "vendor": "dashscope",
                            },
                            "reviewer": {
                                "model": "planner-v3-schema-and-contract-gate",
                                "version": PLANNER_DECISION_V3_VERSION,
                                "vendor": "SoulTuner",
                            },
                            "reviewer_verdict": "accept",
                        },
                        "lineage": {
                            "collector": "collect_v4_teacher_pilot",
                            "collector_version": COLLECTOR_VERSION,
                        },
                    }
                )

    accepted_rows.sort(key=lambda row: row["meta"]["episode_id"])
    by_trajectory = Counter(
        row["meta"]["trajectory_kind"] for row in accepted_rows
    )
    by_kind = Counter(row["meta"]["request_kind"] for row in accepted_rows)
    summary = {
        "requested": len(seeds),
        "accepted": len(accepted_rows),
        "acceptance_rate": round(len(accepted_rows) / max(1, len(seeds)), 4),
        "failure_types": dict(sorted(failure_types.items())),
        "counts_by_trajectory": dict(sorted(by_trajectory.items())),
        "counts_by_request_kind": dict(sorted(by_kind.items())),
        "usage": dict(sorted(usage.items())),
        "batches": len(batches),
        "latency_ms": {
            "min": min(latencies) if latencies else None,
            "max": max(latencies) if latencies else None,
            "mean": round(sum(latencies) / len(latencies), 1) if latencies else None,
        },
        "teacher_model": DASHSCOPE_MODEL,
        "thinking_enabled": False,
        "collector_version": COLLECTOR_VERSION,
    }
    return accepted_rows, summary


def write_private(output: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    if "private" not in {part.casefold() for part in output.parts}:
        raise ValueError("teacher output must stay under a private directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-family", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/teacher/private/v4/teacher_pilot.jsonl"),
    )
    args = parser.parse_args()
    seeds = build_seeds(args.per_family)
    rows, summary = collect(
        seeds,
        api_key=_load_api_key(),
        batch_size=args.batch_size,
        workers=args.workers,
        timeout_seconds=args.timeout,
    )
    write_private(args.output, rows, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["acceptance_rate"] >= 0.8 else 2


if __name__ == "__main__":
    raise SystemExit(main())
