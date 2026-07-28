"""Generate a small, curated V3 seed set for previously empty request kinds.

The seeds are authored examples, not synthetic teacher claims.  Every row keeps
its purpose, source, eligibility, ToolPlan contract, and deterministic lineage.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from schemas.planner_decision_v3 import PlannerDecisionV3  # noqa: E402
from schemas.tool_plan import ToolPlan  # noqa: E402


SEED_VERSION = "planner_v3_gap_seeds_2026_07_28"

CONVERSATION_QUERIES = [
    "你好，今天想随便聊聊音乐。",
    "谢谢你刚才的推荐。",
    "你觉得音乐为什么会让人想起过去？",
    "我今天只是想找个人说说话，不用推歌。",
    "晚安，明天再继续听。",
    "你会不会也有特别喜欢的旋律？",
    "先暂停推荐，我们聊聊最近的心情。",
    "这套系统是怎么理解我的音乐偏好的？",
]

LIBRARY_CASES = [
    ("查看我点赞的歌曲", "liked", ""),
    ("打开我的收藏歌曲", "saved", ""),
    ("我最近听过哪些歌？", "recent", ""),
    ("列出我明确不喜欢的歌曲", "disliked", ""),
    ("在我点赞的歌里找周杰伦", "liked", "周杰伦"),
    ("在收藏里搜索雨天", "saved", "雨天"),
    ("看看最近听过的陈奕迅", "recent", "陈奕迅"),
    ("检查我拉黑的 EDM 歌曲", "disliked", "EDM"),
]

ACQUISITION_CASES = [
    {
        "query": "帮我找到并暂存陈奕迅的《陀飞轮》，先不要正式入库。",
        "requirements": "陈奕迅 陀飞轮",
        "reason": "user requests a preview before ingest",
    },
    {
        "query": "把 Taylor Swift 的 cardigan 找到后放进待入库。",
        "requirements": "Taylor Swift cardigan",
        "reason": "user requests a named track for the ingest queue",
    },
    {
        "query": "搜索坂本龙一的 Merry Christmas Mr. Lawrence，确认版本后暂存。",
        "requirements": "坂本龙一 Merry Christmas Mr. Lawrence original",
        "reason": "version resolution is required before staging",
    },
    {
        "query": "把刚才推荐的那首 Plastic Panorama 暂存，音源暂时不用长期保留。",
        "requirements": "Postiljonen Plastic Panorama",
        "reason": "the referenced recommendation should be staged without permanent audio",
    },
    {
        "query": "找一份原版 Running Up That Hill，先进入待入库让我试听。",
        "requirements": "Kate Bush Running Up That Hill original",
        "reason": "the user explicitly asks for the original recording",
    },
    {
        "query": "把万能青年旅店的《杀死那个石家庄人》加入待处理列表。",
        "requirements": "万能青年旅店 杀死那个石家庄人",
        "reason": "a named song should be discovered and staged",
    },
    {
        "query": "找到 YOASOBI 的《アイドル》正式版后暂存，不要现场版。",
        "requirements": "YOASOBI アイドル official studio version",
        "reason": "the requested version constraint must be preserved",
    },
    {
        "query": "帮我找 Norah Jones 的 Don't Know Why，预览通过后再入库。",
        "requirements": "Norah Jones Don't Know Why original",
        "reason": "staging must remain a reversible preview",
    },
]


def _base_row(index: int, kind: str, query: str) -> dict[str, Any]:
    return {
        "episode_id": f"curated_{kind}_{index:03d}",
        "turn_id": 0,
        "current_query": query,
        "chat_history": "",
        "previous_plan": "",
        "profile_snapshot": "",
        "retrieved_memories": [],
        "available_tools": [
            "read_library",
            "search_external_music",
            "stage_ingest",
        ],
        "alignment_issues": [],
        "provenance": {
            "source_type": "curated_v3_gate_seed",
            "authorship": "manual_contract_example",
            "seed_version": SEED_VERSION,
            "decision_schema_version": "planner_decision_v3",
            "tool_schema_version": "1.1",
            "data_purpose": "planner_v3_sft",
            "training_eligible": True,
        },
        "training_governance": {
            "training_eligible": True,
            "data_purpose": "planner_v3_sft",
            "source_type": "curated_v3_gate_seed",
            "seed_version": SEED_VERSION,
        },
    }


def build_seeds() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for index, query in enumerate(CONVERSATION_QUERIES, 1):
        decision = PlannerDecisionV3(
            request_kind="conversation",
            response_mode="answer",
            decision_summary="非检索对话",
        )
        plan = ToolPlan(
            request_mode="conversation",
            tool_calls=[],
            decision_summary="conversation response; no tools",
            max_replans=0,
        )
        rows.append(
            {
                **_base_row(index, "conversation", query),
                "teacher_decision_v3": decision.model_dump(mode="json", exclude_none=True),
                "tool_plan": plan.model_dump(mode="json", exclude_none=True),
            }
        )

    for index, (query, collection, text_query) in enumerate(LIBRARY_CASES, 1):
        decision = PlannerDecisionV3(
            request_kind="library",
            response_mode="answer",
            tool_names=["library"],
            decision_summary="读取用户曲库",
        )
        plan = ToolPlan.model_validate(
            {
                "request_mode": "library",
                "tool_calls": [
                    {
                        "id": "library_read",
                        "name": "read_library",
                        "arguments": {
                            "collection": collection,
                            "query": text_query,
                            "limit": 30,
                        },
                        "reason": "read the server-bound user's requested collection",
                    }
                ],
                "decision_summary": "bounded read-only library query",
                "max_replans": 0,
            }
        )
        rows.append(
            {
                **_base_row(index, "library", query),
                "teacher_decision_v3": decision.model_dump(mode="json", exclude_none=True),
                "tool_plan": plan.model_dump(mode="json", exclude_none=True),
            }
        )

    for index, case in enumerate(ACQUISITION_CASES, 1):
        decision = PlannerDecisionV3(
            request_kind="acquisition",
            response_mode="answer",
            tool_names=["web", "ingest"],
            decision_summary="发现并预演待入库",
        )
        plan = ToolPlan.model_validate(
            {
                "request_mode": "acquisition",
                "tool_calls": [
                    {
                        "id": "external_discovery",
                        "name": "search_external_music",
                        "arguments": {
                            "requirements": case["requirements"],
                            "limit": 5,
                        },
                        "reason": "find a bounded set of candidates",
                    },
                    {
                        "id": "ingest_preview",
                        "name": "stage_ingest",
                        "arguments": {
                            "candidate_source_ids": [],
                            "preserve_audio": False,
                            "reason": case["reason"],
                            "mode": "preview",
                        },
                        "depends_on": ["external_discovery"],
                        "reason": "validate a staging proposal without applying side effects",
                    },
                ],
                "decision_summary": "external discovery followed by shadow ingest preview",
                "max_replans": 0,
            }
        )
        rows.append(
            {
                **_base_row(index, "acquisition", case["query"]),
                "teacher_decision_v3": decision.model_dump(mode="json", exclude_none=True),
                "tool_plan": plan.model_dump(mode="json", exclude_none=True),
            }
        )

    return rows


def generate(output: Path, summary_output: Path) -> dict[str, Any]:
    rows = build_seeds()
    if len({row["episode_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate seed episode_id")
    if len({row["current_query"].casefold() for row in rows}) != len(rows):
        raise ValueError("duplicate seed query")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    distribution = Counter(
        row["teacher_decision_v3"]["request_kind"] for row in rows
    )
    summary = {
        "seed_version": SEED_VERSION,
        "records": len(rows),
        "request_kind": dict(sorted(distribution.items())),
        "training_eligible": sum(
            bool(row["training_governance"]["training_eligible"]) for row in rows
        ),
        "output": str(output),
    }
    summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/sft/curated_v3_gap_seeds.jsonl"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("data/sft/curated_v3_gap_seeds_summary.json"),
    )
    args = parser.parse_args()
    summary = generate(args.output, args.summary_output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
