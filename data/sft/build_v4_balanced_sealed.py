"""Build a five-way, entity-disjoint candidate pool for blind V4 review."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
from typing import Any

from data.sft.build_v4_sealed_seeds import load_jsonl, row_entities
from schemas.planner_decision import DecisionHard, DecisionMeta, DecisionSoft
from schemas.planner_decision_v3 import PlannerDecisionV3


PER_KIND_DEFAULT = 125


def _decision(row: dict[str, Any]) -> PlannerDecisionV3:
    assistant = [m for m in row.get("messages") or [] if m.get("role") == "assistant"]
    return PlannerDecisionV3.model_validate_json(str(assistant[-1]["content"]))


def _usable_teacher_row(row: dict[str, Any]) -> bool:
    try:
        decision = _decision(row)
    except Exception:
        return False
    lanes = set(decision.tool_names)
    if ("dense" in lanes) != bool(decision.acoustic_queries):
        return False
    return not any(any("\u4e00" <= char <= "\u9fff" for char in query) for query in decision.acoustic_queries)


def _candidate_meta(kind: str, index: int, trajectory: str) -> dict[str, Any]:
    return {
        "seed_source": "curated_seed",
        "episode_id": f"sealed_v4_{kind}_{index:04d}",
        "turn_id": 0,
        "request_kind": kind,
        "trajectory_kind": trajectory,
        "observation_origin": "none",
        "teacher": {"model": "sealed-contract-author", "version": "1", "vendor": "SoulTuner"},
        "reviewer": {"model": "pending-independent-review", "version": "0", "vendor": "pending"},
    }


def _generated_row(
    system: str,
    kind: str,
    index: int,
    query: str,
    decision: PlannerDecisionV3,
    entity: str = "",
    *,
    trajectory: str | None = None,
    turn_id: int = 0,
) -> dict[str, Any]:
    trajectory = trajectory or {
        "library": "library_state",
        "acquisition": "acquisition",
        "conversation": "conversation",
    }.get(kind, "single_turn")
    meta = _candidate_meta(kind, index, trajectory)
    meta["turn_id"] = turn_id
    user_content = query if query.lstrip().startswith("[") else f"[当前输入] {query}"
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": decision.model_dump_json(exclude_none=True)},
        ],
        "meta": meta,
        "lineage": {"builder": "build_v4_balanced_sealed", "entity": entity},
    }


def _recommendation_candidates(
    system: str,
    sources: list[dict[str, Any]],
    artists: list[str],
    per_kind: int,
) -> list[dict[str, Any]]:
    """Interleave five behaviours so the first reviewed 100 stay balanced."""
    if not sources:
        raise ValueError("sealed teacher pool has no usable recommendation rows")
    output: list[dict[str, Any]] = []
    for index in range(per_kind):
        artist = artists[index % len(artists)]
        mode = index % 5
        if mode == 0:
            row = copy.deepcopy(sources[(index // 5) % len(sources)])
            row["meta"].update(_candidate_meta("recommendation", index, "single_turn"))
            output.append(row)
            continue
        if mode == 1:
            query = (
                f"[对话历史]\n用户: 今晚先围绕 {artist} 找些克制的夜间音乐。\n"
                "助手: [recommendation]\n"
                f"[上轮检索计划] artist={artist}; scene=late-night\n"
                "[当前输入] 接着刚才的范围，但把鼓点收住，保留一点向前的律动。"
            )
            decision = PlannerDecisionV3(
                request_kind="recommendation",
                tool_names=["graph", "dense"],
                hard=DecisionHard(artist=[artist]),
                soft=DecisionSoft(
                    goal="继承明确的歌手范围并应用当前听感修正",
                    vibe=["restrained late-night momentum"],
                    avoid=["heavy drums", "aggressive"],
                ),
                acoustic_queries=[
                    "Restrained late-night music with a soft pulse, light drums, and calm forward motion."
                ],
                decision_summary="继承可恢复的上轮约束，以当前修正为最高优先级",
            )
            output.append(
                _generated_row(
                    system,
                    "recommendation",
                    index,
                    query,
                    decision,
                    artist,
                    trajectory="multi_turn_inheritance",
                    turn_id=1 + index % 4,
                )
            )
            continue
        if mode == 2:
            query = (
                f"[长期记忆] 最近常听 {artist}，平时偏爱强节奏和现场感。\n"
                "[当前输入] 今天有点头疼，不沿用那些高能偏好，只要低动态、柔和、别抢注意力。"
            )
            decision = PlannerDecisionV3(
                request_kind="recommendation",
                tool_names=["dense"],
                soft=DecisionSoft(
                    goal="当前身体状态覆盖冲突的长期高能偏好",
                    avoid=["loud", "driving", "heavy drums"],
                ),
                acoustic_queries=[
                    "Very soft low-dynamic music with gentle transients, sparse percussion, and an unobtrusive calm mood."
                ],
                decision_summary="当前明确需求优先，长期偏好只作非冲突背景",
            )
            output.append(
                _generated_row(
                    system,
                    "recommendation",
                    index,
                    query,
                    decision,
                    artist,
                    trajectory="memory_vs_current_request",
                    turn_id=1 + index % 4,
                )
            )
            continue
        if mode == 3:
            query = f"[当前输入] 就放 {artist} 刚才那个版本，我这里没有上一轮记录。"
            decision = PlannerDecisionV3(
                request_kind="recommendation",
                response_mode="clarify",
                hard=DecisionHard(artist=[artist]),
                clarification="你指的是哪首歌，以及录音室版、现场版还是其他版本？",
                decision_summary="关键指代无法从现有上下文恢复",
            )
            output.append(
                _generated_row(
                    system,
                    "recommendation",
                    index,
                    query,
                    decision,
                    artist,
                    trajectory="clarification_positive",
                )
            )
            continue

        query = f"[当前输入] 以 {artist} 为范围，找柔和但不困倦、节奏清楚但不吵的歌。"
        decision = PlannerDecisionV3(
            request_kind="recommendation",
            tool_names=["graph", "dense"],
            hard=DecisionHard(artist=[artist]),
            soft=DecisionSoft(
                goal="约束已经足够明确，直接检索而不是过度澄清",
                vibe=["gentle but alert"],
                avoid=["harsh", "sleepy"],
            ),
            acoustic_queries=[
                "Gentle but alert music with a clear restrained rhythm, soft texture, and no harsh peaks."
            ],
            decision_summary="直接执行可同时满足的软约束",
        )
        output.append(
            _generated_row(
                system,
                "recommendation",
                index,
                query,
                decision,
                artist,
                trajectory="clarification_negative",
            )
        )
    return output


def build_candidates(teacher_rows: list[dict[str, Any]], per_kind: int = PER_KIND_DEFAULT) -> list[dict[str, Any]]:
    if per_kind < 100:
        raise ValueError("per_kind must leave room for a 100-row reviewed sealed class")
    system = next(m["content"] for m in teacher_rows[0]["messages"] if m.get("role") == "system")
    entities = []
    artist_entities = []
    for row in teacher_rows:
        artists, songs = row_entities(row)
        artist_entities.extend(artists)
        entities.extend(artists + songs)
        entity = str((row.get("lineage") or {}).get("entity") or "").strip()
        if entity:
            entities.append(entity)
    entities = list(dict.fromkeys(value for value in entities if value))
    artist_entities = list(dict.fromkeys(value for value in artist_entities if value))
    if len(entities) < 100:
        raise ValueError("sealed teacher pool has too few unseen entities")
    if len(artist_entities) < 100:
        raise ValueError("sealed teacher pool has too few unseen artists")

    output = []
    teacher_by_kind: dict[str, list[dict[str, Any]]] = {kind: [] for kind in ("recommendation", "information")}
    for row in teacher_rows:
        kind = str((row.get("meta") or {}).get("request_kind") or "")
        if kind in teacher_by_kind and _usable_teacher_row(row):
            teacher_by_kind[kind].append(row)

    output.extend(
        _recommendation_candidates(
            system,
            teacher_by_kind["recommendation"],
            artist_entities,
            per_kind,
        )
    )

    for kind in ("information",):
        for index, source in enumerate(teacher_by_kind[kind][:per_kind]):
            row = copy.deepcopy(source)
            row["meta"].update(_candidate_meta(kind, index, "single_turn"))
            output.append(row)

    existing_information = len(teacher_by_kind["information"][:per_kind])
    for index in range(existing_information, per_kind):
        artist = entities[index % len(entities)]
        query = f"请核实 {artist} 的活跃年代、地区与代表风格；这是资料问题，不要直接生成歌单。"
        decision = PlannerDecisionV3(
            request_kind="information",
            tool_names=["graph", "web"],
            hard=DecisionHard(artist=[artist]),
            metadata=DecisionMeta(external_knowledge_required=True),
            decision_summary="本地知识优先，缺失时外部查证",
        )
        output.append(_generated_row(system, "information", index, query, decision, artist))

    for index in range(per_kind):
        entity = artist_entities[index % len(artist_entities)]
        acquisition = PlannerDecisionV3(
            request_kind="acquisition",
            tool_names=["web", "ingest"],
            hard=DecisionHard(artist=[entity]),
            metadata=DecisionMeta(external_knowledge_required=True),
            decision_summary="发现可播放正式版本并进入可撤销的入库预演",
        )
        output.append(_generated_row(system, "acquisition", index, f"查找 {entity} 最近发行的一首正式录音，解析到可播放版本后先放进待入库让我确认。", acquisition, entity))

        library = PlannerDecisionV3(
            request_kind="library",
            tool_names=["library"],
            hard=DecisionHard(artist=[entity]),
            decision_summary="只读查询用户曲库",
        )
        output.append(_generated_row(system, "library", index, f"只查我的点赞和收藏里有没有 {entity}，不要修改任何记录。", library, entity))

        conversation = PlannerDecisionV3(
            request_kind="conversation",
            decision_summary="非检索对话，不调用工具",
        )
        output.append(_generated_row(system, "conversation", index, f"先不要推歌。聊聊为什么人会在第 {index + 1} 次重听时听见不同的情绪。", conversation))

    counts = Counter((row.get("meta") or {}).get("request_kind") for row in output)
    if any(counts.get(kind, 0) < per_kind for kind in ("recommendation", "information", "acquisition", "library", "conversation")):
        raise ValueError(f"sealed candidate pool is incomplete: {dict(counts)}")
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-candidates", type=Path, default=Path("data/teacher/private/v4/sealed_v4_teacher_candidate.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/teacher/private/v4/sealed_v4_balanced_candidates.jsonl"))
    parser.add_argument("--per-kind", type=int, default=PER_KIND_DEFAULT)
    args = parser.parse_args()
    rows = build_candidates(load_jsonl(args.teacher_candidates), args.per_kind)
    if "private" not in {part.casefold() for part in args.output.parts}:
        raise ValueError("sealed candidates must stay private")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "counts": dict(Counter(row["meta"]["request_kind"] for row in rows)), "output": str(args.output)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
