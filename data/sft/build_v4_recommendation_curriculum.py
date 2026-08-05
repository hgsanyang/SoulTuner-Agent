"""Build the recommendation half of the balanced Planner V4 curriculum.

These are contract-owned examples, not claimed user traffic.  They target the
three behaviours that were sparse in V3: multi-turn refinement, current intent
overriding a long-term preference, and clarification boundaries.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from data.sft.build_v4_contract_curriculum import entity_pool
from data.sft.build_v4_sealed_seeds import load_jsonl
from schemas.planner_decision import DecisionHard, DecisionSoft
from schemas.planner_decision_v3 import PlannerDecisionV3


VERSION = "v4_recommendation_curriculum_1"
DEFAULT_COUNTS = {
    "multi_turn_inheritance": 1300,
    "memory_vs_current_request": 1200,
    "clarification_positive": 200,
    "clarification_negative": 211,
}

VIBES = (
    ("雨天安静", "Rainy evening music with soft dynamics, sparse drums, warm intimate vocals, and a calm reflective mood."),
    ("夜间开车", "Confident night-driving music with a steady pulse, spacious production, and controlled forward energy."),
    ("专注工作", "Unobtrusive focus music with a stable groove, restrained dynamics, and minimal vocal distraction."),
    ("失落但不沉溺", "Bittersweet music that acknowledges sadness while gradually becoming warmer and more hopeful."),
    ("清晨醒来", "Gentle morning music with clear acoustic textures, light rhythm, and quietly optimistic energy."),
    ("运动热身", "Rhythmic workout music with a clean build, firm drums, and energetic but non-aggressive momentum."),
    ("深夜独处", "Intimate late-night music with close vocals, muted instrumentation, and a private contemplative atmosphere."),
    ("周末做饭", "Relaxed cooking music with an easy groove, bright organic instruments, and a sociable warm mood."),
)

REFINEMENTS = (
    ("再柔和一点，但保留刚才的雨天感。", ["harsh", "aggressive"], "soft, low-dynamic, rain-soaked"),
    ("这次少一点人声，节奏别完全消失。", ["vocal-forward"], "mostly instrumental with a restrained pulse"),
    ("延续刚才的歌手范围，不过换成更早期的作品。", [], "earlier catalog with the same artist focus"),
    ("别那么悲伤，慢慢往明亮一点走。", ["despairing"], "bittersweet with a gradual hopeful lift"),
    ("保留夜路氛围，但不要太躁。", ["aggressive", "chaotic"], "controlled night-drive momentum"),
    ("再冷门一点，整体听感不要变。", [], "same sonic character with less familiar selections"),
)

MEMORY_PROFILES = (
    "长期偏好：常听摇滚和现场录音，也喜欢强鼓点。",
    "长期偏好：喜欢华语民谣、温暖男声和叙事歌词。",
    "长期偏好：收藏很多电子舞曲，周末常听高能量歌单。",
    "长期偏好：偏爱欧美独立和阴郁氛围。",
)

CURRENT_OVERRIDES = (
    ("但我今天头疼，只想听非常安静、低动态的音乐。", ["loud", "energetic", "driving"], "Very quiet low-dynamic music with soft transients and a soothing unobtrusive atmosphere."),
    ("这会儿在开车，先不要民谣，来点稳健有推进感的。", ["fragile", "sleepy"], "Steady road music with clear rhythm, controlled energy, and a confident forward flow."),
    ("今晚想休息，不要舞曲，也不要强烈鼓点。", ["dance", "party", "heavy drums"], "Restful evening music with sparse percussion, soft texture, and low physical intensity."),
    ("现在情绪已经很低了，别再推太丧的，想慢慢缓过来。", ["despairing", "devastating"], "Bittersweet but gently uplifting music that moves from sadness toward warmth."),
)

CLARIFY_POSITIVE = (
    ("帮我放 {artist} 的那个。", "你指的是这位歌手的哪首歌？"),
    ("还是 {artist} 刚才那种，不过换一个。", "我这里没有可继承的具体听感，你想保留哪种感觉？"),
    ("给我来点适合现在的 {artist}，但我没说现在是什么状态。", "你现在是什么心情或场景？"),
    ("放 {artist} 那首歌。", "这位歌手有很多作品，你想听哪一首或哪个时期？"),
    ("听 {artist}，但必须完全安静而且最炸最吵。", "这两个方向冲突，你更优先要安静还是高强度？"),
    ("就按她上一首那样，可能是 {artist} 也可能不是。", "我还不知道“她”和“上一首”分别指什么，可以补充一下吗？"),
    ("来点 {artist} 那个年代的。", "你说的是哪个年代？"),
    ("找 {artist} 的一个版本给我。", "你要找哪首歌，以及原版、现场版还是翻唱版？"),
)

CLARIFY_NEGATIVE = (
    "沿用上轮的雨天氛围，再安静一些。",
    "只要陈奕迅的粤语歌，偏早期一点。",
    "现在在开车，来点有推进感但别太躁的。",
    "我有点难过，想听抒情但不要继续下沉的歌。",
    "从我的收藏里找周杰伦的歌。",
    "推荐低动态、少鼓、偏器乐的专注音乐。",
)


def _system_prompt(rows: list[dict[str, Any]]) -> str:
    for message in rows[0].get("messages") or []:
        if message.get("role") == "system":
            return str(message.get("content") or "")
    raise ValueError("training source has no system prompt")


def _row(
    *,
    index: int,
    trajectory: str,
    user_content: str,
    decision: PlannerDecisionV3,
    system_prompt: str,
    turn_id: int,
    trope: str = "",
) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": decision.model_dump_json(exclude_none=True)},
        ],
        "meta": {
            "seed_source": "template_expansion",
            "episode_id": f"v4_recommendation_{trajectory}_{index:05d}",
            "turn_id": turn_id,
            "request_kind": "recommendation",
            "trajectory_kind": trajectory,
            "observation_origin": "none",
            "teacher": {"model": "deterministic-planner-contract", "version": "1", "vendor": "SoulTuner"},
            "reviewer": {"model": "schema-and-behavior-tests", "version": "1", "vendor": "SoulTuner"},
            "reviewer_verdict": "accept",
        },
        "lineage": {
            "builder": "build_v4_recommendation_curriculum",
            "builder_version": VERSION,
            **({"clarification_trope": trope} if trope else {}),
        },
    }


def build_rows(train_rows: list[dict[str, Any]], counts: dict[str, int] | None = None) -> list[dict[str, Any]]:
    requested = {**DEFAULT_COUNTS, **(counts or {})}
    artists, _ = entity_pool(train_rows)
    system_prompt = _system_prompt(train_rows)
    output: list[dict[str, Any]] = []

    for index in range(requested["multi_turn_inheritance"]):
        artist = artists[index % len(artists)]
        vibe_name, acoustic = VIBES[index % len(VIBES)]
        refinement, avoid, refined_vibe = REFINEMENTS[index % len(REFINEMENTS)]
        user = (
            f"[对话历史]\n用户: 想听 {artist} 附近的作品，适合{vibe_name}。\n"
            "助手: [recommendation]\n"
            f"[上轮检索计划] artist={artist}; vibe={vibe_name}\n[当前输入] {refinement}"
        )
        decision = PlannerDecisionV3(
            request_kind="recommendation",
            tool_names=["graph", "dense"],
            hard=DecisionHard(artist=[artist]),
            soft=DecisionSoft(goal="继承上一轮范围并应用当前微调", vibe=[refined_vibe], avoid=avoid),
            acoustic_queries=[acoustic],
            decision_summary="继承可解析的上轮约束，以当前修正为最高优先级",
        )
        output.append(_row(index=index, trajectory="multi_turn_inheritance", user_content=user, decision=decision, system_prompt=system_prompt, turn_id=1 + index % 5))

    for index in range(requested["memory_vs_current_request"]):
        artist = artists[index % len(artists)]
        profile = MEMORY_PROFILES[(index // len(artists)) % len(MEMORY_PROFILES)]
        current, avoid, acoustic = CURRENT_OVERRIDES[
            (index // (len(artists) * len(MEMORY_PROFILES))) % len(CURRENT_OVERRIDES)
        ]
        user = f"[长期记忆] 最近常听 {artist}；{profile}\n[当前输入] {current}"
        decision = PlannerDecisionV3(
            request_kind="recommendation",
            tool_names=["dense"],
            soft=DecisionSoft(goal="优先满足当前状态，长期偏好只作弱背景", avoid=avoid),
            acoustic_queries=[acoustic],
            decision_summary="当前明确需求覆盖冲突的长期偏好",
        )
        output.append(_row(index=index, trajectory="memory_vs_current_request", user_content=user, decision=decision, system_prompt=system_prompt, turn_id=1 + index % 5))

    for index in range(requested["clarification_positive"]):
        artist = artists[index % len(artists)]
        query_template, question = CLARIFY_POSITIVE[index % len(CLARIFY_POSITIVE)]
        query = query_template.format(artist=artist)
        decision = PlannerDecisionV3(
            request_kind="recommendation",
            response_mode="clarify",
            clarification=question,
            decision_summary="缺少无法从上下文恢复的关键指代或约束冲突",
        )
        trope = ("referent", "missing_context", "underspecified", "entity_ambiguity", "hard_conflict", "pronoun", "era", "version")[index % len(CLARIFY_POSITIVE)]
        output.append(_row(index=index, trajectory="clarification_positive", user_content=f"[当前输入] {query}", decision=decision, system_prompt=system_prompt, turn_id=0, trope=trope))

    for index in range(requested["clarification_negative"]):
        artist = artists[index % len(artists)]
        query = f"{CLARIFY_NEGATIVE[index % len(CLARIFY_NEGATIVE)]} 参考歌手范围是 {artist}。"
        vibe_name, acoustic = VIBES[index % len(VIBES)]
        decision = PlannerDecisionV3(
            request_kind="recommendation",
            tool_names=["graph", "dense"],
            hard=DecisionHard(artist=[artist]),
            soft=DecisionSoft(goal="需求足够明确，直接执行", vibe=[vibe_name]),
            acoustic_queries=[acoustic],
            decision_summary="上下文已足够，不应过度澄清",
        )
        output.append(_row(index=index, trajectory="clarification_negative", user_content=f"[当前输入] {query}", decision=decision, system_prompt=system_prompt, turn_id=0, trope="answerable_request"))

    ids = [row["meta"]["episode_id"] for row in output]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate recommendation curriculum episode ids")
    users = [next(message["content"] for message in row["messages"] if message["role"] == "user") for row in output]
    if len(users) != len(set(users)):
        raise ValueError("duplicate recommendation curriculum inputs")
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("data/teacher/private/v4/train_v3_repaired.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/teacher/private/v4/recommendation_curriculum.jsonl"))
    for key, value in DEFAULT_COUNTS.items():
        parser.add_argument(f"--{key.replace('_', '-')}-count", type=int, default=value)
    args = parser.parse_args()
    counts = {key: int(getattr(args, f"{key}_count")) for key in DEFAULT_COUNTS}
    rows = build_rows(load_jsonl(args.train), counts)
    if "private" not in {part.casefold() for part in args.output.parts}:
        raise ValueError("curriculum output must stay private")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "trajectory": dict(Counter(row["meta"]["trajectory_kind"] for row in rows)), "output": str(args.output)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
