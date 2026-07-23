"""Generate a balanced pilot EPISODE set for distillation (Phase B).

The legacy 600 seeds are narrow ("play/download" style) and abandoned. This
generates fresh, training-appropriate episodes with a deliberate composition
that over-weights the behaviors the student most needs to learn — especially
HyDE-heavy vibe/mood/scenario/acoustic cases (~47%) — plus multi-turn
inheritance, memory use, memory conflict, and contradiction→clarification.

Uses the strong model as a user simulator to produce realistic, spoken-style
episodes. Output shape matches ``collect_episodes`` input:
    {"episode_id","category","profile","memories":[...],"turns":[...]}

Deduped and excluded against the frozen eval/holdout/blind cases so the pilot
never leaks the evaluation sets.

Usage:
    python -m data.sft.generate_pilot_episodes --target 1200 \
        --out data/sft/pilot_episodes.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("EVAL_DISABLE_SIDE_EFFECTS", "1")
os.environ.setdefault("TEACHER_LOG", "0")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pydantic import BaseModel, Field  # noqa: E402

# (category, weight, shape, guidance). shape drives the expected episode fields.
COMPOSITION: list[tuple[str, int, str, str]] = [
    ("specific_song", 12, "single", "点名具体歌曲或歌手要求播放/查找，中英日韩粤多语混合，含外语歌名"),
    ("artist_catalog", 8, "single", "问某歌手有哪些歌/代表作/某风格作品/早期作品"),
    ("genre_mood", 14, "single", "按流派叠加情绪推荐（治愈民谣/深情国摇/暗黑电子/燃系金属…），可含语言标签"),
    ("scenario", 12, "single", "按场景推荐（专注写代码/通勤地铁/睡前/健身/雨天/独自开夜路/约会做饭）带画面感"),
    ("vibe_only", 13, "single", "纯氛围/画面感/听感的模糊请求，不点名歌手不带硬标签（午后慵懒阳光/像浮在水里/清晨微凉）"),
    ("acoustic_specific", 8, "single", "具体声学需求（低动态少鼓/纯器乐无人声/女声突出/lo-fi 颗粒感/不刺耳的清亮吉他）"),
    ("negation", 8, "single", "带否定约束（不要太吵/别太苦情/避开 EDM/不要女团舞曲/别有说教感）"),
    ("timeliness_web", 4, "single", "最新/近期新歌/榜单/需要联网知识（某歌手今年新专、最近很火的那首）"),
    ("multi_turn", 12, "multi", "2-3 轮真实追问：首轮给个需求，后续微调（再安静点/换成英文/人声少一点/保留氛围换语言）"),
    ("profile_memory", 6, "memory", "给一段用户画像+1-2 条长期记忆，当前 query 与画像相关但需当前优先"),
    ("memory_conflict", 4, "memory_conflict", "画像/记忆写明长期偏好（如爱重金属），但当前 query 明确要相反的（今晚只想要安静）"),
    ("contradiction", 3, "single", "自相矛盾或不可同时满足（无歌词的说唱/纯音乐但要突出人声/既要冷门又要人人会唱）"),
]


class GenEpisode(BaseModel):
    profile: str = ""
    memories: list[str] = Field(default_factory=list)
    turns: list[str] = Field(default_factory=list)


class GenBatch(BaseModel):
    episodes: list[GenEpisode] = Field(default_factory=list)


def _normalize(text: str) -> str:
    return " ".join(str(text or "").split()).strip().casefold()


def _eval_exclusions() -> set[str]:
    from data.sft.seeds_from_legacy import _eval_case_exclusions

    return _eval_case_exclusions()


def _shape_instruction(shape: str, per_call: int) -> str:
    if shape == "single":
        return (
            f"生成 {per_call} 条**互不相同**的单轮请求。每条 episode 的 turns 只含 1 句用户口语，"
            'profile 与 memories 留空。输出 JSON：{"episodes":[{"turns":["..."]}, ...]}'
        )
    if shape == "multi":
        return (
            f"生成 {per_call} 个**互不相同**的多轮对话。每个 episode 的 turns 含 2-3 句：首轮是完整需求，"
            "后续是承接上一轮的微调追问（换语言/更安静/人声更少/保留氛围换风格）。profile 与 memories 留空。"
            '输出 JSON：{"episodes":[{"turns":["首轮","追问1","追问2"]}, ...]}'
        )
    if shape == "memory":
        return (
            f"生成 {per_call} 个 episode。每个含 profile（1 句用户长期画像）、memories（1-2 条历史偏好），"
            "turns 含 1 句与画像相关、但当前需求应优先的请求。"
            '输出 JSON：{"episodes":[{"profile":"...","memories":["..."],"turns":["..."]}, ...]}'
        )
    if shape == "memory_conflict":
        return (
            f"生成 {per_call} 个 episode。profile/memories 写明一个长期偏好（如常听重金属），"
            "turns 的当前请求明确要**相反**的东西（今晚只想安静）。目的是训练'当前压过长期记忆'。"
            '输出 JSON：{"episodes":[{"profile":"...","memories":["..."],"turns":["..."]}, ...]}'
        )
    raise ValueError(shape)


async def _generate_category(llm, category: str, guidance: str, shape: str, per_call: int, avoid: list[str]) -> list[GenEpisode]:
    from langchain_core.messages import HumanMessage, SystemMessage

    system = (
        "你是音乐推荐产品的用户模拟器。产出真实、口语化、多样的听歌请求，像真实用户随口说的话。"
        "不同长度、不同语言（含中英日韩粤混合）、不同人群。只产出 JSON，不要解释。"
    )
    human = (
        f"类别：{category}\n要求：{guidance}\n\n{_shape_instruction(shape, per_call)}\n"
        f"不要与这些重复：{json.dumps(avoid[:8], ensure_ascii=False)}"
    )
    try:
        structured = llm.with_structured_output(GenBatch, method="json_mode")
    except (TypeError, ValueError):
        structured = llm.with_structured_output(GenBatch)
    result = await structured.ainvoke([SystemMessage(content=system), HumanMessage(content=human)])
    batch = result if isinstance(result, GenBatch) else GenBatch.model_validate(result)
    return [ep for ep in batch.episodes if ep.turns and all(str(t).strip() for t in ep.turns)]


def _targets(total: int, only: set[str] | None = None) -> list[tuple[str, int, str, str]]:
    comp = [c for c in COMPOSITION if not only or c[0] in only]
    weight_sum = sum(w for _, w, _, _ in comp) or 1
    out = []
    for name, weight, shape, guidance in comp:
        out.append((name, max(1, round(total * weight / weight_sum)), shape, guidance))
    return out


async def generate(target: int, per_call: int, out_path: Path, only: set[str] | None = None, id_prefix: str = "") -> dict:
    from config.settings import settings
    from llms.chat_models import get_chat_model

    llm = get_chat_model(
        provider=settings.intent_llm_provider or settings.llm_default_provider,
        model_name=settings.intent_llm_model or settings.llm_default_model,
        temperature=0.9,
        max_tokens=2500,
    )
    excluded = _eval_exclusions()
    seen: set[str] = set()
    kept: list[dict] = []
    per_category: dict[str, int] = {}

    for name, want, shape, guidance in _targets(target, only):
        got: list[str] = []
        rounds = 0
        while per_category.get(name, 0) < want and rounds < 40:
            rounds += 1
            try:
                episodes = await _generate_category(llm, name, guidance, shape, min(per_call, want), got[-8:])
            except Exception as exc:
                print(f"{name} r{rounds}: FAIL {type(exc).__name__}")
                continue
            new = 0
            for ep in episodes:
                first = _normalize(ep.turns[0])
                if not first or first in seen or first in excluded:
                    continue
                seen.add(first)
                kept.append(
                    {
                        "episode_id": f"{id_prefix}{name}_{per_category.get(name, 0):04d}",
                        "category": name,
                        "profile": ep.profile.strip(),
                        "memories": [m.strip() for m in ep.memories if m.strip()],
                        "turns": [t.strip() for t in ep.turns if t.strip()],
                        "provenance": {"source_type": "pilot_synthetic", "category": name},
                    }
                )
                got.append(ep.turns[0])
                per_category[name] = per_category.get(name, 0) + 1
                new += 1
                if per_category[name] >= want:
                    break
            print(f"{name} r{rounds}: +{new} ({per_category.get(name, 0)}/{want})")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as sink:
        for ep in kept:
            sink.write(json.dumps(ep, ensure_ascii=False) + "\n")
    return {
        "target": target,
        "generated": len(kept),
        "by_category": per_category,
        "excluded_set": len(excluded),
        "out": str(out_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=int, default=1200)
    parser.add_argument("--per-call", type=int, default=20)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "data" / "sft" / "pilot_episodes.jsonl")
    parser.add_argument("--only", type=str, default="", help="Comma-separated category names to generate (subset of COMPOSITION)")
    parser.add_argument("--id-prefix", dest="id_prefix", type=str, default="", help="Namespace for episode_id (use e.g. 'supp_' for supplements to avoid id collision)")
    args = parser.parse_args()
    only = {c.strip() for c in args.only.split(",") if c.strip()} or None
    report = asyncio.run(generate(args.target, args.per_call, args.out, only=only, id_prefix=args.id_prefix))
    print(json.dumps({"summary": report}, ensure_ascii=False))
    return 0 if report["generated"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
