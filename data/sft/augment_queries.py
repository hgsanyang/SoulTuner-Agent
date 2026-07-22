"""Generate a diverse query set for teacher-trajectory collection (Phase 4).

The curated seed set has ~600 unique inputs; distillation wants 5k-10k
trajectories. This augments the query pool with the strong model itself:
diverse, realistic, spoken-style music requests across an intent x language x
scenario x difficulty taxonomy. Generated queries are deduped against the
seeds AND against the frozen eval/holdout/blind cases so SFT never leaks the
evaluation sets.

Usage:
    python -m data.sft.augment_queries --target 3000 --out data/sft/queries.jsonl
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

# (category, guidance) — the LLM fills each with realistic, varied queries.
CATEGORIES: list[tuple[str, str]] = [
    ("specific_song", "点名具体歌曲/歌手要求播放或查找，中英日韩粤多语言混合"),
    ("artist_catalog", "问某歌手有哪些歌/代表作/某风格作品"),
    ("genre_mood", "按流派或情绪推荐（摇滚/民谣/电子/治愈/emo/燃…）"),
    ("scenario", "按场景推荐（开车/通勤/专注写代码/睡前/健身/雨天/约会/派对/学习）"),
    ("vibe_vector", "自由文本氛围/画面感/声学质感的模糊请求（不点名歌手）"),
    ("negation", "带否定约束（不要太吵/别太悲伤/不要情歌/避开 EDM/不要女团舞曲）"),
    ("multi_turn_followup", "承接上一轮的微调追问（再安静一点/换成中文/保留氛围换个语言/人声少一点）"),
    ("timeliness_web", "最新/榜单/近期新歌/需要联网知识的请求"),
    ("contradiction_edge", "自相矛盾或边界请求（无歌词的说唱/纯音乐但要有人声/既要冷门又要耳熟）"),
    ("acoustic_specific", "具体声学需求（低动态/少鼓/纯器乐/女声突出/lo-fi beat/不刺耳）"),
]


class QueryBatch(BaseModel):
    queries: list[str] = Field(default_factory=list)


def _normalize(text: str) -> str:
    return " ".join(str(text or "").split()).strip().casefold()


def _load_exclusions() -> set[str]:
    """Seed inputs + frozen eval/holdout/blind queries — never generate these."""
    excluded: set[str] = set()
    try:
        from data.sft.generate_planner_sft import data as seed_data

        for item in seed_data:
            q = _normalize(item.get("input") or "")
            if q:
                excluded.add(q)
    except Exception:
        pass
    cases_dir = PROJECT_ROOT / "tests" / "eval" / "cases"
    for path in cases_dir.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        rows = payload if isinstance(payload, list) else payload.get("cases") or payload.get("examples") or []
        for row in rows:
            if isinstance(row, dict):
                q = _normalize(row.get("query") or row.get("input") or "")
                if q:
                    excluded.add(q)
    return excluded


async def _generate_category(llm, category: str, guidance: str, per_call: int, examples: list[str]) -> list[str]:
    from langchain_core.messages import HumanMessage, SystemMessage

    system = (
        "你是音乐推荐产品的用户模拟器。生成真实、口语化、多样的中文/多语言听歌请求，"
        "像真实用户随口说的话。不要编号、不要解释，只产出 JSON。"
    )
    human = (
        f"类别：{category}\n要求：{guidance}\n"
        f"生成 {per_call} 条**互不相同**的该类请求，风格口语化、长度不一、覆盖不同歌手/语言/场景。\n"
        f"不要与这些示例重复：{json.dumps(examples[:8], ensure_ascii=False)}\n"
        '输出 JSON：{"queries": ["...", "..."]}'
    )
    try:
        structured = llm.with_structured_output(QueryBatch, method="json_mode")
    except (TypeError, ValueError):
        structured = llm.with_structured_output(QueryBatch)
    result = await structured.ainvoke([SystemMessage(content=system), HumanMessage(content=human)])
    batch = result if isinstance(result, QueryBatch) else QueryBatch.model_validate(result)
    return [q.strip() for q in batch.queries if q and q.strip()]


async def augment(target: int, per_call: int, out_path: Path) -> dict:
    from llms.chat_models import get_chat_model
    from config.settings import settings

    llm = get_chat_model(
        provider=settings.intent_llm_provider or settings.llm_default_provider,
        model_name=settings.intent_llm_model or settings.llm_default_model,
        temperature=0.9,
        max_tokens=2000,
    )
    excluded = _load_exclusions()
    kept: list[str] = []
    seen: set[str] = set(excluded)
    rounds = 0
    while len(kept) < target and rounds < 200:
        rounds += 1
        for category, guidance in CATEGORIES:
            if len(kept) >= target:
                break
            try:
                queries = await _generate_category(llm, category, guidance, per_call, kept[-8:])
            except Exception as exc:
                print(f"round {rounds} {category}: FAIL {type(exc).__name__}")
                continue
            new = 0
            for q in queries:
                key = _normalize(q)
                if key and key not in seen:
                    seen.add(key)
                    kept.append(q)
                    new += 1
            print(f"round {rounds} {category}: +{new} (total {len(kept)})")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as sink:
        for q in kept[:target]:
            sink.write(json.dumps({"query": q}, ensure_ascii=False) + "\n")
    return {"generated": len(kept), "written": min(len(kept), target), "excluded_seeds": len(excluded), "out": str(out_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=int, default=3000)
    parser.add_argument("--per-call", type=int, default=25)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "data" / "sft" / "queries.jsonl")
    args = parser.parse_args()
    report = asyncio.run(augment(args.target, args.per_call, args.out))
    print(json.dumps({"summary": report}, ensure_ascii=False))
    return 0 if report["written"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
