"""Build balanced, deterministic V4 rows for contract-level request kinds.

Conversation, library reads, acquisition staging, and factual information have
an unambiguous lane contract. They do not need an expensive teacher call for
every surface-form variation. Complex multi-turn, memory, and clarification
examples remain strong-teacher data and are deliberately out of scope here.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.sft.build_v4_sealed_seeds import (  # noqa: E402
    canonical_entity,
    load_jsonl,
    row_entities,
)
from schemas.planner_decision import (  # noqa: E402
    DecisionHard,
    DecisionMeta,
    DecisionSoft,
)
from schemas.planner_decision_v3 import PlannerDecisionV3  # noqa: E402


VERSION = "v4_contract_curriculum_2026_07_29"
DEFAULT_COUNTS = {
    "conversation": 800,
    "library": 900,
    "acquisition": 900,
    "information": 1000,
}

CONVERSATION_TEMPLATES = (
    "今天先不推歌，想聊聊音乐为什么会勾起记忆：{topic}。",
    "暂停推荐，我们聊聊{topic}和听歌习惯之间的关系。",
    "谢谢刚才的陪伴。现在只想说说{topic}，不用调用工具。",
    "我不想找歌，只想听你谈谈{topic}。",
    "先把播放器放一边：你觉得{topic}为什么会影响情绪？",
    "今晚不需要歌单，陪我聊一会儿{topic}。",
    "这个问题不需要检索：{topic}对你来说意味着什么？",
    "推荐先暂停，我想讨论一下{topic}。",
)
TOPICS = (
    "童年听过的旋律",
    "现场演出带来的共同感",
    "失恋后反复听同一首歌",
    "歌词和个人经历的联系",
    "深夜独自听歌的安全感",
    "音乐品味会不会随年龄改变",
    "为什么某些和弦让人怀旧",
    "通勤路上听歌的陪伴感",
    "语言不通时仍然会被声音打动",
    "专辑顺序是否会改变理解",
)
CONVERSATION_ANGLES = (
    "从个人经历的角度",
    "从声音记忆的角度",
    "从情绪调节的角度",
    "从文化背景的角度",
    "从现场与录音差异的角度",
    "从歌词叙事的角度",
    "从习惯形成的角度",
    "从人与人分享音乐的角度",
    "从年龄变化的角度",
    "从注意力与环境的角度",
)

LIBRARY_TEMPLATES = (
    "在我的{collection}里查找和 {artist} 有关的歌。",
    "只查看我的{collection}，筛出 {artist} 的作品。",
    "打开{collection}，搜索关键词 {artist}，不要修改曲库。",
    "我以前{action}过哪些 {artist} 的歌？",
    "从{collection}中列出和 {artist} 相关的记录。",
    "检查我的{collection}里是否已有 {artist}。",
)
COLLECTIONS = (
    ("点赞", "喜欢"),
    ("收藏", "保存"),
    ("最近播放", "播放"),
    ("不喜欢", "明确拒绝"),
)

ACQUISITION_TEMPLATES = (
    "找到 {artist} 的《{song}》，先暂存到待入库，不要直接永久保存音源。",
    "搜索 {artist}《{song}》的正式版，确认后放进待处理队列。",
    "把 {artist} 的《{song}》作为入库候选，先让我试听。",
    "查找《{song}》的 {artist} 原版，只做可撤销的入库预演。",
    "请发现并暂存 {artist}《{song}》，排除现场和翻唱版本。",
    "将 {artist} 的《{song}》加入待入库，但现在不要执行不可逆操作。",
)
ACQUISITION_QUALIFIERS = (
    "优先正式录音室版本",
    "核对歌手与专辑信息",
    "排除同名翻唱和现场版",
    "保留原始音质，不做不必要转码",
)

INFORMATION_TEMPLATES = (
    "{artist} 的主要音乐风格和活跃年代是什么？本地资料不足再联网。",
    "介绍一下 {artist} 的乐队背景、代表时期和风格变化。",
    "{artist} 来自哪里，职业生涯中有哪些重要阶段？",
    "我想了解 {artist} 的音乐脉络，不是要立即生成歌单。",
    "查询 {artist} 的代表流派和主要成就，已有知识卡优先。",
    "{artist} 大致在哪些年代活跃，声音风格后来有什么变化？",
    "请核对 {artist} 的背景资料和创作风格，缺证据时再查外部来源。",
    "告诉我 {artist} 的成员或个人经历，以及这些经历如何影响作品。",
)


def _unique(values: Iterable[str]) -> list[str]:
    output = []
    seen = set()
    for value in values:
        text = str(value or "").strip()
        key = canonical_entity(text)
        if text and key not in seen:
            seen.add(key)
            output.append(text)
    return output


def entity_pool(
    train_rows: list[dict[str, Any]],
) -> tuple[list[str], list[tuple[str, str]]]:
    artists = []
    song_pairs = []
    for row in train_rows:
        row_artists, row_songs = row_entities(row)
        artists.extend(row_artists)
        if row_artists and row_songs:
            for artist in row_artists[:1]:
                for song in row_songs[:1]:
                    song_pairs.append((artist, song))
    unique_artists = _unique(artists)
    pair_seen = set()
    unique_pairs = []
    for artist, song in song_pairs:
        key = (canonical_entity(artist), canonical_entity(song))
        if all(key) and key not in pair_seen:
            pair_seen.add(key)
            unique_pairs.append((artist, song))
    if not unique_artists:
        raise ValueError("training source contains no artist entities")
    if not unique_pairs:
        raise ValueError("training source contains no artist/song pairs")
    return unique_artists, unique_pairs


def _row(
    index: int,
    kind: str,
    trajectory: str,
    query: str,
    decision: PlannerDecisionV3,
    system_prompt: str,
) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"[当前输入] {query}"},
            {
                "role": "assistant",
                "content": decision.model_dump_json(exclude_none=True),
            },
        ],
        "meta": {
            "seed_source": "template_expansion",
            "episode_id": f"v4_contract_{kind}_{index:05d}",
            "turn_id": 0,
            "request_kind": decision.request_kind,
            "trajectory_kind": trajectory,
            "observation_origin": "none",
            "teacher": {
                "model": "deterministic-request-contract",
                "version": "1",
                "vendor": "SoulTuner",
            },
            "reviewer": {
                "model": "planner-v3-schema-and-contract-tests",
                "version": "1",
                "vendor": "SoulTuner",
            },
            "reviewer_verdict": "accept",
        },
        "lineage": {
            "builder": "build_v4_contract_curriculum",
            "builder_version": VERSION,
        },
    }


def build_rows(
    train_rows: list[dict[str, Any]],
    *,
    counts: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    requested = {**DEFAULT_COUNTS, **(counts or {})}
    artists, song_pairs = entity_pool(train_rows)
    first_messages = train_rows[0].get("messages") or []
    system_prompt = next(
        (
            str(message.get("content") or "")
            for message in first_messages
            if message.get("role") == "system"
        ),
        "你是 SoulTuner 的 Planner。只输出严格的 PlannerDecisionV3 JSON。",
    )
    rows = []

    for index in range(requested["conversation"]):
        topic = TOPICS[index % len(TOPICS)]
        angle = CONVERSATION_ANGLES[
            (index // len(TOPICS)) % len(CONVERSATION_ANGLES)
        ]
        query = CONVERSATION_TEMPLATES[
            (index // (len(TOPICS) * len(CONVERSATION_ANGLES)))
            % len(CONVERSATION_TEMPLATES)
        ].format(topic=f"{topic}（{angle}）")
        decision = PlannerDecisionV3(
            request_kind="conversation",
            response_mode="answer",
            decision_summary="非检索对话，不调用工具",
        )
        rows.append(
            _row(index, "conversation", "conversation", query, decision, system_prompt)
        )

    for index in range(requested["library"]):
        artist = artists[index % len(artists)]
        collection, action = COLLECTIONS[
            (index // len(artists)) % len(COLLECTIONS)
        ]
        query = LIBRARY_TEMPLATES[
            (index // (len(artists) * len(COLLECTIONS))) % len(LIBRARY_TEMPLATES)
        ].format(collection=collection, action=action, artist=artist)
        decision = PlannerDecisionV3(
            request_kind="library",
            response_mode="answer",
            tool_names=["library"],
            hard=DecisionHard(artist=[artist]),
            decision_summary="只读用户个人曲库",
        )
        rows.append(
            _row(index, "library", "library_state", query, decision, system_prompt)
        )

    for index in range(requested["acquisition"]):
        artist, song = song_pairs[index % len(song_pairs)]
        qualifier = ACQUISITION_QUALIFIERS[
            (index // len(song_pairs)) % len(ACQUISITION_QUALIFIERS)
        ]
        query = ACQUISITION_TEMPLATES[
            (index // (len(song_pairs) * len(ACQUISITION_QUALIFIERS)))
            % len(ACQUISITION_TEMPLATES)
        ].format(artist=artist, song=song) + f"；{qualifier}。"
        decision = PlannerDecisionV3(
            request_kind="acquisition",
            response_mode="answer",
            tool_names=["web", "ingest"],
            hard=DecisionHard(artist=[artist], song=[song]),
            metadata=DecisionMeta(external_knowledge_required=True),
            decision_summary="外部发现后进行可撤销的入库预演",
        )
        rows.append(
            _row(index, "acquisition", "acquisition", query, decision, system_prompt)
        )

    for index in range(requested["information"]):
        artist = artists[index % len(artists)]
        query = INFORMATION_TEMPLATES[
            (index // max(1, len(artists))) % len(INFORMATION_TEMPLATES)
        ].format(artist=artist)
        decision = PlannerDecisionV3(
            request_kind="information",
            response_mode="answer",
            tool_names=["graph", "web"],
            hard=DecisionHard(artist=[artist]),
            soft=DecisionSoft(goal="查询歌手背景、风格、年代和可核验事实"),
            metadata=DecisionMeta(external_knowledge_required=True),
            decision_summary="本地知识优先，缺失时外部补证",
        )
        rows.append(
            _row(index, "information", "single_turn", query, decision, system_prompt)
        )

    ids = [row["meta"]["episode_id"] for row in rows]
    inputs = [row["messages"][1]["content"].casefold() for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate curriculum episode ids")
    if len(inputs) != len(set(inputs)):
        raise ValueError("duplicate curriculum inputs")
    return rows


def write_rows(output: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if "private" not in {part.casefold() for part in output.parts}:
        raise ValueError("curriculum output must stay private")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    by_kind = Counter(row["meta"]["request_kind"] for row in rows)
    return {
        "rows": len(rows),
        "counts_by_request_kind": dict(sorted(by_kind.items())),
        "output": str(output),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train",
        type=Path,
        default=Path("data/teacher/private/v4/train_v3_repaired.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/teacher/private/v4/contract_curriculum.jsonl"),
    )
    for kind, count in DEFAULT_COUNTS.items():
        parser.add_argument(f"--{kind}-count", type=int, default=count)
    args = parser.parse_args()
    counts = {
        kind: int(getattr(args, f"{kind}_count"))
        for kind in DEFAULT_COUNTS
    }
    rows = build_rows(load_jsonl(args.train), counts=counts)
    print(json.dumps(write_rows(args.output, rows), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
