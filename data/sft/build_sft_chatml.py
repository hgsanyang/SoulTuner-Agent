"""Build ChatML SFT data from collected episodes (Phase B).

Turns verified teacher episodes into the exact (system, user, assistant) triples
the student model is fine-tuned on. The target is the compact PlannerDecisionV2
JSON — NOT the verbose planner output — so the student learns a small, fast
output. The `system` prompt is a condensed (~800 token) instruction, so student
inference stays cheap; the deterministic compiler expands the output downstream.

The `user` message serializes the SAME conditioning the teacher saw (profile,
long-term memory, dialog history, previous plan, current query), so the student's
training input matches its production inference input.

Input: a VERIFIED-CLEAN episode jsonl (output of verify_episodes --write-clean).
Output: train/eval ChatML jsonl, stratified by decision intent.

Usage:
    python -m data.sft.build_sft_chatml --in data/teacher/private/legacy_100_clean.jsonl \
        --train data/sft/train_v2_chatml.jsonl --eval data/sft/eval_v2_chatml.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Condensed student system prompt for the PlannerDecisionV2 compact target.
# Deliberately short: the heavy few-shot planner prompt is for the API teacher;
# the student only needs the schema + the non-derivable decision rules.
STUDENT_SYSTEM_PROMPT = """你是音乐推荐系统的决策器。读用户上下文，输出一个严格 JSON 决策（PlannerDecisionV2），不要任何多余文字。

## intent（选一个）
graph_search（实体/硬属性精确查找）| hybrid_search（有标签又有听感）| vector_search（纯情绪/氛围）| web_search（时效/最新）| clarification（指代无法解析、严重矛盾、请求过空泛时反问）| general_chat（闲聊）| acquire_music（确认下载）| recommend_by_favorites（查收藏）

## 输出 JSON 字段
- intent: 上述之一
- hard: {artist:[], song:[], language:null|Chinese/English/Japanese/Korean/Cantonese, region:null, instrumental:false}
- soft: {goal:"", trajectory:"", vibe:[], avoid:[]}
- hints: {mood:[], scenario:[], genre:[]}
- metadata: {era:null, release_year_from:null, release_year_to:null, recency_required:false, external_knowledge_required:false}
- acoustic_queries: []  // hybrid/vector 时填 1-4 条英文声学描述（MusicCaps 风格：乐器/速度/音色/能量/氛围；禁歌手名与精确 BPM）
- tool_names: []  // 用到的召回通道：graph / dense / web
- clarification: null  // 仅当 intent=clarification 时填反问文本
- decision_summary: ""  // 不超过 30 字结论

## 硬规则
1. 语种（中文/英文/日语/粤语）与"纯音乐/器乐/无人声"是硬约束，必须写入 hard.language / hard.instrumental，不能只靠声学描述——文搜音模型听不出语种。
2. hybrid_search / vector_search 必须给 acoustic_queries；graph_search 不填。
3. 当前请求永远优先于长期画像/记忆；画像只作排序偏好，别写进 hard 约束（除非当前输入明确提到）。
4. 指代（那首/上一首/这个歌手）能从历史或上轮解析就解析并继承，解析不了才 clarification。
5. 严重矛盾（如"纯音乐但突出人声"）用 clarification 反问，不要硬猜。"""


def _load_clean(path: Path) -> list[dict]:
    records: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def build_user_message(rec: dict) -> str:
    """Serialize the same conditioning the teacher saw (omit empty sections)."""
    parts: list[str] = []
    profile = str(rec.get("profile_snapshot") or "").strip()
    if profile:
        parts.append(f"[用户画像] {profile}")
    memories = [str(m).strip() for m in (rec.get("retrieved_memories") or []) if str(m).strip()]
    if memories:
        parts.append("[长期记忆] " + "；".join(memories))
    history = str(rec.get("chat_history") or "").strip()
    if history:
        parts.append("[对话历史]\n" + history)
    previous = str(rec.get("previous_plan") or "").strip()
    if previous:
        parts.append(f"[上轮检索计划] {previous}")
    parts.append(f"[当前输入] {rec.get('current_query') or ''}")
    return "\n".join(parts)


def to_chatml(rec: dict) -> dict:
    target = json.dumps(rec["teacher_decision"], ensure_ascii=False, separators=(",", ":"))
    return {
        "messages": [
            {"role": "system", "content": STUDENT_SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(rec)},
            {"role": "assistant", "content": target},
        ],
        "meta": {
            "episode_id": rec.get("episode_id"),
            "turn_id": rec.get("turn_id"),
            "intent": (rec.get("teacher_decision") or {}).get("intent"),
            "source_type": (rec.get("provenance") or {}).get("source_type"),
        },
    }


def stratified_split(records: list[dict], eval_frac: float, seed: int) -> tuple[list[dict], list[dict]]:
    random.seed(seed)
    groups: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        intent = (rec.get("teacher_decision") or {}).get("intent") or "unknown"
        groups[intent].append(rec)
    train: list[dict] = []
    eval_: list[dict] = []
    for intent, items in groups.items():
        random.shuffle(items)
        n_eval = max(1, round(len(items) * eval_frac)) if len(items) > 1 else 0
        eval_.extend(items[:n_eval])
        train.extend(items[n_eval:])
    random.shuffle(train)
    random.shuffle(eval_)
    return train, eval_


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="inp", type=Path, required=True)
    parser.add_argument("--train", type=Path, default=PROJECT_ROOT / "data" / "sft" / "train_v2_chatml.jsonl")
    parser.add_argument("--eval", type=Path, default=PROJECT_ROOT / "data" / "sft" / "eval_v2_chatml.jsonl")
    parser.add_argument("--eval-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = _load_clean(args.inp)
    if not records:
        print("no records")
        return 1
    train, eval_ = stratified_split(records, args.eval_frac, args.seed)

    for split, path in ((train, args.train), (eval_, args.eval)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as sink:
            for rec in split:
                sink.write(json.dumps(to_chatml(rec), ensure_ascii=False) + "\n")

    def dist(recs: list[dict]) -> dict:
        return dict(Counter((r.get("teacher_decision") or {}).get("intent") for r in recs))

    summary = {
        "input": str(args.inp),
        "records": len(records),
        "train": len(train),
        "eval": len(eval_),
        "train_intent_dist": dist(train),
        "eval_intent_dist": dist(eval_),
        "train_out": str(args.train),
        "eval_out": str(args.eval),
    }
    print(json.dumps({"summary": summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
