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
1. 语种（中文/英文/日语/粤语）是硬过滤属性，必须写入 hard.language——文搜音模型听不出语种，只能靠图谱硬过滤。
2. 纯音乐/器乐/无人声是声学意图，不靠稀疏标签硬排除：置 hard.instrumental=true 作信号，并在 acoustic_queries 里明确写 "purely instrumental, no vocals"，由声学召回+排序满足。
3. hybrid_search / vector_search 必须给 acoustic_queries；graph_search 不填。
4. tool_names 是权威的召回通道选择，必须与 intent 自洽（vector/hybrid 含 dense，graph 含 graph，web 含 web，clarification/general_chat 为空）；纯 web 资讯可只 web，实体查询可 graph+web 补充。
5. 当前请求永远优先于长期画像/记忆；画像只作排序偏好，别写进 hard 约束（除非当前输入明确提到）。
6. 指代（那首/上一首/这个歌手）能从历史或上轮解析就解析并继承，解析不了才 clarification。
7. 严重矛盾（如"纯音乐但突出人声"）用 clarification 反问，不要硬猜。"""


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
    # Split by SEED FAMILY, never by turn: a multi-turn conversation — and every
    # augmentation/rewrite sharing a parent_seed_id — must live entirely in one
    # split, or context ability is evaluated on leaked/near-duplicate history.
    by_family: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        by_family[_family_key(rec)].append(rec)
    for recs in by_family.values():
        recs.sort(key=lambda r: (str(r.get("episode_id")), r.get("turn_id", 0)))
    strata: dict[str, list[list[dict]]] = defaultdict(list)
    for recs in by_family.values():
        intent = (recs[-1].get("teacher_decision") or {}).get("intent") or "unknown"
        strata[intent].append(recs)
    train: list[dict] = []
    eval_: list[dict] = []
    for intent, episodes in sorted(strata.items()):
        random.shuffle(episodes)
        n_eval = max(1, round(len(episodes) * eval_frac)) if len(episodes) > 1 else 0
        for recs in episodes[:n_eval]:
            eval_.extend(recs)
        for recs in episodes[n_eval:]:
            train.extend(recs)
    random.shuffle(train)
    random.shuffle(eval_)
    return train, eval_


def _family_key(rec: dict) -> str:
    """Split unit: the seed family (parent_seed_id) so rewrites never split;
    falls back to episode_id for data collected before lineage was recorded."""
    prov = rec.get("provenance") or {}
    return str(prov.get("parent_seed_id") or rec.get("episode_id"))


def _split_audit(train: list[dict], eval_: list[dict]) -> dict:
    """Overlap audit — episode, seed-family, query-hash must ALL be zero."""
    def eids(recs: list[dict]) -> set[str]:
        return {str(r.get("episode_id")) for r in recs}

    def families(recs: list[dict]) -> set[str]:
        return {_family_key(r) for r in recs}

    def qhashes(recs: list[dict]) -> set[str]:
        return {" ".join(str(r.get("current_query") or "").split()).casefold() for r in recs}

    ep_overlap = eids(train) & eids(eval_)
    fam_overlap = families(train) & families(eval_)
    q_overlap = qhashes(train) & qhashes(eval_)
    q_overlap.discard("")
    return {
        "episode_overlap": len(ep_overlap),
        "seed_family_overlap": len(fam_overlap),
        "query_overlap": len(q_overlap),
        "episode_overlap_examples": sorted(ep_overlap)[:5],
    }


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

    # Fail-closed: never write leaked splits.
    audit = _split_audit(train, eval_)
    if audit["episode_overlap"] or audit["seed_family_overlap"] or audit["query_overlap"]:
        print(json.dumps({"ABORT_leakage": audit}, ensure_ascii=False, indent=2))
        return 1

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
        "leakage_audit": audit,
        "train_intent_dist": dist(train),
        "eval_intent_dist": dist(eval_),
        "train_out": str(args.train),
        "eval_out": str(args.eval),
    }
    print(json.dumps({"summary": summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
