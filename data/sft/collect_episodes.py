"""Episode-based teacher collection for distillation (Phase A-2).

The single-query collector could not teach multi-turn inheritance, reference
resolution, current-intent-vs-long-term-memory conflict, memory silence, or
recovery — the planner was always run with empty context. This collects
EPISODES: multi-turn conversations threaded through the real dialog state, with
an injected profile snapshot and retrieved memories, so the teacher decision
for each turn reflects the true conditioning.

The training TARGET is the compact PlannerDecisionV2 (schemas/planner_decision),
not the verbose MusicQueryPlan. Full text is written to a private directory that
is git-ignored; only sanitized, consented trajectories may later enter training.

Episode spec input (jsonl), one per line:
    {"episode_id": "...", "profile": "偏好文本", "memories": ["..."],
     "turns": ["第一句", "追问1", "追问2"]}

Usage:
    python -m data.sft.collect_episodes --episodes data/sft/episodes.jsonl \
        --out data/teacher/private/episodes_collected.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("EVAL_DISABLE_SIDE_EFFECTS", "1")
os.environ.setdefault("TEACHER_LOG", "0")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Default output goes to a PRIVATE, git-ignored directory (full teacher text).
DEFAULT_OUT = PROJECT_ROOT / "data" / "teacher" / "private" / "episodes_collected.jsonl"


def load_episodes(path: Path) -> list[dict]:
    episodes: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("turns"):
            episodes.append(obj)
    return episodes


def _plan_summary(plan) -> str:
    """Compact previous-plan text threaded into the next turn's planner call."""
    rp = plan.retrieval_plan
    parts = [f"intent={plan.intent_type}"]
    if rp.hard_constraints.artist_entities:
        parts.append("artist=" + ",".join(rp.hard_constraints.artist_entities))
    if rp.hints.genres:
        parts.append("genre=" + ",".join(rp.hints.genres))
    if rp.hints.mood:
        parts.append(f"mood={rp.hints.mood}")
    if rp.hints.scenario:
        parts.append(f"scenario={rp.hints.scenario}")
    if rp.vector_acoustic_queries:
        parts.append("acoustic=" + rp.vector_acoustic_queries[0][:80])
    return "; ".join(parts)


async def collect(episodes: list[dict], out_path: Path, case_timeout: float) -> dict:
    from agent.intent.planner import UNIFIED_PLANNER_PROMPT_VERSION, IntentPlanner
    from llms.multi_llm import get_intent_chat_model
    from schemas.planner_decision import decision_token_estimate, from_query_plan
    from schemas.tool_plan import TOOL_PLAN_VERSION, tool_plan_alignment_issues
    from config.settings import settings

    planner = IntentPlanner(get_intent_chat_model)
    provider = settings.intent_llm_provider or settings.llm_default_provider
    model_name = settings.intent_llm_model or settings.llm_default_model

    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    failed = 0
    started = time.time()
    with out_path.open("w", encoding="utf-8") as sink:
        for episode in episodes:
            episode_id = str(episode.get("episode_id") or f"ep_{written}")
            profile = str(episode.get("profile") or "无")
            memories = list(episode.get("memories") or [])
            memory_text = "；".join(str(m) for m in memories) if memories else ""
            history_lines: list[str] = []
            previous_plan_text = ""
            for turn_index, query in enumerate(episode["turns"]):
                query = str(query or "").strip()
                if not query:
                    continue
                turn_started = time.perf_counter()
                try:
                    plan = await asyncio.wait_for(
                        planner.plan(
                            user_input=query,
                            user_preferences=profile,
                            chat_history="\n".join(history_lines[-6:]),
                            previous_plan=previous_plan_text,
                            graphzep_facts=memory_text,
                            user_id="__episode_collect__",
                        ),
                        timeout=case_timeout if case_timeout > 0 else None,
                    )
                except Exception as exc:
                    failed += 1
                    print(f"[{episode_id} t{turn_index}] FAIL {query[:32]}: {type(exc).__name__}")
                    break
                decision = from_query_plan(plan)
                record = {
                    "episode_id": episode_id,
                    "turn_id": turn_index,
                    "current_query": query,
                    "previous_plan": previous_plan_text,
                    "profile_snapshot": profile if profile != "无" else "",
                    "retrieved_memories": memories,
                    "available_tools": sorted({call.name.value for call in (plan.tool_plan.tool_calls or [])}),
                    "teacher_decision": decision.model_dump(mode="json", exclude_none=True),
                    "target_token_estimate": decision_token_estimate(decision),
                    "alignment_issues": tool_plan_alignment_issues(plan),
                    "provenance": {
                        "source_type": "synthetic",
                        "teacher_model": model_name,
                        "teacher_provider": provider,
                        "planner_prompt_version": UNIFIED_PLANNER_PROMPT_VERSION,
                        "tool_schema_version": TOOL_PLAN_VERSION,
                        "decision_schema_version": "planner_decision_v2",
                        "collected_at": int(time.time()),
                        "latency_ms": round((time.perf_counter() - turn_started) * 1000, 1),
                    },
                }
                sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                sink.flush()
                written += 1
                history_lines.append(f"用户: {query}")
                history_lines.append(f"助手: [{plan.intent_type}]")
                previous_plan_text = _plan_summary(plan)
                print(f"[{episode_id} t{turn_index}] OK {plan.intent_type} tok~{record['target_token_estimate']} {query[:28]}")

    return {
        "episodes": len(episodes),
        "turns_written": written,
        "failed": failed,
        "elapsed_s": round(time.time() - started, 1),
        "out": str(out_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--case-timeout", type=float, default=120.0)
    args = parser.parse_args()
    episodes = load_episodes(args.episodes)
    if not episodes:
        print("no episodes loaded")
        return 1
    print(f"collecting {len(episodes)} episodes -> {args.out}")
    report = asyncio.run(collect(episodes, args.out, args.case_timeout))
    print(json.dumps({"summary": report}, ensure_ascii=False))
    return 0 if report["turns_written"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
