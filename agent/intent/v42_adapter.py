"""Adapter from the SoulTuner V4.2 planner contract to the production query plan.

The fine-tuned model emits a compact lane-policy decision.  Production code
never executes that JSON directly: the shared deterministic guard validates or
replaces it, and this module compiles the guarded decision into the existing
``MusicQueryPlan`` / ``ToolPlan`` boundary.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from deploy.self_hosted_35b.planner_guard import guard_candidate, parse_candidate_content
from deploy.self_hosted_35b.prompt_v42 import (
    STUDENT_SYSTEM_PROMPT_V4_2,
    format_student_user_message,
)
from schemas.query_plan import MusicQueryPlan
from schemas.tool_plan import ToolPlan

from .adapters import PlannerPayload, _model_name, _openai_base_url, _post_json


V42_CONTRACT = "v42"
LEGACY_CONTRACT = "legacy"
AUTO_CONTRACT = "auto"
SUPPORTED_CONTRACTS = {AUTO_CONTRACT, LEGACY_CONTRACT, V42_CONTRACT}
V42_PROMPT_VERSION = "soultuner_planner_v4_2_guarded_v1"


def uses_v42_contract(provider: str, model_name: str, configured: str = AUTO_CONTRACT) -> bool:
    """Return whether an endpoint should use the compact V4.2 contract."""

    contract = str(configured or AUTO_CONTRACT).strip().lower()
    if contract not in SUPPORTED_CONTRACTS:
        raise ValueError(
            f"Unsupported planner contract {configured!r}; expected auto, legacy, or v42"
        )
    if contract == V42_CONTRACT:
        return True
    if contract == LEGACY_CONTRACT:
        return False

    provider_key = str(provider or "").strip().lower()
    model_key = str(model_name or "").strip().lower().replace("_", "-")
    is_soultuner_v42 = "soultuner" in model_key and (
        "v4.2" in model_key or "v42" in model_key
    )
    return provider_key in {"vllm", "sglang"} and is_soultuner_v42


def build_v42_user_message(
    payload: PlannerPayload,
    reference_context: Mapping[str, str] | None = None,
) -> str:
    """Serialize production context in the same public format used for SFT."""

    reference = dict(reference_context or {})
    return format_student_user_message(
        payload.user_input,
        profile_snapshot=payload.user_preferences,
        retrieved_memories=payload.retrieved_memories,
        chat_history=payload.chat_history,
        previous_plan=payload.previous_plan,
        reference_title=str(reference.get("reference_title") or ""),
        reference_artist=str(reference.get("reference_artist") or ""),
    )


def _strings(value: Any, *, limit: int = 30) -> list[str]:
    if not isinstance(value, list):
        return []
    cleaned = [str(item or "").strip() for item in value]
    return list(dict.fromkeys(item for item in cleaned if item))[:limit]


def compile_guarded_v42(user_input: str, guarded: Mapping[str, Any]) -> MusicQueryPlan:
    """Compile a guarded V4.2 decision into the production planner protocol."""

    policy = dict(guarded.get("lane_policy") or {})
    hard = dict(guarded.get("hard") or {})
    soft = dict(guarded.get("soft") or {})
    hints = dict(guarded.get("hints") or {})
    metadata = dict(guarded.get("metadata") or {})
    evidence = dict(guarded.get("evidence") or {})
    graph_mode = str(policy.get("graph") or "off")
    dense_mode = str(policy.get("dense") or "off")
    web_mode = str(policy.get("web") or "off")
    brief_reason = str(evidence.get("brief_reason") or "").strip()

    task_mode = str(guarded.get("task_mode") or "recommendation")
    dialogue_mode = guarded.get("dialogue_mode")
    response_mode = str(guarded.get("response_mode") or "answer")
    clarification = str(guarded.get("clarification") or "").strip()

    if response_mode == "clarify":
        return MusicQueryPlan.model_validate({
            "intent_type": "clarification",
            "parameters": {
                "query": user_input,
                "question": clarification or "请再具体说明一下你的音乐需求。",
                "reason": brief_reason,
                "options": [],
            },
            "context": clarification or brief_reason,
            "reasoning": brief_reason,
        })

    if task_mode == "dialogue" and dialogue_mode in {"chat", "library_guidance"}:
        return MusicQueryPlan.model_validate({
            "intent_type": "general_chat",
            "parameters": {"query": user_input, "dialogue_mode": dialogue_mode},
            "context": brief_reason,
            "reasoning": brief_reason,
        })

    if task_mode == "dialogue" and dialogue_mode == "information":
        intent_type = "graph_search"
    elif graph_mode != "off" and dense_mode == "required":
        intent_type = "hybrid_search"
    elif dense_mode == "required":
        intent_type = "vector_search"
    else:
        intent_type = "graph_search"

    artist_entities = _strings(hard.get("artist"))
    song_entities = _strings(hard.get("song"))
    genres = _strings(hints.get("genre"))
    moods = _strings(hints.get("mood"))
    scenarios = _strings(hints.get("scenario"))
    vibe = _strings(soft.get("vibe"), limit=12)
    acoustic_queries = _strings(guarded.get("acoustic_queries"), limit=4)
    graph_entities = list(dict.fromkeys([*artist_entities, *song_entities]))

    plan = MusicQueryPlan.model_validate({
        "intent_type": intent_type,
        "parameters": {
            "query": user_input,
            "entities": graph_entities,
            "planner_contract": V42_CONTRACT,
            "lane_policy": policy,
            "execution_profile": (guarded.get("execution") or {}).get("profile"),
        },
        "context": brief_reason,
        "reasoning": brief_reason,
        "retrieval_plan": {
            "use_graph": graph_mode != "off",
            "hard_constraints": {
                "artist_entities": artist_entities,
                "song_entities": song_entities,
                "language": hard.get("language"),
                "region": hard.get("region"),
                "instrumental": bool(hard.get("instrumental", False)),
            },
            "soft_intent": {
                "goal": str(soft.get("goal") or "").strip(),
                "trajectory": str(soft.get("trajectory") or "").strip(),
                "avoid": _strings(soft.get("avoid"), limit=12),
                "vibe": "；".join(vibe),
            },
            "hints": {
                "genres": genres,
                "mood": moods[0] if moods else None,
                "scenario": scenarios[0] if scenarios else None,
            },
            "metadata_constraints": {
                "release_year_from": metadata.get("release_year_from"),
                "release_year_to": metadata.get("release_year_to"),
                "era": metadata.get("era"),
                "recency_required": bool(metadata.get("recency_required", False)),
                "external_knowledge_required": bool(
                    metadata.get("external_knowledge_required", False)
                ),
            },
            "use_vector": dense_mode == "required",
            "vector_acoustic_query": acoustic_queries[0] if acoustic_queries else None,
            "vector_acoustic_queries": acoustic_queries,
            "use_web_search": web_mode != "off",
            "web_search_keywords": user_input if web_mode != "off" else "",
        },
    })
    if dialogue_mode == "information" and plan.tool_plan is not None:
        plan.tool_plan = ToolPlan.model_validate({
            **plan.tool_plan.model_dump(mode="json"),
            "request_mode": "information",
        })
    return plan


async def plan_with_v42_contract(
    llm: Any,
    payload: PlannerPayload,
    *,
    max_tokens: int,
    timeout: float,
    api_key: str = "",
    reference_context: Mapping[str, str] | None = None,
) -> tuple[MusicQueryPlan, list[str]]:
    """Call an OpenAI-compatible endpoint and return a guarded production plan."""

    user_message = build_v42_user_message(payload, reference_context)
    request_body = {
        "model": _model_name(llm, "soultuner-planner-v4.2-35b"),
        "messages": [
            {"role": "system", "content": STUDENT_SYSTEM_PROMPT_V4_2},
            {"role": "user", "content": user_message},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
        "enable_thinking": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    response = await _post_json(
        f"{_openai_base_url(llm)}/chat/completions",
        request_body,
        timeout=timeout,
        headers=headers,
    )
    content = response["choices"][0]["message"]["content"]
    try:
        candidate = parse_candidate_content(content)
    except (ValueError, TypeError, json.JSONDecodeError):
        candidate = None
    context = dict(reference_context or {})
    guarded, findings = guard_candidate(payload.user_input, candidate, context)
    return compile_guarded_v42(payload.user_input, guarded), findings
