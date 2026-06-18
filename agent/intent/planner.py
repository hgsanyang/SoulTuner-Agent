"""Unified intent planner independent from LangGraph orchestration."""

from __future__ import annotations

import os
from datetime import date
from typing import Any, Callable

from config.logging_config import get_logger
from config.settings import settings
from llms.prompts import LOCAL_PLANNER_PROMPT, UNIFIED_PLANNER_HUMAN, UNIFIED_PLANNER_SYSTEM
from retrieval.gssc_context_builder import build_context
from schemas.query_plan import MusicQueryPlan

from .adapters import (
    PlannerPayload,
    plan_with_dashscope,
    plan_with_generic_structured_output,
    plan_with_local_structured_output,
    plan_with_sglang,
)

logger = get_logger(__name__)
LOCAL_PROVIDERS = {"sglang", "vllm", "ollama"}
MUSIC_REQUEST_CUES = (
    "歌",
    "音乐",
    "听",
    "推荐",
    "来几首",
    "来点",
    "playlist",
    "song",
    "music",
)


def apply_routing_guardrails(plan: MusicQueryPlan, user_input: str) -> MusicQueryPlan:
    """Prevent explicit music requests from being rounded into general chat."""
    normalized_input = user_input.lower()
    if plan.intent_type == "general_chat" and any(cue in normalized_input for cue in MUSIC_REQUEST_CUES):
        plan.intent_type = "vector_search"
        plan.parameters = {"query": user_input, "entities": []}
        plan.context = plan.context or "模糊音乐推荐"
        plan.reasoning = "明确求歌，向量兜底"
        plan.retrieval_plan.use_graph = False
        plan.retrieval_plan.use_vector = True
        plan.retrieval_plan.use_web_search = False
        plan.retrieval_plan.vector_acoustic_query = (
            plan.retrieval_plan.vector_acoustic_query or user_input
        )
        plan.retrieval_plan.soft_intent.vibe = (
            plan.retrieval_plan.soft_intent.vibe or user_input
        )
        logger.info("[IntentPlanner] guardrail corrected general_chat to vector_search")
    return plan


class IntentPlanner:
    """Select a provider adapter and return one validated query plan."""

    def __init__(self, llm_factory: Callable[[], Any]):
        self._llm_factory = llm_factory

    async def plan(
        self,
        *,
        user_input: str,
        user_preferences: str,
        chat_history: str,
        previous_plan: str,
        graphzep_facts: str = "",
    ) -> MusicQueryPlan:
        if os.getenv("MUSIC_MOCK_MODE", "0").lower() in {"1", "true", "yes"}:
            return MusicQueryPlan.model_validate({
                "intent_type": "vector_search",
                "parameters": {"query": user_input, "entities": []},
                "context": "mock mode",
                "retrieval_plan": {
                    "use_graph": False,
                    "use_vector": True,
                    "soft_intent": {"vibe": user_input},
                    "vector_acoustic_query": user_input,
                },
                "reasoning": "mock mode",
            })

        llm = self._llm_factory()
        provider = (settings.intent_llm_provider or settings.llm_default_provider).lower()
        model_name = (
            getattr(llm, "model_name", "")
            or settings.intent_llm_model
            or settings.llm_default_model
        )
        context = await build_context(
            graphzep_facts=graphzep_facts,
            chat_history=chat_history,
            total_budget=0,
        )
        payload = PlannerPayload(
            user_input=user_input,
            user_preferences=user_preferences,
            chat_history=context["chat_history"],
            previous_plan=previous_plan,
            current_date=str(date.today()),
        )
        logger.info("[IntentPlanner] provider=%s model=%s", provider, model_name)

        if provider == "sglang":
            plan = await plan_with_sglang(
                llm,
                LOCAL_PLANNER_PROMPT,
                payload,
                max_tokens=settings.intent_max_tokens,
                timeout=settings.llm_timeout,
            )
        elif provider in LOCAL_PROVIDERS:
            plan = await plan_with_local_structured_output(llm, LOCAL_PLANNER_PROMPT, payload)
        elif provider == "dashscope":
            plan = await plan_with_dashscope(
                api_key=os.getenv("DASHSCOPE_API_KEY", ""),
                model_name=model_name or "qwen3.7-plus",
                system_prompt=UNIFIED_PLANNER_SYSTEM,
                human_prompt=UNIFIED_PLANNER_HUMAN,
                payload=payload,
                max_tokens=settings.intent_max_tokens,
                timeout=settings.llm_timeout,
                base_url=os.getenv(
                    "DASHSCOPE_BASE_URL",
                    "https://dashscope.aliyuncs.com/compatible-mode/v1",
                ),
            )
        else:
            plan = await plan_with_generic_structured_output(
                llm,
                UNIFIED_PLANNER_SYSTEM,
                UNIFIED_PLANNER_HUMAN,
                payload,
            )
        return apply_routing_guardrails(plan, user_input)
