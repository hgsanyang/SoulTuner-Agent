"""Unified intent planner independent from LangGraph orchestration."""

from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
import os
from datetime import date
import time
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
UNIFIED_PLANNER_PROMPT_VERSION = "unified_planner_2026_07_10"
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
HK_REGION_ALIASES = {"hk", "hong kong", "香港", "hongkong"}
CANTONESE_CONTEXT_CUES = (
    "港乐",
    "港樂",
    "粤语",
    "粵語",
    "cantopop",
    "cantonese",
)
MANDARIN_CONTEXT_CUES = ("国语", "國語", "普通话", "mandarin")


class PlannerResultCache:
    """Small process-local TTL/LRU cache for validated planner outputs."""

    def __init__(
        self,
        ttl_seconds: int = 300,
        max_entries: int = 256,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.ttl_seconds = max(0, int(ttl_seconds))
        self.max_entries = max(1, int(max_entries))
        self._clock = clock
        self._items: OrderedDict[str, tuple[float, MusicQueryPlan]] = OrderedDict()

    @staticmethod
    def make_key(
        *,
        user_input: str,
        user_preferences: str,
        chat_history: str,
        previous_plan: str,
        graphzep_facts: str,
        provider: str,
        model_name: str,
        current_date: str,
        user_id: str = "",
    ) -> str:
        profile_context = json.dumps(
            [user_preferences, chat_history, previous_plan, graphzep_facts],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        profile_hash = hashlib.sha256(profile_context.encode("utf-8")).hexdigest()
        material = "\0".join(
            [user_input.strip(), profile_hash, provider, model_name, current_date, user_id]
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def get(self, key: str) -> MusicQueryPlan | None:
        if self.ttl_seconds <= 0:
            return None
        entry = self._items.get(key)
        if entry is None:
            return None
        expires_at, plan = entry
        if self._clock() >= expires_at:
            self._items.pop(key, None)
            return None
        self._items.move_to_end(key)
        return plan.model_copy(deep=True)

    def put(self, key: str, plan: MusicQueryPlan) -> None:
        if self.ttl_seconds <= 0:
            return
        self._items[key] = (
            self._clock() + self.ttl_seconds,
            plan.model_copy(deep=True),
        )
        self._items.move_to_end(key)
        while len(self._items) > self.max_entries:
            self._items.popitem(last=False)


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
        if not plan.retrieval_plan.vector_acoustic_queries:
            plan.retrieval_plan.vector_acoustic_queries = [plan.retrieval_plan.vector_acoustic_query]
        plan.retrieval_plan.soft_intent.vibe = (
            plan.retrieval_plan.soft_intent.vibe or user_input
        )
        logger.info("[IntentPlanner] guardrail corrected general_chat to vector_search")
    return plan


def normalize_planner_grounding(plan: MusicQueryPlan, user_input: str) -> MusicQueryPlan:
    """Normalize source-grounded music constraints without replacing LLM intent.

    The LLM remains responsible for intent understanding.  This helper only
    fixes a narrow catalog-grounding gap observed in eval: "港乐/Cantopop"
    often comes back as region=Hong Kong without the Cantonese language
    preference, which makes downstream language checks and ranking too weak.
    """
    rp = plan.retrieval_plan
    hard = rp.hard_constraints
    text = " ".join(
        [
            user_input,
            str(rp.graph_region_filter or ""),
            str(hard.region or ""),
            " ".join(rp.hints.genres or []),
            str(rp.soft_intent.vibe or ""),
            str(rp.soft_intent.goal or ""),
        ]
    ).casefold()
    region = str(hard.region or rp.graph_region_filter or "").strip().casefold()
    has_hk_region = region in HK_REGION_ALIASES or any(alias in text for alias in HK_REGION_ALIASES)
    wants_cantopop = any(cue.casefold() in text for cue in CANTONESE_CONTEXT_CUES)
    wants_mandarin = any(cue.casefold() in text for cue in MANDARIN_CONTEXT_CUES)

    generic_chinese = str(hard.language or "").strip().casefold() in {"", "chinese", "中文", "华语"}
    if has_hk_region and wants_cantopop and not wants_mandarin and generic_chinese:
        hard.language = "Cantonese"
        rp.graph_language_filter = "Cantonese"
        if not hard.region:
            hard.region = "Hong Kong"
        if not rp.graph_region_filter:
            rp.graph_region_filter = "Hong Kong"
        logger.info("[IntentPlanner] normalized HK/Cantopop grounding to Cantonese")

    # Re-validate so layered and legacy fields stay synchronized.
    plan.retrieval_plan = type(rp).model_validate(rp.model_dump())
    return plan


class IntentPlanner:
    """Select a provider adapter and return one validated query plan."""

    def __init__(self, llm_factory: Callable[[], Any]):
        self._llm_factory = llm_factory
        self._cache = PlannerResultCache(
            ttl_seconds=settings.planner_cache_ttl_seconds,
            max_entries=settings.planner_cache_max_entries,
        )

    async def plan(
        self,
        *,
        user_input: str,
        user_preferences: str,
        chat_history: str,
        previous_plan: str,
        graphzep_facts: str = "",
        user_id: str = "local_admin",
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
                    "vector_acoustic_queries": [user_input],
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
        current_date = str(date.today())
        cache_key = self._cache.make_key(
            user_input=user_input,
            user_preferences=user_preferences,
            chat_history=chat_history,
            previous_plan=previous_plan,
            graphzep_facts=graphzep_facts,
            provider=provider,
            model_name=model_name,
            current_date=current_date,
            user_id=user_id,
        )
        cached = self._cache.get(cache_key)
        if cached is not None:
            logger.info("[IntentPlanner] cache hit provider=%s model=%s", provider, model_name)
            return cached

        context = await build_context(
            graphzep_facts=graphzep_facts,
            chat_history=chat_history,
            total_budget=0,
            user_id=user_id,
        )
        payload = PlannerPayload(
            user_input=user_input,
            user_preferences=user_preferences,
            chat_history=context["chat_history"],
            previous_plan=previous_plan,
            current_date=current_date,
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
                temperature=settings.intent_temperature,
            )
        else:
            plan = await plan_with_generic_structured_output(
                llm,
                UNIFIED_PLANNER_SYSTEM,
                UNIFIED_PLANNER_HUMAN,
                payload,
            )
        plan = apply_routing_guardrails(plan, user_input)
        plan = normalize_planner_grounding(plan, user_input)
        try:
            from services.teacher_log import log_teacher_example

            log_teacher_example(
                "planner",
                inputs=payload.as_dict(),
                output=plan,
                metadata={
                    "provider": provider,
                    "model": model_name,
                    "temperature": settings.intent_temperature,
                    "prompt_version": UNIFIED_PLANNER_PROMPT_VERSION,
                    "planner_quality_mode": settings.planner_quality_mode,
                },
            )
        except Exception:
            pass
        self._cache.put(cache_key, plan)
        return plan
