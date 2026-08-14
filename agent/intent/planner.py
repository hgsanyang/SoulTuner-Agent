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
from schemas.tool_plan import tool_plan_alignment_issues

from .adapters import (
    PlannerPayload,
    plan_with_dashscope,
    plan_with_generic_structured_output,
    plan_with_local_structured_output,
    plan_with_sglang,
)
from .v42_adapter import V42_PROMPT_VERSION, plan_with_v42_contract, uses_v42_contract

logger = get_logger(__name__)
LOCAL_PROVIDERS = {"sglang", "vllm", "ollama"}
UNIFIED_PLANNER_PROMPT_VERSION = "unified_planner_toolplan_v1_2026_07_13"


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
        planner_contract: str = "",
        reference_context: dict[str, str] | None = None,
        user_id: str = "",
        conversation_id: str = "",
    ) -> str:
        profile_context = json.dumps(
            [
                user_preferences,
                chat_history,
                previous_plan,
                graphzep_facts,
                reference_context or {},
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        profile_hash = hashlib.sha256(profile_context.encode("utf-8")).hexdigest()
        material = "\0".join(
            [
                user_input.strip(),
                profile_hash,
                provider,
                model_name,
                current_date,
                planner_contract,
                user_id,
                conversation_id,
            ]
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
        reference_context: dict[str, str] | None = None,
        conversation_id: str = "",
        context_preassembled: bool = False,
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
        planner_contract = str(settings.intent_planner_contract or "auto").strip().lower()
        v42_enabled = uses_v42_contract(provider, model_name, planner_contract)
        current_date = str(date.today())
        if context_preassembled:
            bounded_preferences = user_preferences
            bounded_history = chat_history
            # The canonical assembler has already merged and budgeted long-term
            # memory into ``user_preferences``.  Do not let the raw memory text
            # influence either the prompt cache key or a later provider path.
            bounded_graphzep_facts = ""
        else:
            context = await build_context(
                explicit_profile=user_preferences,
                graphzep_facts=graphzep_facts,
                chat_history=chat_history,
                total_budget=0,
                user_id=user_id,
                conversation_id=conversation_id,
                user_input=user_input,
            )
            preference_parts = []
            if context.get("explicit_profile"):
                preference_parts.append(context["explicit_profile"])
            if context.get("graphzep_facts") and context["graphzep_facts"] != "暂无用户长期记忆":
                preference_parts.append(context["graphzep_facts"])
            bounded_preferences = "\n".join(preference_parts) or "无"
            bounded_history = context["chat_history"]
            bounded_graphzep_facts = context.get("graphzep_facts", "")

        cache_key = self._cache.make_key(
            user_input=user_input,
            user_preferences=bounded_preferences,
            chat_history=bounded_history,
            previous_plan=previous_plan,
            graphzep_facts=bounded_graphzep_facts,
            provider=provider,
            model_name=model_name,
            current_date=current_date,
            planner_contract=planner_contract,
            reference_context=reference_context,
            user_id=user_id,
            conversation_id=conversation_id,
        )
        cached = self._cache.get(cache_key)
        if cached is not None:
            logger.info("[IntentPlanner] cache hit provider=%s model=%s", provider, model_name)
            return cached

        payload = PlannerPayload(
            user_input=user_input,
            user_preferences=bounded_preferences,
            chat_history=bounded_history,
            previous_plan=previous_plan,
            current_date=current_date,
            retrieved_memories=bounded_graphzep_facts,
        )
        logger.info("[IntentPlanner] provider=%s model=%s", provider, model_name)

        guard_findings: list[str] = []
        if v42_enabled:
            plan, guard_findings = await plan_with_v42_contract(
                llm,
                payload,
                max_tokens=settings.intent_max_tokens,
                timeout=settings.llm_timeout,
                api_key=settings.vllm_api_key if provider == "vllm" else settings.sglang_api_key,
                reference_context=reference_context,
            )
            logger.info(
                "[IntentPlanner] V4.2 guard: %s",
                "; ".join(guard_findings),
            )
        elif provider == "sglang":
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
        # LLM-first: semantic intent and entity grounding stay with the planner.
        alignment_issues = tool_plan_alignment_issues(plan)
        if alignment_issues:
            logger.warning(
                "[IntentPlanner] ToolPlan/RetrievalPlan alignment issues: %s",
                ", ".join(alignment_issues),
            )
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
                    "prompt_version": (
                        V42_PROMPT_VERSION if v42_enabled else UNIFIED_PLANNER_PROMPT_VERSION
                    ),
                    "planner_contract": "v42" if v42_enabled else "legacy",
                    "guard_findings": guard_findings,
                    "planner_quality_mode": settings.planner_quality_mode,
                },
            )
        except Exception:
            pass
        self._cache.put(cache_key, plan)
        return plan
