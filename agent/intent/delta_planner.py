"""A7 follow-up planner that emits state mutations instead of full plans."""

from __future__ import annotations

import json
import os
from typing import Any, Callable

from langchain_core.prompts import ChatPromptTemplate

from config.logging_config import get_logger
from config.settings import settings
from llms.prompts import INTENT_DELTA_HUMAN_PROMPT, INTENT_DELTA_SYSTEM_PROMPT
from schemas.dialog_state import (
    DialogMusicState,
    PlanDelta,
    build_deterministic_plan_delta,
    load_dialog_state,
)

from .adapters import _post_json

logger = get_logger(__name__)
LOCAL_PROVIDERS = {"sglang", "vllm", "ollama"}


def _compact_state(state: DialogMusicState) -> dict[str, Any]:
    return {
        "hard_constraints": state.hard_constraints.model_dump(),
        "soft_intent": state.soft_intent.model_dump(),
        "hints": state.hints.model_dump(),
        "last_query": state.last_query,
        "last_result_titles": state.last_result_titles[:8],
        "last_result_artists": state.last_result_artists[:8],
        "turn_count": state.turn_count,
    }


def _parse_delta_json(content: Any) -> PlanDelta:
    if isinstance(content, dict):
        return PlanDelta.model_validate(content)
    text = str(content or "").strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```")
        text = text.removesuffix("```").strip()
    return PlanDelta.model_validate(json.loads(text))


class IntentDeltaPlanner:
    """Resolve established follow-ups with a small, auditable PlanDelta."""

    def __init__(self, llm_factory: Callable[[], Any]):
        self._llm_factory = llm_factory

    async def plan(
        self,
        *,
        user_input: str,
        dialog_state: DialogMusicState | dict[str, Any],
    ) -> PlanDelta:
        state = load_dialog_state(dialog_state)
        deterministic = build_deterministic_plan_delta(user_input, state)
        if deterministic is not None:
            return deterministic

        if os.getenv("MUSIC_MOCK_MODE", "0").lower() in {"1", "true", "yes"}:
            return PlanDelta(confidence=0.8, planner_mode="delta_llm")

        llm = self._llm_factory()
        provider = (settings.intent_llm_provider or settings.llm_default_provider).lower()
        model_name = (
            getattr(llm, "model_name", "")
            or settings.intent_llm_model
            or settings.llm_default_model
            or "qwen3.7-plus"
        )
        state_json = json.dumps(_compact_state(state), ensure_ascii=False, separators=(",", ":"))
        pending = (
            json.dumps(state.pending_clarification.model_dump(), ensure_ascii=False)
            if state.pending_clarification
            else "无"
        )

        if provider == "dashscope":
            api_key = os.getenv("DASHSCOPE_API_KEY", "")
            if not api_key:
                raise RuntimeError("DASHSCOPE_API_KEY is not configured")
            schema = json.dumps(PlanDelta.model_json_schema(), ensure_ascii=False)
            body = {
                "model": model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            f"{INTENT_DELTA_SYSTEM_PROMPT}\n\n"
                            f"请严格按以下 JSON Schema 输出，只输出 JSON：\n{schema}"
                        ),
                    },
                    {
                        "role": "user",
                        "content": INTENT_DELTA_HUMAN_PROMPT.format(
                            dialog_state=state_json,
                            user_input=user_input,
                            pending_clarification=pending,
                        ),
                    },
                ],
                "temperature": float(settings.intent_temperature),
                "max_tokens": min(int(settings.intent_max_tokens), 1200),
                "enable_thinking": False,
                "response_format": {"type": "json_object"},
            }
            response = await _post_json(
                (
                    os.getenv(
                        "DASHSCOPE_BASE_URL",
                        "https://dashscope.aliyuncs.com/compatible-mode/v1",
                    ).rstrip("/")
                    + "/chat/completions"
                ),
                body,
                timeout=settings.llm_timeout,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
            )
            delta = _parse_delta_json(response["choices"][0]["message"]["content"])
        else:
            structured = llm.with_structured_output(PlanDelta, method="json_mode")
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", INTENT_DELTA_SYSTEM_PROMPT),
                    ("human", INTENT_DELTA_HUMAN_PROMPT),
                ]
            )
            delta = await (prompt | structured).ainvoke(
                {
                    "dialog_state": state_json,
                    "user_input": user_input,
                    "pending_clarification": pending,
                }
            )
            delta = PlanDelta.model_validate(delta)

        delta.planner_mode = "delta_llm"
        logger.info(
            "[IntentDeltaPlanner] provider=%s model=%s operations=%d confidence=%.2f",
            provider,
            model_name,
            len(delta.operations),
            delta.confidence,
        )
        return delta
