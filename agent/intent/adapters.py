"""Provider-specific transports for intent planning."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import httpx
from langchain_core.prompts import ChatPromptTemplate

from config.logging_config import get_logger
from schemas.query_plan import MusicQueryPlan

from .parsing import parse_music_query_plan

logger = get_logger(__name__)


@dataclass(frozen=True)
class PlannerPayload:
    user_input: str
    user_preferences: str
    chat_history: str
    previous_plan: str
    current_date: str

    def as_dict(self) -> dict[str, str]:
        return {
            "user_input": self.user_input,
            "user_preferences": self.user_preferences,
            "chat_history": self.chat_history,
            "previous_plan": self.previous_plan,
            "current_date": self.current_date,
        }


def _model_name(llm: Any, fallback: str) -> str:
    return str(getattr(llm, "model_name", "") or fallback)


def _temperature(llm: Any, fallback: float = 0.3) -> float:
    value = getattr(llm, "temperature", fallback)
    return fallback if value is None else float(value)


def _openai_base_url(llm: Any) -> str:
    for attribute in ("openai_api_base", "base_url"):
        value = getattr(llm, attribute, None)
        if value:
            return str(value).rstrip("/")
    raise RuntimeError("OpenAI-compatible planner is missing a base URL")


async def plan_with_local_structured_output(
    llm: Any,
    prompt_template: str,
    payload: PlannerPayload,
) -> MusicQueryPlan:
    structured_llm = llm.with_structured_output(MusicQueryPlan, method="json_mode")
    chain = ChatPromptTemplate.from_template(prompt_template) | structured_llm
    return await chain.ainvoke(payload.as_dict())


async def plan_with_generic_structured_output(
    llm: Any,
    system_prompt: str,
    human_prompt: str,
    payload: PlannerPayload,
) -> MusicQueryPlan:
    structured_llm = llm.with_structured_output(MusicQueryPlan, include_raw=True)
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt),
    ])
    result = await (prompt | structured_llm).ainvoke(payload.as_dict())
    plan = result.get("parsed")
    if plan is None:
        raise RuntimeError("Planner returned no parsed MusicQueryPlan")

    raw_message = result.get("raw")
    usage = getattr(raw_message, "usage_metadata", None) or {}
    if usage:
        logger.info(
            "[IntentPlanner] tokens prompt=%s completion=%s cache_hit=%s",
            usage.get("input_tokens", 0),
            usage.get("output_tokens", 0),
            usage.get("prompt_cache_hit_tokens", 0)
            or usage.get("cache_read_input_tokens", 0)
            or (usage.get("input_token_details") or {}).get("cache_read", 0),
        )
    return plan


async def plan_with_sglang(
    llm: Any,
    prompt_template: str,
    payload: PlannerPayload,
    max_tokens: int,
    timeout: float,
) -> MusicQueryPlan:
    messages = ChatPromptTemplate.from_template(prompt_template).format_messages(**payload.as_dict())
    role_map = {"human": "user", "ai": "assistant"}
    api_messages = [
        {
            "role": role_map.get(getattr(message, "type", "human"), getattr(message, "type", "user")),
            "content": message.content,
        }
        for message in messages
    ]
    request_body = {
        "model": _model_name(llm, "local-planner-qwen3-4b-fp8"),
        "messages": api_messages,
        "max_tokens": max_tokens,
        "temperature": _temperature(llm),
        "response_format": {"type": "json_object"},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    response = await _post_json(
        f"{_openai_base_url(llm)}/chat/completions",
        request_body,
        timeout=timeout,
    )
    return parse_music_query_plan(response["choices"][0]["message"]["content"])


async def plan_with_dashscope(
    api_key: str,
    model_name: str,
    system_prompt: str,
    human_prompt: str,
    payload: PlannerPayload,
    max_tokens: int,
    timeout: float,
    base_url: str,
    temperature: float = 0.0,
) -> MusicQueryPlan:
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY is not configured")

    schema = json.dumps(MusicQueryPlan.model_json_schema(), ensure_ascii=False)
    request_body = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": f"{system_prompt}\n\n请严格按以下 JSON Schema 输出，只输出 JSON：\n{schema}",
            },
            {"role": "user", "content": human_prompt.format(**payload.as_dict())},
        ],
        "temperature": float(temperature),
        "max_tokens": max_tokens,
        "enable_thinking": False,
        "response_format": {"type": "json_object"},
    }
    response = await _post_json(
        f"{base_url.rstrip('/')}/chat/completions",
        request_body,
        timeout=timeout,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    usage = response.get("usage", {})
    logger.info(
        "[IntentPlanner] DashScope tokens prompt=%s completion=%s cache=%s",
        usage.get("prompt_tokens", 0),
        usage.get("completion_tokens", 0),
        (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
    )
    return parse_music_query_plan(response["choices"][0]["message"]["content"])


async def _post_json(
    url: str,
    body: dict[str, Any],
    timeout: float,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json=body, headers=headers)
        response.raise_for_status()
        return response.json()
    except httpx.TimeoutException as exc:
        raise RuntimeError(f"Planner request timed out after {timeout:.0f}s") from exc
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        detail = exc.response.text[:300]
        raise RuntimeError(f"Planner API returned HTTP {status}: {detail}") from exc
