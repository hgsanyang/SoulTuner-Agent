"""LangChain chat model factory functions."""

from __future__ import annotations

import logging
from typing import Optional

try:
    from config.settings import settings
except Exception:  # pragma: no cover - keeps standalone imports usable
    settings = None

from .registry import (
    OPENAI_COMPAT_PROVIDERS,
    get_provider_config,
    inject_provider_env,
    provider_api_key,
    provider_base_url,
    with_provider_prefix,
)

logger = logging.getLogger(__name__)


def _settings_value(name: str, default):
    if settings is None:
        return default
    return getattr(settings, name, default)


def _chat_openai_extra_body(provider: str, target_model: str) -> dict:
    extra_body = {}
    is_qwen_family = any(kw in target_model.lower() for kw in ["qwen3", "qwen-3", "qwen2.5"])
    if is_qwen_family or provider == "sglang":
        if provider == "sglang":
            extra_body = {
                "chat_template_kwargs": {"enable_thinking": False},
            }
        else:
            extra_body = {
                "enable_thinking": False,
            }
        logger.info(
            "[LLM] 检测到 Qwen3 系模型(%s)，已关闭 Thinking Mode (provider=%s)",
            target_model,
            provider,
        )
    return extra_body


def get_chat_model(
    provider: str = "dashscope",
    model_name: Optional[str] = None,
    temperature: float = 0.7,
    timeout: Optional[int] = None,
    max_tokens: Optional[int] = None,
):
    """Return a LangChain ChatModel for graph workflow nodes."""
    provider_key = provider.lower()
    config = get_provider_config(provider_key)
    api_key = provider_api_key(provider_key)
    base_url = provider_base_url(provider_key)
    inject_provider_env(provider_key, api_key, base_url)

    if provider_key in OPENAI_COMPAT_PROVIDERS:
        from langchain_openai import ChatOpenAI

        target_model = model_name or config["default_model"]
        request_timeout = timeout if timeout is not None else _settings_value("llm_timeout", 80)
        token_budget = max_tokens if max_tokens is not None else 4000
        chat_kwargs = {}
        extra_body = _chat_openai_extra_body(provider_key, target_model)
        if extra_body:
            chat_kwargs["extra_body"] = extra_body
        return ChatOpenAI(
            api_key=api_key or "fake-key",
            base_url=base_url,
            model=target_model,
            temperature=temperature,
            max_tokens=token_budget,
            request_timeout=request_timeout,
            **chat_kwargs,
        )

    try:
        from langchain_litellm import ChatLiteLLM
    except ImportError as exc:
        raise ImportError("请确保安装了依赖: pip install litellm langchain-litellm langchain-openai") from exc

    base_model = model_name or config["default_model"]
    target_model = with_provider_prefix(provider_key, base_model)
    if provider_key == "google" and temperature < 1.0:
        temperature = 1.0

    return ChatLiteLLM(
        model=target_model,
        temperature=temperature,
        max_tokens=max_tokens if max_tokens is not None else 4000,
    )


def deepseek_llm(temperature: float = 0.7):
    return get_chat_model(provider="deepseek", temperature=temperature)


def qwen_llm(temperature: float = 0.7):
    return get_chat_model(provider="dashscope", temperature=temperature)


def siliconflow_llm(temperature: float = 0.7):
    return get_chat_model(provider="siliconflow", temperature=temperature)


def get_intent_chat_model():
    """Return the Planner LLM, falling back to the main LLM settings."""
    try:
        provider = _settings_value("intent_llm_provider", "") or _settings_value("llm_default_provider", "dashscope")
        quality_mode = str(_settings_value("planner_quality_mode", "teacher") or "teacher").strip().lower()
        if quality_mode == "fast":
            model_name = (
                _settings_value("intent_llm_fast_model", "")
                or _settings_value("intent_llm_model", "")
                or _settings_value("llm_default_model", "")
                or None
            )
        else:
            model_name = _settings_value("intent_llm_model", "") or _settings_value("llm_default_model", "") or None
        max_tokens = _settings_value("intent_max_tokens", 2048)
        temperature = _settings_value("intent_temperature", 0.3)
        return get_chat_model(provider=provider, model_name=model_name, temperature=temperature, max_tokens=max_tokens)
    except Exception:
        return get_chat_model(provider="dashscope", temperature=0.3, max_tokens=2048)


def gemini_llm(model_name: Optional[str] = None, temperature: float = 1.0):
    return get_chat_model(provider="google", model_name=model_name, temperature=temperature)


def get_compress_chat_model():
    """Return the GSSC compression LLM, defaulting to the main LLM."""
    try:
        provider = _settings_value("compress_llm_provider", "") or _settings_value("llm_default_provider", "dashscope")
        model_name = _settings_value("compress_llm_model", "") or None
        return get_chat_model(provider=provider, model_name=model_name, temperature=0.3, max_tokens=2048)
    except Exception:
        return get_chat_model(provider="dashscope", temperature=0.3, max_tokens=2048)


def get_explain_chat_model():
    """Return the explanation-generation LLM, defaulting to the main LLM."""
    try:
        provider = _settings_value("explain_llm_provider", "") or _settings_value("llm_default_provider", "dashscope")
        model_name = _settings_value("explain_llm_model", "") or _settings_value("llm_default_model", "") or None
        return get_chat_model(provider=provider, model_name=model_name, temperature=0.7)
    except Exception:
        return get_chat_model(provider="dashscope", temperature=0.7)


def ollama_llm(model_name: Optional[str] = None, temperature: float = 0.6):
    return get_chat_model(provider="ollama", model_name=model_name, temperature=temperature)


def vllm_llm(model_name: Optional[str] = None, temperature: float = 0.6):
    return get_chat_model(provider="vllm", model_name=model_name, temperature=temperature)


def sglang_llm(model_name: Optional[str] = None, temperature: float = 0.6):
    return get_chat_model(provider="sglang", model_name=model_name, temperature=temperature)
