"""Provider registry and environment helpers for LLM routing."""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

load_dotenv(override=True)


MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "openai": {
        "prefix": "openai/",
        "default_model": "gpt-4o",
        "api_key_env": "OPENAI_API_KEY",
        "base_url_env": "OPENAI_BASE_URL",
    },
    "deepseek": {
        "prefix": "deepseek/",
        "default_model": "deepseek-chat",
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url_env": "DEEPSEEK_BASE_URL",
    },
    "zhipu": {
        "prefix": "zhipu/",
        "default_model": "glm-4",
        "api_key_env": "ZHIPU_API_KEY",
        "base_url_env": "ZHIPU_BASE_URL",
    },
    "minimax": {
        "prefix": "minimax/",
        "default_model": "abab6.5s-chat",
        "api_key_env": "MINIMAX_API_KEY",
        "base_url_env": "MINIMAX_BASE_URL",
    },
    "dashscope": {
        "prefix": "dashscope/",
        # qwen3.7-flash is not available on the current DashScope account; qwen3.7-plus is.
        "default_model": "qwen3.7-plus",
        "api_key_env": "DASHSCOPE_API_KEY",
        "base_url_env": "DASHSCOPE_BASE_URL",
        "default_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    },
    "siliconflow": {
        "prefix": "openai/",
        "default_model": "deepseek-ai/DeepSeek-V3.2",
        "api_key_env": ["SiliconFlow_API_KEY", "SILICONFLOW_API_KEY"],
        "base_url_env": "SILICONFLOW_BASE_URL",
        "default_base_url": "https://api.siliconflow.cn/v1",
    },
    "ollama": {
        "prefix": "openai/",
        "default_model": "qwen2.5:7b",
        "api_key_env": ["OLLAMA_API_KEY", "LLM_API_KEY"],
        "base_url_env": "OLLAMA_BASE_URL",
        "default_base_url": "http://localhost:11434/v1",
    },
    "vllm": {
        "prefix": "openai/",
        "default_model": "Qwen/Qwen2.5-7B-Instruct",
        "api_key_env": ["VLLM_API_KEY", "LLM_API_KEY"],
        "base_url_env": "VLLM_BASE_URL",
        "default_base_url": "http://localhost:8000/v1",
    },
    "sglang": {
        "prefix": "openai/",
        "default_model": "local-planner-qwen3-4b-fp8",
        "api_key_env": ["SGLANG_API_KEY", "LLM_API_KEY"],
        "base_url_env": "SGLANG_BASE_URL",
        "default_base_url": "http://localhost:8000/v1",
    },
    "google": {
        "prefix": "gemini/",
        "default_model": "gemini-3-flash-preview",
        "api_key_env": "GOOGLE_API_KEY",
        "base_url_env": "",
    },
    "volcengine": {
        "prefix": "openai/",
        "default_model": "ep-20260405142751-x4jm6",
        "api_key_env": "VOLCENGINE_API_KEY",
        "base_url_env": "VOLCENGINE_BASE_URL",
        "default_base_url": "https://ark.cn-beijing.volces.com/api/v3",
    },
}


OPENAI_COMPAT_PROVIDERS = {"siliconflow", "volcengine", "dashscope", "ollama", "vllm", "sglang"}
THINKING_DISABLED_PROVIDERS = OPENAI_COMPAT_PROVIDERS | {"openai"}


def get_env_value(env_keys: str | list[str] | tuple[str, ...] | None, default: str | None = None) -> str | None:
    """Return the first non-empty value from one or more environment variable names."""
    if not env_keys:
        return default
    if isinstance(env_keys, str):
        env_keys = [env_keys]
    for key in env_keys:
        val = os.getenv(key)
        if val:
            return val
    return default


def get_provider_config(provider: str) -> dict[str, Any]:
    provider_key = provider.lower()
    if provider_key not in MODEL_REGISTRY:
        raise ValueError(f"不支持的厂商: {provider_key}。支持列表: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[provider_key]


def provider_api_key(provider: str) -> str | None:
    config = get_provider_config(provider)
    return get_env_value(config["api_key_env"])


def provider_base_url(provider: str) -> str | None:
    config = get_provider_config(provider)
    return get_env_value(config.get("base_url_env", ""), config.get("default_base_url"))


def inject_provider_env(provider: str, api_key: str | None = None, base_url: str | None = None) -> None:
    """Set environment variables expected by LiteLLM/LangChain provider adapters."""
    provider_key = provider.lower()
    api_key = api_key if api_key is not None else provider_api_key(provider_key)
    base_url = base_url if base_url is not None else provider_base_url(provider_key)

    if provider_key == "dashscope":
        os.environ["DASHSCOPE_API_KEY"] = api_key or ""
    elif provider_key == "zhipu":
        os.environ["ZHIPUAI_API_KEY"] = api_key or ""
    elif provider_key == "deepseek":
        os.environ["DEEPSEEK_API_KEY"] = api_key or ""
    elif provider_key == "minimax":
        os.environ["MINIMAX_API_KEY"] = api_key or ""
    elif provider_key == "google":
        os.environ["GOOGLE_API_KEY"] = api_key or ""

    if provider_key in OPENAI_COMPAT_PROVIDERS or provider_key == "openai":
        os.environ["OPENAI_API_KEY"] = api_key or ""
        if base_url:
            os.environ["OPENAI_API_BASE"] = base_url


def with_provider_prefix(provider: str, model_name: str) -> str:
    config = get_provider_config(provider)
    prefix = config["prefix"]
    if model_name.startswith(prefix):
        return model_name
    return f"{prefix}{model_name}"
