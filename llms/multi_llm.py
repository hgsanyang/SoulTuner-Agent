"""Backward-compatible LLM facade.

The implementation now lives in smaller modules:
- llms.registry: provider defaults and environment helpers
- llms.chat_models: LangChain ChatModel factories
- llms.native: direct LiteLLM wrapper
"""

from .chat_models import (
    deepseek_llm,
    gemini_llm,
    get_chat_model,
    get_compress_chat_model,
    get_explain_chat_model,
    get_intent_chat_model,
    ollama_llm,
    qwen_llm,
    sglang_llm,
    siliconflow_llm,
    vllm_llm,
)
from .native import MultiLLM
from .registry import MODEL_REGISTRY, get_env_value

_get_env_val = get_env_value

__all__ = [
    "MODEL_REGISTRY",
    "MultiLLM",
    "_get_env_val",
    "deepseek_llm",
    "gemini_llm",
    "get_chat_model",
    "get_compress_chat_model",
    "get_explain_chat_model",
    "get_intent_chat_model",
    "ollama_llm",
    "qwen_llm",
    "sglang_llm",
    "siliconflow_llm",
    "vllm_llm",
]
