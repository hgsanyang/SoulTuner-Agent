"""Native LiteLLM wrapper used by scripts and ingestion tools."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .base import BaseLLM
from .registry import (
    OPENAI_COMPAT_PROVIDERS,
    get_provider_config,
    inject_provider_env,
    provider_api_key,
    with_provider_prefix,
)


class MultiLLM(BaseLLM):
    """
    Lightweight native LLM caller.

    This is for utility scripts that need a direct string-in/string-out call
    instead of a LangChain ChatModel.
    """

    def __init__(
        self,
        provider: str = "siliconflow",
        model_name: Optional[str] = None,
        temperature: float = 0.7,
    ):
        self.provider = provider.lower()
        config = get_provider_config(self.provider)

        api_key = provider_api_key(self.provider)
        if not api_key:
            print(f"⚠️ 警告: 尚未在环境变量中找到 {self.provider} 的 API_KEY")
        inject_provider_env(self.provider, api_key)

        self.model_name = model_name or config["default_model"]
        if self.provider in OPENAI_COMPAT_PROVIDERS:
            self.litellm_model = f"openai/{self.model_name}"
        elif self.provider == "openai":
            self.litellm_model = self.model_name
        else:
            self.litellm_model = with_provider_prefix(self.provider, self.model_name)

        self.temperature = 1.0 if self.provider == "google" and temperature < 1.0 else temperature
        super().__init__(api_key=api_key, model_name=self.model_name)

    def get_default_model(self) -> str:
        return self.model_name

    def invoke(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        """Call the configured provider through LiteLLM."""
        try:
            from litellm import completion
        except ImportError as exc:
            raise ImportError("环境缺少 litellm 库。请执行: pip install litellm") from exc

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            response = completion(
                model=self.litellm_model,
                messages=messages,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", 4000),
            )
            content = response.choices[0].message.content
            return self.validate_response(content)
        except Exception as exc:
            print(f"[{self.provider}] LiteLLM API调用错误: {exc}")
            return ""

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "model_name": self.model_name,
            "routing_target": self.litellm_model,
        }
