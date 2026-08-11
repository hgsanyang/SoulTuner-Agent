"""Model choices shared by the self-hosted UI and deployment docs."""

from __future__ import annotations

import os
from typing import Any


PROFILE_QWEN = "qwen3.7-plus"
PROFILE_SOULTUNER = "soultuner-v4.2-35b"
PROFILE_SAFE = "safe"

PROFILE_LABELS = {
    PROFILE_QWEN: "Qwen3.7 Plus（云端，4070 可用）",
    PROFILE_SOULTUNER: "SoulTuner V4.2 35B（自托管）",
    PROFILE_SAFE: "安全演示（无需模型）",
}


def profile_choices() -> list[tuple[str, str]]:
    """Stable dropdown choices; the value is also the deployment switch."""

    return [(label, profile) for profile, label in PROFILE_LABELS.items()]


def default_profile() -> str:
    configured = os.getenv("SOULTUNER_MODEL_PROFILE", "").strip()
    if configured in PROFILE_LABELS:
        return configured
    if os.getenv("DASHSCOPE_API_KEY", "").strip():
        return PROFILE_QWEN
    if os.getenv("SOULTUNER_PLANNER_ENDPOINT", "").strip():
        return PROFILE_SOULTUNER
    return PROFILE_SAFE


def resolve_profile(profile: str) -> tuple[dict[str, Any] | None, str]:
    """Resolve a UI choice without ever returning or displaying credentials."""

    selected = profile if profile in PROFILE_LABELS else default_profile()
    if selected == PROFILE_SAFE:
        return None, "安全演示不调用外部模型"
    if selected == PROFILE_QWEN:
        token = os.getenv("DASHSCOPE_API_KEY", "").strip()
        if not token:
            return None, "未配置 DASHSCOPE_API_KEY，已使用安全演示"
        return {
            "profile": selected,
            "endpoint": os.getenv(
                "DASHSCOPE_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            ).rstrip("/")
            + "/chat/completions",
            "model": os.getenv("SOULTUNER_QWEN_MODEL", PROFILE_QWEN),
            "token": token,
            "protocol": "openai",
        }, "Qwen3.7 Plus 云端端点"
    endpoint = os.getenv("SOULTUNER_PLANNER_ENDPOINT", "").strip()
    if not endpoint:
        return None, "未配置 35B 端点，已使用安全演示"
    return {
        "profile": selected,
        "endpoint": endpoint,
        "model": os.getenv("SOULTUNER_PLANNER_MODEL", PROFILE_SOULTUNER),
        "token": os.getenv("SOULTUNER_PLANNER_TOKEN", "").strip(),
        "protocol": os.getenv("SOULTUNER_PLANNER_PROTOCOL", "openai").strip().casefold(),
    }, "SoulTuner V4.2 35B 端点"
