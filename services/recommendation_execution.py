"""Request-local execution choices; never mutate process defaults per request."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import os
from typing import Any, Iterator, Literal


ModelRole = Literal["main", "intent", "conversation", "explain"]


@dataclass(frozen=True)
class RecommendationModels:
    main: Any
    intent: Any
    conversation: Any
    explain: Any


@dataclass(frozen=True)
class RecommendationExecution:
    web_search_enabled: bool
    models: RecommendationModels


_CURRENT: ContextVar[RecommendationExecution | None] = ContextVar(
    "soultuner_recommendation_execution", default=None
)


def current_execution() -> RecommendationExecution | None:
    return _CURRENT.get()


@contextmanager
def execution_scope(execution: RecommendationExecution) -> Iterator[RecommendationExecution]:
    token = _CURRENT.set(execution)
    try:
        yield execution
    finally:
        _CURRENT.reset(token)


def request_model(role: ModelRole) -> Any:
    execution = current_execution()
    return getattr(execution.models, role) if execution is not None else None


def web_search_allowed() -> bool:
    execution = current_execution()
    if execution is not None:
        return execution.web_search_enabled
    # CLI and offline tools retain the startup default, not another HTTP turn's flag.
    return os.getenv("MUSIC_WEB_SEARCH_ENABLED", "1").strip().lower() not in {
        "0", "false", "no", "off"
    }
