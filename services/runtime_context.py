"""Async-task-local runtime context.

``ContextVar`` keeps concurrent users and modes isolated without process-global
environment flags. Child asyncio tasks inherit the context at creation time.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Iterator

from schemas.runtime_context import DataPurpose, RuntimeContext, build_runtime_context

_DEFAULT = build_runtime_context(
    profile_id="__unscoped__",
    profile_type="test",
    interaction_mode="legacy",
)
_CURRENT: ContextVar[RuntimeContext] = ContextVar("soultuner_runtime_context", default=_DEFAULT)


def current_runtime_context() -> RuntimeContext:
    return _CURRENT.get()


def set_runtime_context(context: RuntimeContext) -> Token[RuntimeContext]:
    return _CURRENT.set(context)


def reset_runtime_context(token: Token[RuntimeContext]) -> None:
    _CURRENT.reset(token)


@contextmanager
def runtime_context_scope(context: RuntimeContext) -> Iterator[RuntimeContext]:
    token = set_runtime_context(context)
    try:
        yield context
    finally:
        reset_runtime_context(token)


def provenance_fields(
    purpose: DataPurpose,
    *,
    context: RuntimeContext | None = None,
) -> dict[str, object]:
    ctx = context or current_runtime_context()
    return {
        "runtime_context_version": ctx.schema_version,
        "profile_id": ctx.profile_id,
        "profile_type": ctx.profile_type,
        "interaction_mode": ctx.interaction_mode,
        "training_eligible": bool(ctx.training_eligible),
        "data_purpose": purpose,
        "session_id": ctx.session_id,
    }


def shared_catalog_side_effects_allowed(
    *,
    context: RuntimeContext | None = None,
) -> bool:
    """Only normal use by a personal profile may mutate shared catalog data."""

    ctx = context or current_runtime_context()
    return bool(
        ctx.interaction_mode == "personal"
        and ctx.profile_type == "personal"
        and ctx.training_eligible
    )


def normalize_provenance(
    payload: dict,
    *,
    legacy_default: bool = True,
) -> dict:
    """Return a copy with explicit provenance.

    Records written before this contract are fail-closed: they remain available
    for audit, but cannot silently become training samples.
    """

    row = dict(payload)
    if "interaction_mode" not in row and legacy_default:
        row.update(
            {
                "runtime_context_version": "legacy",
                "profile_id": str(row.get("user_id") or "local_admin"),
                "profile_type": "test",
                "interaction_mode": "legacy",
                "training_eligible": False,
                "data_purpose": "legacy_unclassified",
                "session_id": str(
                    row.get("session_id")
                    or (row.get("context") or {}).get("session_id")
                    or ""
                ),
            }
        )
    row["training_eligible"] = bool(
        row.get("training_eligible") is True
        and row.get("interaction_mode") == "personal"
    )
    return row


def is_training_eligible(payload: dict) -> bool:
    row = normalize_provenance(payload)
    return bool(row.get("training_eligible"))
