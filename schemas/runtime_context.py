"""Per-request identity and data-governance contract.

Profile identity and interaction mode are deliberately separate:

* ``profile_id`` identifies the local person/profile.
* ``interaction_mode`` says whether this request is normal use or a test.

Developer requests use a derived sandbox user id. They may exercise memory and
feedback paths, but their records are never eligible for training.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

RUNTIME_CONTEXT_VERSION = "runtime_context_v1"

InteractionMode = Literal["personal", "developer", "legacy"]
RuntimeProfileType = Literal["personal", "test"]
DataPurpose = Literal[
    "ranking",
    "preference_and_ranking",
    "planner_teacher",
    "diagnostics",
    "legacy_unclassified",
]


class RuntimeContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = RUNTIME_CONTEXT_VERSION
    profile_id: str = Field(default="local_admin", min_length=1, max_length=128)
    profile_type: RuntimeProfileType = "personal"
    interaction_mode: InteractionMode = "personal"
    effective_user_id: str = Field(default="local_admin", min_length=1, max_length=160)
    session_id: str = Field(default="", max_length=160)
    training_eligible: bool = True
    teacher_log_eligible: bool = True


def build_runtime_context(
    *,
    profile_id: str = "",
    user_id: str = "",
    profile_type: str = "personal",
    interaction_mode: str = "personal",
    session_id: str = "",
) -> RuntimeContext:
    """Validate client input and derive the server-authoritative user id."""

    profile = str(profile_id or user_id or "local_admin").strip() or "local_admin"
    raw_mode = str(interaction_mode or "personal").strip().lower()
    # Unknown or misspelled modes must never silently become training data.
    mode: InteractionMode = raw_mode if raw_mode in {"personal", "developer", "legacy"} else "legacy"  # type: ignore[assignment]
    canonical_profile_type: RuntimeProfileType = (
        "test" if str(profile_type or "").strip().lower() == "test" else "personal"
    )
    if mode == "developer":
        effective_user_id = f"__dev__:{profile}"
        eligible = False
    elif mode == "legacy":
        effective_user_id = profile
        eligible = False
    else:
        effective_user_id = profile
        eligible = canonical_profile_type == "personal"
    return RuntimeContext(
        profile_id=profile,
        profile_type=canonical_profile_type,
        interaction_mode=mode,
        effective_user_id=effective_user_id,
        session_id=str(session_id or "").strip(),
        training_eligible=eligible,
        teacher_log_eligible=eligible,
    )
