"""Contracts for local user profiles.

Profiles are local application identities, not authenticated accounts. New
profiles use UUIDs; ``local_admin`` is the sole legacy identifier accepted for
backward compatibility.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

DEFAULT_PROFILE_ID = "local_admin"
ProfileType = Literal["personal", "test"]
ProfileStatus = Literal["active", "deleted"]


def validate_profile_id(value: str | UUID) -> str:
    """Return a canonical profile id or reject unsafe legacy-like values."""

    normalized = str(value).strip()
    if normalized == DEFAULT_PROFILE_ID:
        return normalized
    try:
        return str(UUID(normalized))
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError("profile_id must be a UUID or 'local_admin'") from exc


def developer_sandbox_id(profile_id: str | UUID) -> str:
    """Derive the isolated developer-mode User id for a registered profile."""

    return f"__dev__:{validate_profile_id(profile_id)}"


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class UserProfile(_StrictModel):
    profile_id: str
    display_name: str = Field(min_length=1, max_length=64)
    profile_type: ProfileType = "personal"
    status: ProfileStatus = "active"
    created_at: datetime | None = None
    updated_at: datetime | None = None
    deleted_at: datetime | None = None

    @field_validator("profile_id", mode="before")
    @classmethod
    def _validate_profile_id(cls, value: str | UUID) -> str:
        return validate_profile_id(value)

    @field_validator("display_name")
    @classmethod
    def _clean_display_name(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("display_name must not be blank")
        return cleaned


class CreateUserProfile(_StrictModel):
    display_name: str = Field(min_length=1, max_length=64)
    profile_type: ProfileType = "personal"

    @field_validator("display_name")
    @classmethod
    def _clean_display_name(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("display_name must not be blank")
        return cleaned


class UpdateUserProfile(_StrictModel):
    display_name: str | None = Field(default=None, min_length=1, max_length=64)
    profile_type: ProfileType | None = None

    @field_validator("display_name")
    @classmethod
    def _clean_display_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("display_name must not be blank")
        return cleaned

    @model_validator(mode="after")
    def _require_change(self) -> "UpdateUserProfile":
        if self.display_name is None and self.profile_type is None:
            raise ValueError("at least one profile field must be supplied")
        return self


class DeveloperSandboxReset(_StrictModel):
    profile_id: str
    sandbox_user_id: str
    reset: bool

    @field_validator("profile_id", mode="before")
    @classmethod
    def _validate_profile_id(cls, value: str | UUID) -> str:
        return validate_profile_id(value)

