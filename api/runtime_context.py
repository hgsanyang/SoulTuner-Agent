"""Resolve the server-authoritative runtime context from HTTP + request data."""

from __future__ import annotations

from functools import lru_cache

from fastapi import Request

from schemas.runtime_context import RuntimeContext, build_runtime_context
from schemas.user_profile import DEFAULT_PROFILE_ID, validate_profile_id


@lru_cache(maxsize=256)
def _registered_profile_type(profile_id: str) -> str:
    """Resolve server-owned profile metadata without trusting a client header."""

    if profile_id == DEFAULT_PROFILE_ID:
        return "personal"
    try:
        validate_profile_id(profile_id)
    except ValueError:
        # Unknown legacy-like ids remain usable as isolated test identities, but
        # can never manufacture training-eligible records.
        return "test"
    try:
        from services.user_profiles import UserProfileService

        profile = UserProfileService().get_profile(profile_id)
        return profile.profile_type if profile.status == "active" else "test"
    except Exception:
        # Unverified UUID profiles must never silently become trainable.
        return "test"


def clear_profile_type_cache() -> None:
    _registered_profile_type.cache_clear()


def runtime_context_from_request(
    raw_request: Request,
    *,
    profile_id: str = "",
    user_id: str = "",
    interaction_mode: str = "",
    session_id: str = "",
) -> RuntimeContext:
    headers = raw_request.headers
    resolved_profile_id = (
        headers.get("X-SoulTuner-Profile")
        or profile_id
        or user_id
        or DEFAULT_PROFILE_ID
    )
    return build_runtime_context(
        profile_id=resolved_profile_id,
        user_id=user_id,
        profile_type=_registered_profile_type(str(resolved_profile_id).strip()),
        interaction_mode=(
            headers.get("X-SoulTuner-Mode")
            or interaction_mode
            or "personal"
        ),
        session_id=(
            headers.get("X-SoulTuner-Session")
            or session_id
        ),
    )


def assert_exposure_owner(exposure: dict, context: RuntimeContext) -> None:
    """Reject unowned or cross-profile exposure feedback."""

    from fastapi import HTTPException

    owner = str(exposure.get("user_id") or "").strip()
    if not owner or owner != context.effective_user_id:
        raise HTTPException(
            status_code=403,
            detail=(
                "exposure has no trusted owner"
                if not owner
                else "exposure belongs to another profile or interaction mode"
            ),
        )


def reject_developer_catalog_mutation(raw_request: Request) -> None:
    """Keep developer-mode UI experiments from changing the shared catalog."""

    from fastapi import HTTPException

    context = runtime_context_from_request(raw_request)
    if context.interaction_mode == "developer":
        raise HTTPException(
            status_code=409,
            detail="catalog changes are disabled in developer mode",
        )
