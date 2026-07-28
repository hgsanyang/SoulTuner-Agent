"""Local profile registry API.

These are local application profiles, not authenticated internet accounts.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from api.security import require_admin_api_key
from schemas.user_profile import CreateUserProfile, UpdateUserProfile
from services.user_profiles import (
    UserProfileConflict,
    UserProfileNotFound,
    UserProfileService,
    UserProfileStoreUnavailable,
)

router = APIRouter(prefix="/api/profiles", tags=["profiles"])


def _service() -> UserProfileService:
    return UserProfileService()


def _raise_profile_error(exc: Exception) -> None:
    if isinstance(exc, UserProfileNotFound):
        raise HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, UserProfileConflict):
        raise HTTPException(status_code=409, detail=str(exc))
    if isinstance(exc, (UserProfileStoreUnavailable, ValueError)):
        raise HTTPException(
            status_code=503 if isinstance(exc, UserProfileStoreUnavailable) else 422,
            detail=str(exc),
        )
    raise exc


def _clear_runtime_profile_cache() -> None:
    from api.runtime_context import clear_profile_type_cache

    clear_profile_type_cache()


@router.get("")
async def list_profiles():
    try:
        profiles = _service().list_profiles()
        return {"success": True, "profiles": [item.model_dump(mode="json") for item in profiles]}
    except Exception as exc:
        _raise_profile_error(exc)


@router.post("")
async def create_profile(request: CreateUserProfile):
    try:
        profile = _service().create_profile(request)
        _clear_runtime_profile_cache()
        return {"success": True, "profile": profile.model_dump(mode="json")}
    except Exception as exc:
        _raise_profile_error(exc)


@router.patch("/{profile_id}")
async def update_profile(profile_id: str, request: UpdateUserProfile):
    try:
        profile = _service().update_profile(profile_id, request)
        _clear_runtime_profile_cache()
        return {"success": True, "profile": profile.model_dump(mode="json")}
    except Exception as exc:
        _raise_profile_error(exc)


@router.delete("/{profile_id}")
async def delete_profile(
    profile_id: str,
    _: None = Depends(require_admin_api_key),
):
    try:
        profile = _service().soft_delete_profile(profile_id)
        _clear_runtime_profile_cache()
        return {"success": True, "profile": profile.model_dump(mode="json")}
    except Exception as exc:
        _raise_profile_error(exc)


@router.post("/{profile_id}/developer-sandbox/reset")
async def reset_developer_sandbox(
    profile_id: str,
    _: None = Depends(require_admin_api_key),
):
    try:
        result = _service().reset_developer_sandbox(profile_id)
        return {"success": True, **result.model_dump(mode="json")}
    except Exception as exc:
        _raise_profile_error(exc)
