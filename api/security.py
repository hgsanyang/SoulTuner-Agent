"""Small security helpers for local/admin API routes."""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import unquote

try:  # FastAPI is present at runtime; CI unit tests may use a minimal deps set.
    from fastapi import Header, HTTPException, status
except Exception:  # pragma: no cover - exercised only in minimal dependency CI
    class HTTPException(Exception):
        def __init__(self, status_code: int, detail: str):
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    class status:
        HTTP_401_UNAUTHORIZED = 401
        HTTP_403_FORBIDDEN = 403

    def Header(default=None, alias: str | None = None):  # type: ignore[override]
        return default


def _truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def shared_safe_mode_enabled() -> bool:
    """Return whether shared-environment protections should be enforced."""
    try:
        from config.settings import settings

        return bool(getattr(settings, "public_demo_mode", False))
    except Exception:
        return _truthy(os.getenv("SHARED_SAFE_MODE")) or _truthy(os.getenv("PUBLIC_DEMO_MODE"))


def _admin_api_key() -> str:
    try:
        from config.settings import settings

        return str(getattr(settings, "admin_api_key", "") or "").strip()
    except Exception:
        return str(os.getenv("ADMIN_API_KEY", "") or "").strip()


def admin_key_required() -> bool:
    """Whether DESTRUCTIVE ops (delete/config/rebuild) need the admin key.

    Triggered by the admin key being set, an explicit API_KEY_REQUIRED, or shared
    safe mode. This gates only the dangerous-op dependencies — NOT every /api/*.
    """
    try:
        from config.settings import settings

        configured = bool(getattr(settings, "api_key_required", False))
    except Exception:
        configured = _truthy(os.getenv("API_KEY_REQUIRED"))
    return configured or bool(_admin_api_key()) or shared_safe_mode_enabled()


def _access_api_key() -> str:
    """Key for blanket /api/* access. Prefer an explicit API_ACCESS_KEY; fall back
    to the admin key so a single-key LAN lockdown still works."""
    key = str(os.getenv("API_ACCESS_KEY", "") or "").strip()
    if not key:
        try:
            from config.settings import settings

            key = str(getattr(settings, "api_access_key", "") or "").strip()
        except Exception:
            key = ""
    return key or _admin_api_key()


def access_control_required() -> bool:
    """Whether ALL /api/* needs a key (LAN / shared mode).

    Deliberately NOT triggered by an admin key alone: protecting a delete button
    must not 401 the recommend/library/feedback pages, which the browser calls
    without any key. Turn this on with API_ACCESS_KEY or an explicit
    API_KEY_REQUIRED (or shared safe mode).
    """
    if str(os.getenv("API_ACCESS_KEY", "") or "").strip():
        return True
    try:
        from config.settings import settings

        if str(getattr(settings, "api_access_key", "") or "").strip():
            return True
        if bool(getattr(settings, "api_key_required", False)):
            return True
    except Exception:
        if _truthy(os.getenv("API_KEY_REQUIRED")):
            return True
    return shared_safe_mode_enabled()


async def require_admin_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    """FastAPI dependency for admin/destructive operations."""
    if not admin_key_required():
        return
    expected = _admin_api_key()
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin API key is required in shared safe mode.",
        )
    if x_api_key != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key.",
        )


# Served without a key even in LAN mode: liveness, API docs. Static assets and
# the SPA live outside /api/ and are exempt by not matching the /api/ prefix.
_AUTH_EXEMPT_PATHS = {"/health", "/docs", "/redoc", "/openapi.json"}


def api_request_needs_key(path: str, method: str) -> bool:
    """Whether this HTTP request must carry the key when auth is enabled.

    Blankets ALL of /api/* (reads included) rather than a hand-maintained list of
    endpoints — the gap Codex found was an unlisted mutating route (profile edit,
    memory delete) with no dependency. A prefix rule cannot be forgotten when a
    new route is added. CORS preflight and non-/api paths pass through.
    """
    if str(method or "").upper() == "OPTIONS":
        return False
    if path in _AUTH_EXEMPT_PATHS:
        return False
    return path.startswith("/api/")


def check_api_request_auth(path: str, method: str, x_api_key: str | None) -> tuple[int, str] | None:
    """Return (status, detail) if the request must be rejected, else None.

    The blanket /api/* gate uses ACCESS control, not admin: an admin key alone
    leaves the UI open (only destructive ops are gated by their own dependency).
    A purely local single-user install has no key and stays completely open.
    """
    if not access_control_required():
        return None
    if not api_request_needs_key(path, method):
        return None
    expected = _access_api_key()
    if not expected:
        return (403, "API access key required but none is configured on the server.")
    if x_api_key != expected:
        return (401, "Invalid or missing X-API-Key.")
    return None


def reject_shared_safe_action(action: str) -> None:
    """Block filesystem-changing actions in shared safe mode."""
    if shared_safe_mode_enabled():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"{action} is disabled in shared safe mode.",
        )


def safe_resolve_child(root: Path, relative_path: str) -> Path:
    """Resolve a user-controlled relative path under root.

    Raises ValueError if the path is absolute, empty, or tries to escape root.
    """
    raw = unquote(str(relative_path or "")).replace("\\", "/").strip()
    if not raw:
        raise ValueError("empty path")
    candidate_rel = Path(raw)
    if candidate_rel.is_absolute() or raw.startswith("/") or raw.startswith("~"):
        raise ValueError("absolute paths are not allowed")
    parts = [part for part in raw.split("/") if part not in {"", "."}]
    if any(part == ".." for part in parts):
        raise ValueError("path traversal is not allowed")

    root_resolved = root.resolve()
    candidate = root_resolved.joinpath(*parts).resolve()
    if candidate != root_resolved and root_resolved not in candidate.parents:
        raise ValueError("resolved path escapes media root")
    return candidate


def safe_static_url_to_path(url: str, prefix: str, root: Path) -> Path | None:
    """Map a known /static/... URL prefix to a filesystem path under root."""
    if not url or not url.startswith(prefix):
        return None
    return safe_resolve_child(root, url[len(prefix) :])
