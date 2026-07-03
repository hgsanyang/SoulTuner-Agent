"""Small security helpers for local/admin API routes.

The project defaults to a friendly single-user local mode.  When a public demo
or admin API key is configured, destructive routes must be protected and file
paths must stay inside the intended media roots.
"""

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


def public_demo_enabled() -> bool:
    """Return whether public demo protections should be enforced."""
    try:
        from config.settings import settings

        return bool(getattr(settings, "public_demo_mode", False))
    except Exception:
        return _truthy(os.getenv("PUBLIC_DEMO_MODE"))


def _admin_api_key() -> str:
    try:
        from config.settings import settings

        return str(getattr(settings, "admin_api_key", "") or "").strip()
    except Exception:
        return str(os.getenv("ADMIN_API_KEY", "") or "").strip()


def admin_key_required() -> bool:
    """Local mode stays keyless unless a key or public demo is configured."""
    try:
        from config.settings import settings

        configured = bool(getattr(settings, "api_key_required", False))
    except Exception:
        configured = _truthy(os.getenv("API_KEY_REQUIRED"))
    return configured or bool(_admin_api_key()) or public_demo_enabled()


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
            detail="Admin API key is required in public/demo mode.",
        )
    if x_api_key != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key.",
        )


def reject_public_demo_action(action: str) -> None:
    """Block filesystem-changing actions in public demo mode."""
    if public_demo_enabled():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"{action} is disabled in PUBLIC_DEMO_MODE.",
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
