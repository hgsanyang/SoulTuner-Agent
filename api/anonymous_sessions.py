"""Signed anonymous browser identities for the shared, same-origin API."""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import os
import re
import secrets
import time

from starlette.requests import Request
from starlette.responses import JSONResponse

COOKIE = "__Host-soultuner-visitor"
TTL = 30 * 24 * 3600
MAX_BODY_BYTES = 64 * 1024
BODY_TIMEOUT_SECONDS = 10
# Explicitly reviewed visitor routes. Management and unreviewed new APIs are
# denied, rather than accidentally inheriting anonymous access.
READ = {
    "/api/profiles",
    "/api/user-profile", "/api/user-portrait", "/api/memory/profile",
    "/api/memory/profile-views", "/api/song-feedback/history",
    "/api/liked-songs", "/api/disliked-songs", "/api/library-songs",
}
WRITE = {
    "/api/visitor/memory/retry-deletions",
    "/api/visitor/memory/retry-writes",
    "/api/visitor/memory/forget",
    "/api/visitor/memory/clear-learned",
    "/api/recommendations", "/api/recommendations/stream",
    "/api/playlist", "/api/playlist/stream", "/api/search",
    "/api/user-profile", "/api/user-portrait/refresh",
    "/api/user-event", "/api/song-feedback", "/api/slate-feedback",
}


def sign_identity(subject: str, key: str, now: int) -> str:
    payload = f"v1.{subject}.{now}"
    signature = hmac.new(key.encode(), payload.encode(), hashlib.sha256).hexdigest()
    return f"{payload}.{signature}"


def verify_identity(token: str, key: str, now: int) -> str | None:
    if len(token) > 180:
        return None
    parts = token.split(".")
    if len(parts) != 4 or parts[0] != "v1" or not re.fullmatch(r"[0-9a-f]{32}", parts[1]):
        return None
    if not re.fullmatch(r"[0-9a-f]{64}", parts[3]):
        return None
    try:
        issued = int(parts[2])
    except ValueError:
        return None
    if issued > now or now - issued >= TTL:
        return None
    expected = sign_identity(parts[1], key, issued)
    return parts[1] if hmac.compare_digest(token, expected) else None


class AnonymousSessionMiddleware:
    def __init__(self, app, enabled=None):
        from api.visitor_limits import VisitorLimits
        self.app = app
        self.enabled = enabled
        self.limits = VisitorLimits()

    async def __call__(self, scope, receive, send):
        from api.security import shared_safe_mode_enabled

        enabled = self.enabled if self.enabled is not None else shared_safe_mode_enabled()
        if scope["type"] != "http" or not scope["path"].startswith("/api/"):
            return await self.app(scope, receive, send)
        bootstrap = scope["path"] == "/api/anonymous-session"
        if not enabled:
            if bootstrap:
                return await JSONResponse({"anonymous": False})(scope, receive, send)
            return await self.app(scope, receive, send)
        key = os.getenv("SOULTUNER_SESSION_SECRET", "")
        origin = os.getenv("SOULTUNER_PUBLIC_ORIGIN", "").rstrip("/")
        if len(key) < 32 or not origin.startswith("https://"):
            return await JSONResponse({"error": "anonymous_sessions_not_configured"}, 503)(scope, receive, send)
        request = Request(scope)
        method, path = scope["method"], scope["path"]
        if request.headers.get("sec-fetch-site") == "cross-site" or (
            request.headers.get("origin") and request.headers["origin"] != origin
        ):
            return await JSONResponse({"error": "cross_origin_denied"}, 403)(scope, receive, send)
        if not ((bootstrap and method == "GET") or (method == "GET" and path in READ)
                or (method == "POST" and path in WRITE)):
            return await JSONResponse({"error": "visitor_route_denied"}, 403)(scope, receive, send)
        now = int(time.time())
        if not self.limits.allow("global", 600):
            return await JSONResponse({"error": "rate_limited"}, 429, headers={"Retry-After": "60"})(scope, receive, send)
        subject = verify_identity(request.cookies.get(COOKIE, ""), key, now)
        if bootstrap:
            if subject is None and not self.limits.allow("bootstrap", 60):
                return await JSONResponse({"error": "rate_limited"}, 429, headers={"Retry-After": "60"})(scope, receive, send)
            response = JSONResponse({"anonymous": True, "expires_in": TTL})
            response.headers["Cache-Control"] = "no-store"
            if subject is None:
                response.set_cookie(COOKIE, sign_identity(secrets.token_hex(16), key, now),
                                    max_age=TTL, secure=True, httponly=True, samesite="lax", path="/")
            return await response(scope, receive, send)
        if subject is None:
            return await JSONResponse({"error": "anonymous_session_required"}, 401)(scope, receive, send)
        if not self.limits.allow("visitor:" + subject, 120):
            return await JSONResponse({"error": "rate_limited"}, 429, headers={"Retry-After": "60"})(scope, receive, send)
        # Bound actual bytes, including chunked bodies without Content-Length.
        # Read before dispatch so a downstream handler cannot emit a response first.
        if method == "POST":
            body = bytearray()
            try:
                # One total deadline, not a fresh timeout for each tiny chunk.
                async with asyncio.timeout(BODY_TIMEOUT_SECONDS):
                    while True:
                        message = await receive()
                        if message["type"] == "http.disconnect":
                            return
                        body.extend(message.get("body", b""))
                        if len(body) > MAX_BODY_BYTES:
                            return await JSONResponse({"error": "request_too_large"}, 413)(scope, receive, send)
                        if not message.get("more_body", False):
                            break
            except TimeoutError:
                return await JSONResponse({"error": "request_body_timeout"}, 408)(scope, receive, send)
            original_receive = receive
            delivered = False

            async def bounded_receive():
                nonlocal delivered
                if not delivered:
                    delivered = True
                    return {"type": "http.request", "body": bytes(body), "more_body": False}
                return await original_receive()

            receive = bounded_receive
        scope.setdefault("state", {})["anonymous_subject"] = "anon:" + subject
        if path == "/api/profiles":
            response = JSONResponse({"success": True, "profiles": [{
                "profile_id": "anon:" + subject, "display_name": "我的访客档案",
                "profile_type": "test", "status": "active",
            }]}, headers={"Cache-Control": "private, no-store"})
            return await response(scope, receive, send)

        async def private_send(message):
            if message["type"] == "http.response.start":
                message["headers"] = [(k, v) for k, v in message.get("headers", []) if k.lower() != b"cache-control"]
                message["headers"].append((b"cache-control", b"private, no-store"))
            await send(message)
        expensive = method == "POST" and path in {
            "/api/recommendations", "/api/recommendations/stream", "/api/playlist",
            "/api/playlist/stream", "/api/search", "/api/user-portrait/refresh",
        }
        if expensive and (not self.limits.allow("inference:" + subject, 10) or not self.limits.enter(subject)):
            return await JSONResponse({"error": "inference_capacity_limited"}, 429, headers={"Retry-After": "10"})(scope, receive, send)
        try:
            await self.app(scope, receive, private_send)
        finally:
            if expensive:
                self.limits.leave(subject)
