import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from api.anonymous_sessions import AnonymousSessionMiddleware, COOKIE, TTL, sign_identity, verify_identity
from api.runtime_context import runtime_context_from_request, assert_exposure_owner


def app(enabled=True):
    application = FastAPI()
    application.add_middleware(AnonymousSessionMiddleware, enabled=enabled)

    @application.get("/api/user-profile")
    async def profile(request: Request, user_id: str = "local_admin"):
        return runtime_context_from_request(request, user_id=user_id).model_dump()

    @application.post("/api/slate-feedback")
    async def feedback(request: Request):
        data = await request.json()
        context = runtime_context_from_request(request, user_id=data.get("user_id", ""))
        assert_exposure_owner(data["exposure"], context)
        return {"success": True}

    return application


@pytest.fixture(autouse=True)
def config(monkeypatch):
    monkeypatch.setenv("SOULTUNER_SESSION_SECRET", "a" * 64)
    monkeypatch.setenv("SOULTUNER_PUBLIC_ORIGIN", "https://music.test")


def test_two_visitors_and_forged_identity_are_isolated():
    application = app()
    with TestClient(application, base_url="https://music.test") as a, TestClient(application, base_url="https://music.test") as b:
        bootstrap = a.get("/api/anonymous-session")
        b.get("/api/anonymous-session")
        cookie = bootstrap.headers["set-cookie"]
        assert "HttpOnly" in cookie and "Secure" in cookie and "SameSite=lax" in cookie
        first = a.get("/api/user-profile").json()
        second = b.get("/api/user-profile", params={"user_id": first["profile_id"]},
                       headers={"X-SoulTuner-Profile": first["profile_id"], "X-SoulTuner-Mode": "personal"}).json()
        assert first["effective_user_id"] != second["effective_user_id"]
        assert not first["training_eligible"] and not first["teacher_log_eligible"]
        result = b.post("/api/slate-feedback", json={"exposure": {"user_id": first["effective_user_id"]}})
        assert result.status_code == 403
        assert a.get("/api/user-profile").json()["profile_id"] == first["profile_id"]


def test_no_cookie_and_unreviewed_route_are_denied():
    with TestClient(app(), base_url="https://music.test") as client:
        assert client.get("/api/user-profile").status_code == 401
        client.get("/api/anonymous-session")
        profiles = client.get("/api/profiles").json()["profiles"]
        assert len(profiles) == 1 and profiles[0]["profile_id"].startswith("anon:")
        assert client.post("/api/profiles").status_code == 403
        assert client.post("/api/settings").status_code == 403
        assert client.get("/api/user-profile").headers["cache-control"] == "private, no-store"


def test_cross_origin_write_is_denied():
    with TestClient(app(), base_url="https://music.test") as client:
        client.get("/api/anonymous-session")
        assert client.post("/api/slate-feedback", headers={"Origin": "https://evil.test"}).status_code == 403
        assert client.get("/api/anonymous-session", headers={"Sec-Fetch-Site": "cross-site"}).status_code == 403


def test_visitor_actual_request_body_is_bounded():
    with TestClient(app(), base_url="https://music.test") as client:
        client.get("/api/anonymous-session")
        response = client.post("/api/slate-feedback", content=b"x" * 65537)
        assert response.status_code == 413
        assert response.json()["error"] == "request_too_large"


def test_clear_learned_uses_signed_owner_and_rejects_identity_override(monkeypatch):
    from types import SimpleNamespace
    from api.visitor_memory import router
    seen = []
    monkeypatch.setattr("services.memory_gateway.get_memory_gateway", lambda: SimpleNamespace(
        clear_learned_preferences=lambda **kw: seen.append(kw["user_id"]) or True))
    application = app()
    application.include_router(router)
    with TestClient(application, base_url="https://music.test") as client:
        client.get("/api/anonymous-session")
        assert client.post("/api/visitor/memory/clear-learned", json={"user_id": "victim"}).status_code == 422
        assert client.post("/api/visitor/memory/clear-learned", json={}).status_code == 200
        assert seen == [client.get("/api/user-profile").json()["effective_user_id"]]


def test_deletion_recovery_is_owner_scoped_and_pending_is_not_success(monkeypatch):
    from types import SimpleNamespace
    from api.visitor_memory import router
    seen = []
    monkeypatch.setattr("services.memory_gateway.get_memory_gateway", lambda: SimpleNamespace(
        retry_pending_deletions=lambda **kw: seen.append(kw) or {"completed": 1, "pending": 1},
        pending_deletion_count=lambda **kw: 1))
    application = app()
    application.include_router(router)
    with TestClient(application, base_url="https://music.test") as client:
        client.get("/api/anonymous-session")
        endpoint = "/api/visitor/memory/retry-deletions"
        assert client.post(endpoint, json={"user_id": "victim"}).status_code == 422
        result = client.post(endpoint, json={}).json()
        assert result["success"] is False and result["remaining"] == 1
        assert seen == [{"user_id": client.get("/api/user-profile").json()["effective_user_id"], "limit": 20}]


def test_token_tampering_expiry_and_future():
    key = "a" * 64
    token = sign_identity("b" * 32, key, 100)
    assert verify_identity(token, key, 101) == "b" * 32
    assert verify_identity(token, "wrong", 101) is None
    assert verify_identity(token, key, 100 + TTL) is None
    assert verify_identity(token, key, 99) is None
    assert verify_identity(token[:-1] + "中", key, 101) is None


def test_local_mode_unchanged():
    with TestClient(app(False)) as client:
        assert client.get("/api/anonymous-session").json() == {"anonymous": False}
        assert client.get("/api/user-profile").json()["effective_user_id"] == "local_admin"


def test_missing_secret_fails_closed(monkeypatch):
    monkeypatch.delenv("SOULTUNER_SESSION_SECRET")
    with TestClient(app(), base_url="https://music.test") as client:
        assert client.get("/api/anonymous-session").status_code == 503


def test_visitor_deletion_never_accepts_another_owner(monkeypatch):
    from api.visitor_memory import router
    from types import SimpleNamespace
    seen = []
    owner = [None]
    def delete(*, user_id, record_id):
        seen.append((user_id, record_id))
        return user_id == owner[0] and record_id == "record-a"
    monkeypatch.setattr("services.memory_gateway.get_memory_gateway",
                        lambda: SimpleNamespace(delete_memory_record=delete))
    application = app()
    application.include_router(router)
    with TestClient(application, base_url="https://music.test") as a, TestClient(application, base_url="https://music.test") as b:
        a.get("/api/anonymous-session")
        b.get("/api/anonymous-session")
        owner[0] = a.get("/api/user-profile").json()["effective_user_id"]
        assert b.post("/api/visitor/memory/forget", json={"record_id": "record-a"}).status_code == 404
        assert a.post("/api/visitor/memory/forget", json={"record_id": "record-a", "user_id": "other"}).status_code == 422
        result = a.post("/api/visitor/memory/forget", json={"record_id": "record-a"})
        assert result.status_code == 200 and result.json()["audit_history_retained"]
        assert seen[-1] == (owner[0], "record-a")


def test_visitor_rate_limit_returns_retry_after():
    with TestClient(app(), base_url="https://music.test") as client:
        client.get("/api/anonymous-session")
        for _ in range(120):
            assert client.get("/api/user-profile").status_code == 200
        response = client.get("/api/user-profile")
        assert response.status_code == 429 and response.headers["Retry-After"] == "60"


def test_cancelled_inference_releases_admission_slot():
    import asyncio
    import time
    async def run():
        async def application(scope, receive, send):
            raise asyncio.CancelledError
        middleware = AnonymousSessionMiddleware(application, enabled=True)
        token = sign_identity("b" * 32, "a" * 64, int(time.time()))
        scope = {"type": "http", "path": "/api/recommendations/stream", "method": "POST",
                 "headers": [(b"cookie", f"{COOKIE}={token}".encode())]}
        async def receive():
            return {"type": "http.request", "body": b"{}", "more_body": False}
        async def send(message):
            pass
        with pytest.raises(asyncio.CancelledError):
            await middleware(scope, receive, send)
        assert middleware.limits.total_active == 0
        assert middleware.limits.active == {}
    asyncio.run(run())


def test_slow_chunked_body_has_total_deadline(monkeypatch):
    import asyncio
    import time
    monkeypatch.setattr("api.anonymous_sessions.BODY_TIMEOUT_SECONDS", .03)

    async def run():
        dispatched = []
        messages = []
        async def application(scope, receive, send):
            dispatched.append(True)
        middleware = AnonymousSessionMiddleware(application, enabled=True)
        token = sign_identity("b" * 32, "a" * 64, int(time.time()))
        scope = {"type": "http", "path": "/api/recommendations/stream", "method": "POST",
                 "headers": [(b"cookie", f"{COOKIE}={token}".encode())]}
        async def receive():
            await asyncio.sleep(.01)
            return {"type": "http.request", "body": b" ", "more_body": True}
        async def send(message):
            messages.append(message)
        await asyncio.wait_for(middleware(scope, receive, send), timeout=1)
        assert messages[0]["status"] == 408
        assert not dispatched
        assert middleware.limits.total_active == 0
    asyncio.run(run())
