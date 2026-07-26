from pathlib import Path

import pytest

from api.security import HTTPException, reject_shared_safe_action, safe_resolve_child, safe_static_url_to_path
from config.settings import settings


def test_safe_resolve_child_allows_nested_relative_path(tmp_path: Path):
    resolved = safe_resolve_child(tmp_path, "album/song.mp3")
    assert resolved == (tmp_path / "album" / "song.mp3").resolve()


@pytest.mark.parametrize("bad_path", ["../secret.txt", "/tmp/secret.txt", r"..\\secret.txt", "~/.ssh/id_rsa"])
def test_safe_resolve_child_rejects_escape_paths(tmp_path: Path, bad_path: str):
    with pytest.raises(ValueError):
        safe_resolve_child(tmp_path, bad_path)


def test_safe_static_url_to_path_maps_only_known_prefix(tmp_path: Path):
    assert safe_static_url_to_path("/static/audio/a.mp3", "/static/audio/", tmp_path) == (
        tmp_path / "a.mp3"
    ).resolve()
    assert safe_static_url_to_path("/static/covers/a.jpg", "/static/audio/", tmp_path) is None


def test_reject_shared_safe_action_blocks_when_enabled(monkeypatch):
    monkeypatch.setattr(settings, "public_demo_mode", True)
    with pytest.raises(HTTPException) as exc:
        reject_shared_safe_action("delete song")
    assert exc.value.status_code == 403


def test_reject_shared_safe_action_allows_local_mode(monkeypatch):
    monkeypatch.setattr(settings, "public_demo_mode", False)
    reject_shared_safe_action("delete song")


def test_state_changing_endpoints_require_admin_when_auth_on(monkeypatch):
    """When API_KEY_REQUIRED is on, settings/memory writes must reject a keyless
    caller. These endpoints used to have no gate at all — a LAN peer could rewrite
    settings or delete memory."""
    # admin_key_required() reads the settings singleton, so patch that (env is
    # only read at load time).
    monkeypatch.setattr(settings, "api_key_required", True)
    monkeypatch.setattr(settings, "admin_api_key", "secret-key")
    from fastapi.testclient import TestClient

    from api.server import app

    client = TestClient(app)
    # Bodies are valid so the ONLY reason a keyless call can be rejected is auth
    # (a malformed body would 422 before the dependency runs and mask the gate).
    # Keyless calls are safe to fire: the gate rejects them BEFORE the body runs,
    # so nothing is actually reset or deleted.
    guarded = [
        ("post", "/api/settings", {"json": {}}),
        ("post", "/api/settings/reset", {}),
        ("post", "/api/memory/preference", {"json": {"preferences": {"genres": ["rock"]}}}),
        ("delete", "/api/memory/preference?field=genres&value=rock", {}),
        ("delete", "/api/memory/profile", {}),
    ]
    for method, url, kwargs in guarded:
        resp = getattr(client, method)(url, **kwargs)
        assert resp.status_code in (401, 403), f"{method} {url} was not gated: {resp.status_code}"

    # A valid key must pass the gate. Use /api/settings only: firing a valid-key
    # call at /api/settings/reset would actually reload settings and disarm the
    # patched flag for the rest of the suite.
    ok = client.post("/api/settings", json={}, headers={"X-API-Key": "secret-key"})
    assert ok.status_code not in (401, 403), f"valid key rejected: {ok.status_code}"


def test_blanket_gate_covers_routes_with_no_explicit_dependency(monkeypatch):
    """The gap Codex found: state-changing routes with NO require_admin dep
    (profile edit, memory-record delete) were reachable in LAN mode. The blanket
    /api/* middleware must catch them, and reads too, while /health stays open."""
    monkeypatch.setattr(settings, "api_key_required", True)
    monkeypatch.setattr(settings, "admin_api_key", "secret-key")
    from fastapi.testclient import TestClient

    from api.server import app

    client = TestClient(app)
    # routes that carry no explicit auth dependency:
    for method, url in [
        ("post", "/api/user-profile"),
        ("delete", "/api/memory/record/abc"),
        ("post", "/api/recommendations"),        # a read-ish POST, still /api/*
        ("get", "/api/library/songs"),           # a plain GET is gated too
    ]:
        resp = getattr(client, method)(url)
        assert resp.status_code in (401, 403), f"{method} {url} bypassed the gate: {resp.status_code}"
    # liveness must remain open (docker healthcheck, no key)
    assert client.get("/health").status_code == 200


def test_api_request_needs_key_prefix_rule():
    from api.security import api_request_needs_key

    assert api_request_needs_key("/api/settings", "POST") is True
    assert api_request_needs_key("/api/library/songs", "GET") is True
    assert api_request_needs_key("/api/settings", "OPTIONS") is False   # CORS preflight
    assert api_request_needs_key("/health", "GET") is False
    assert api_request_needs_key("/audio/x.mp3", "GET") is False        # static, not /api/


def test_safe_query_redacts_by_default(monkeypatch):
    from config.logging_config import safe_query

    monkeypatch.delenv("MUSIC_LOG_RAW_QUERY", raising=False)
    out = safe_query("我今天特别难过想一个人静静")
    assert "难过" not in out and "redacted" in out and "len=" in out
    monkeypatch.setenv("MUSIC_LOG_RAW_QUERY", "1")
    assert safe_query("我今天特别难过") == "我今天特别难过"
