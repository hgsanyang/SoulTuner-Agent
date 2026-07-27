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


def test_state_changing_endpoints_require_admin_when_key_is_set(monkeypatch):
    """Setting ADMIN_API_KEY must actually gate the settings/memory writes.
    These endpoints used to have no gate at all."""
    # admin_key_required() reads the settings singleton, so patch that (env is
    # only read at load time).
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


def test_admin_key_alone_does_not_lock_the_whole_ui(monkeypatch):
    """Codex R3.5: setting ADMIN_API_KEY must gate destructive ops but must NOT
    401 the recommend/library/feedback pages — the browser sends no key, and
    locking everything the moment you protect a delete button breaks the app."""
    import api.security as sec

    monkeypatch.setattr(settings, "admin_api_key", "admin-secret")
    monkeypatch.setattr(settings, "public_demo_mode", False)
    assert sec.admin_key_required() is True      # destructive ops ARE gated

    from fastapi.testclient import TestClient

    from api.server import app

    client = TestClient(app)
    # ...but the pages the browser actually loads must not 401
    for method, url in [("get", "/api/library/songs"), ("get", "/api/user-profile")]:
        resp = getattr(client, method)(url)
        assert resp.status_code not in (401, 403), f"{method} {url} locked the UI"


def test_delete_memory_record_requires_the_admin_key(monkeypatch):
    """This DELETE had no dependency of its own for a while, and nothing else
    covers it — hit the REAL endpoint with only an admin key set."""
    monkeypatch.setattr(settings, "admin_api_key", "admin-secret")
    monkeypatch.setattr(settings, "public_demo_mode", False)

    from fastapi.testclient import TestClient

    from api.server import app

    client = TestClient(app)
    # Keyless: rejected by the route's own dependency, before the body runs, so
    # nothing is deleted.
    assert client.delete("/api/memory/record/any-id").status_code in (401, 403)
    assert client.delete("/api/memory/record/any-id",
                         headers={"X-API-Key": "wrong"}).status_code in (401, 403)


def test_delete_memory_record_passes_the_gate_with_a_valid_key(monkeypatch):
    """The gate must not be a brick wall: a valid admin key still reaches the
    handler. The gateway is stubbed so no real memory record is touched."""
    monkeypatch.setattr(settings, "admin_api_key", "admin-secret")
    monkeypatch.setattr(settings, "public_demo_mode", False)

    import services.memory_gateway as mg

    class _StubGateway:
        deleted: list[tuple[str, str]] = []

        def delete_memory_record(self, user_id: str, record_id: str) -> bool:
            self.deleted.append((user_id, record_id))
            return True

    stub = _StubGateway()
    monkeypatch.setattr(mg, "get_memory_gateway", lambda: stub)

    from fastapi.testclient import TestClient

    from api.server import app

    resp = TestClient(app).delete("/api/memory/record/rec-1",
                                  headers={"X-API-Key": "admin-secret"})
    assert resp.status_code == 200, resp.text
    assert stub.deleted == [("local_admin", "rec-1")]


# Destructive routes that are not DELETEs, as (method, path). Method matters:
# GET /api/settings reads the config and must stay open; POST /api/settings
# rewrites it and must not.
_DESTRUCTIVE_NON_DELETE = {
    ("POST", "/api/settings"),
    ("POST", "/api/settings/reset"),
    ("POST", "/api/memory/preference"),
    ("POST", "/api/memory/consolidate"),
    ("POST", "/api/ranking-policy/promote"),
    ("POST", "/api/ranking-policy/rollback"),
    ("PATCH", "/api/library-songs/tags"),
}


def test_every_destructive_route_carries_an_admin_dependency():
    """The whole guarantee now rests here. There is no blanket /api/* gate: it
    could not be used (the browser has no way to send a key), so per-route
    dependencies are the only protection. Every DELETE, plus the destructive
    non-DELETEs above, must carry one — a new route that forgets fails here
    rather than on someone's machine."""
    from api.server import app

    ungated = []
    for route in app.routes:
        methods = getattr(route, "methods", set()) or set()
        path = getattr(route, "path", "")
        destructive = "DELETE" in methods or any(
            (m, path) in _DESTRUCTIVE_NON_DELETE for m in methods
        )
        if not destructive:
            continue
        deps = getattr(getattr(route, "dependant", None), "dependencies", []) or []
        names = {getattr(d.call, "__name__", "") for d in deps}
        if "require_admin_api_key" not in names:
            ungated.append(f"{sorted(methods)} {path}")
    assert not ungated, f"destructive routes with no admin gate: {ungated}"


def test_safe_query_redacts_by_default(monkeypatch):
    from config.logging_config import safe_query

    monkeypatch.delenv("MUSIC_LOG_RAW_QUERY", raising=False)
    out = safe_query("我今天特别难过想一个人静静")
    assert "难过" not in out and "redacted" in out and "len=" in out
    monkeypatch.setenv("MUSIC_LOG_RAW_QUERY", "1")
    assert safe_query("我今天特别难过") == "我今天特别难过"


def test_safe_labels_reports_a_count_not_the_preferences(monkeypatch):
    """A preference array identifies a person about as well as the query text.
    Log how many, plus a hash so two lines can be compared — never the values."""
    from config.logging_config import safe_labels

    monkeypatch.delenv("MUSIC_LOG_RAW_QUERY", raising=False)
    out = safe_labels(["粤语", "City Pop", "失恋"])
    assert "粤语" not in out and "City Pop" not in out and "失恋" not in out
    assert "3 labels" in out
    # same set in a different order must hash the same (that's the whole point of
    # keeping a hash instead of just a count)
    assert safe_labels(["失恋", "粤语", "City Pop"]) == out
    assert safe_labels([]) == "<none>" and safe_labels(None) == "<none>"
    monkeypatch.setenv("MUSIC_LOG_RAW_QUERY", "1")
    assert "City Pop" in safe_labels(["City Pop"])


def test_safe_filters_names_the_active_slots_not_their_values(monkeypatch):
    """`artist_filter='周杰伦'` is request content. Which slots were filled is the
    useful signal; what they were filled with is not."""
    from config.logging_config import safe_filters

    monkeypatch.delenv("MUSIC_LOG_RAW_QUERY", raising=False)
    out = safe_filters(artist="周杰伦", genre="", language="Cantonese", region="")
    assert out == "artist+language"
    assert "周杰伦" not in out and "Cantonese" not in out
    assert safe_filters(artist="", genre="") == "<none>"


def test_no_module_logs_user_preferences_or_filters_verbatim():
    """The regression guard for the whole class: Codex found leftover raw sites in
    three separate rounds, each time in a file nobody thought to grep."""
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    # f-string log calls that interpolate a user-derived preference/filter value
    leaky = re.compile(
        r"logger\.(?:info|debug|warning|error)\(\s*f?[\"'][^\"']*\{[^}]*"
        r"(artist_filter|genre_filter|language_filter|region_filter|"
        r"preferred_genres|preferred_moods|preferred_scenarios|preferred_languages|"
        r"favorite_genres|favorite_artists|portrait_text|profile_text)")
    offenders = []
    for directory in ("api", "agent", "tools", "services", "retrieval"):
        for path in sorted((root / directory).rglob("*.py")):
            for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if leaky.search(line):
                    offenders.append(f"{path.relative_to(root)}:{i}")
    assert not offenders, f"user preferences/filters logged verbatim: {offenders}"
