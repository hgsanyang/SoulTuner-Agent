from fastapi.testclient import TestClient

from schemas.user_profile import DeveloperSandboxReset, UserProfile


class _Profiles:
    def list_profiles(self):
        return [
            UserProfile(
                profile_id="local_admin",
                display_name="Default",
                profile_type="personal",
            )
        ]

    def create_profile(self, request):
        return UserProfile(
            profile_id="11111111-1111-4111-8111-111111111111",
            display_name=request.display_name,
            profile_type=request.profile_type,
        )

    def reset_developer_sandbox(self, profile_id):
        return DeveloperSandboxReset(
            profile_id=profile_id,
            sandbox_user_id=f"__dev__:{profile_id}",
            reset=True,
        )


def test_profile_api_lists_and_creates(monkeypatch):
    import api.profiles as profile_api
    from api.server import app

    monkeypatch.setattr(profile_api, "_service", lambda: _Profiles())
    client = TestClient(app)
    listed = client.get("/api/profiles")
    created = client.post(
        "/api/profiles",
        json={"display_name": "Alice", "profile_type": "personal"},
    )
    assert listed.status_code == 200
    assert listed.json()["profiles"][0]["profile_id"] == "local_admin"
    assert created.status_code == 200
    assert created.json()["profile"]["display_name"] == "Alice"


def test_profile_api_resets_developer_sandbox(monkeypatch):
    import api.profiles as profile_api
    from api.server import app

    monkeypatch.setattr(profile_api, "_service", lambda: _Profiles())
    client = TestClient(app)
    profile_id = "11111111-1111-4111-8111-111111111111"
    response = client.post(f"/api/profiles/{profile_id}/developer-sandbox/reset")
    assert response.status_code == 200
    assert response.json()["sandbox_user_id"] == f"__dev__:{profile_id}"


def test_developer_mode_blocks_shared_catalog_mutation():
    from api.server import app

    client = TestClient(app)
    response = client.post(
        "/api/acquire-song",
        headers={
            "X-SoulTuner-Profile": "local_admin",
            "X-SoulTuner-Mode": "developer",
        },
        json={"title": "Test", "artist": "Test"},
    )
    assert response.status_code == 409


def test_playlist_stream_uses_profile_mode_context(monkeypatch):
    import api.server as server

    captured = {}

    async def fake_stream_playlist(**kwargs):
        captured.update(kwargs)
        yield 'data: {"type":"complete"}\n\n'

    monkeypatch.setattr(server, "stream_playlist", fake_stream_playlist)
    client = TestClient(server.app)
    response = client.post(
        "/api/playlist/stream",
        headers={
            "X-SoulTuner-Profile": "profile-a",
            "X-SoulTuner-Mode": "developer",
            "X-SoulTuner-Session": "session-a",
        },
        json={"query": "rainy evening"},
    )

    assert response.status_code == 200
    assert captured["user_id"] == "__dev__:profile-a"
    assert captured["runtime_context"].profile_id == "profile-a"
    assert captured["runtime_context"].training_eligible is False
    assert captured["runtime_context"].session_id == "session-a"
