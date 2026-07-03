import asyncio

from services import feedback_logger
from services.memory_gateway import (
    MemoryGateway,
    derive_preferences_from_slate_feedback,
    reset_memory_gateway_for_tests,
)


class FakePrimary:
    def __init__(self):
        self.events = []
        self.preferences = []
        self.profile = {
            "preferred_genres": ["Indie"],
            "favorite_genres": ["Folk"],
            "favorite_moods": ["Dreamy"],
            "favorite_themes": ["Healing"],
            "favorite_scenarios": ["Late Night"],
            "add_moods": ["Warm"],
            "avoid_genres": ["EDM"],
            "avoid_moods": ["Energetic"],
            "add_scenarios": ["Rainy Day"],
            "avoid_scenarios": ["Party"],
            "activity_contexts": ["less_familiar"],
        }

    def remember_event(self, event_type, title, artist, user_id, extra):
        self.events.append((event_type, title, artist, user_id, extra))

    def remember_preference(self, user_id, preferences):
        self.preferences.append((user_id, preferences))

    def get_user_profile(self, user_id, limit=30):
        return self.profile

    def delete_memory(self, user_id, *, title="", artist="", memory_type=""):
        return True

    def clear_learned_preferences(self, user_id):
        self.cleared_user_id = user_id
        return True


class FakePrimaryWithForget(FakePrimary):
    def __init__(self):
        super().__init__()
        self.manager = self
        self.removed = []

    def remove_semantic_preference(self, user_id, field, value):
        self.removed.append((user_id, field, value))
        return True


class FakeEpisodic:
    def __init__(self, name="fake", context=""):
        self.name = name
        self.context = context
        self.writes = []

    async def remember_text(self, description, *, user_id="local_admin", extra=None):
        self.writes.append((description, user_id, extra or {}))
        return True

    async def retrieve_context(self, query, *, user_id="local_admin", max_facts=8):
        return self.context


def test_slate_feedback_derives_conservative_avoid_preferences():
    prefs = derive_preferences_from_slate_feedback(
        rating="too_noisy",
        reasons=["太吵了"],
        note="少一点 EDM 和土嗨",
    )

    assert "EDM" in prefs["avoid_genres"]
    assert "Energetic" in prefs["avoid_moods"]
    assert "低动态" in prefs["mood_tendency"]


def test_memory_gateway_records_event_and_local_jsonl(tmp_path, monkeypatch):
    monkeypatch.setenv("MUSIC_FEEDBACK_DIR", str(tmp_path))
    primary = FakePrimary()
    gateway = MemoryGateway(primary=primary, enable_graphzep_sidecar=False)

    result = asyncio.run(
        gateway.remember_event(
            event_type="like",
            title="A",
            artist="Singer",
            user_id="u1",
            exposure_id="exp-1",
            extra={"position": 2},
        )
    )

    assert result.success
    assert result.feedback_event_id
    assert primary.events[0][0] == "like"
    rows = feedback_logger.load_jsonl(tmp_path / "events.jsonl")
    assert rows[0]["event_type"] == "like"
    assert rows[0]["exposure_id"] == "exp-1"


def test_memory_gateway_slate_feedback_updates_hot_preferences(tmp_path, monkeypatch):
    monkeypatch.setenv("MUSIC_FEEDBACK_DIR", str(tmp_path))
    primary = FakePrimary()
    gateway = MemoryGateway(primary=primary, enable_graphzep_sidecar=False)

    result = asyncio.run(
        gateway.remember_slate_feedback(
            exposure_id="exp-1",
            rating="too_sad",
            reasons=["太丧了"],
            note="想更温暖",
            user_id="u1",
        )
    )

    assert result.slate_feedback_id
    assert primary.preferences[0][0] == "u1"
    assert "Sad" in primary.preferences[0][1]["avoid_moods"]
    assert "Warm" in primary.preferences[0][1]["add_moods"]


def test_memory_gateway_supports_multiple_episodic_sidecars(tmp_path, monkeypatch):
    monkeypatch.setenv("MUSIC_FEEDBACK_DIR", str(tmp_path))
    primary = FakePrimary()
    episodic_a = FakeEpisodic("a", "用户喜欢雨天 lo-fi")
    episodic_b = FakeEpisodic("b", "用户不喜欢太吵的 EDM")
    gateway = MemoryGateway(
        primary=primary,
        episodic_adapters=[episodic_a, episodic_b],
    )

    result = asyncio.run(
        gateway.remember_text(
            description="用户说喜欢安静雨天歌",
            user_id="u1",
            extra={"source": "test"},
        )
    )
    context = asyncio.run(gateway.retrieve_context(query="雨天", user_id="u1"))

    assert result.graphzep_scheduled
    assert "雨天 lo-fi" in context["episodic"]
    assert "太吵" in context["episodic"]
    assert set(context["episodic_backends"]) == {"a", "b"}


def test_memory_gateway_can_forget_one_hot_preference():
    primary = FakePrimaryWithForget()
    gateway = MemoryGateway(primary=primary, enable_graphzep_sidecar=False)

    ok = gateway.forget_preference_item(
        user_id="u1",
        field="avoid_moods",
        value="Energetic",
    )

    assert ok
    assert primary.removed == [("u1", "avoid_moods", "Energetic")]


def test_memory_gateway_can_clear_learned_preferences():
    primary = FakePrimary()
    gateway = MemoryGateway(primary=primary, enable_graphzep_sidecar=False)

    ok = gateway.clear_learned_preferences(user_id="u1")

    assert ok
    assert primary.cleared_user_id == "u1"


def test_hybrid_pref_loader_reads_gateway_hot_profile(monkeypatch):
    from retrieval.hybrid_retrieval import _load_user_preferences, invalidate_user_pref_cache

    primary = FakePrimary()
    reset_memory_gateway_for_tests(MemoryGateway(primary=primary, enable_graphzep_sidecar=False))
    invalidate_user_pref_cache("u1")

    prefs = _load_user_preferences("u1")

    assert "indie" in prefs["genres"]
    assert "folk" in prefs["genres"]
    assert "dreamy" in prefs["moods"]
    assert "warm" in prefs["moods"]
    assert "healing" in prefs["themes"]
    assert "late night" in prefs["scenarios"]
    assert "rainy day" in prefs["scenarios"]
    assert "energetic" in prefs["avoid_moods"]
    assert "edm" in prefs["expanded_avoid_genres"]
