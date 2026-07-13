import asyncio

from services import feedback_logger
from services.memory_gateway import (
    MemoryGateway,
    editable_memory_sections,
    reset_memory_gateway_for_tests,
    summarize_memory_profile,
)
from services.memory_event_store import MemoryEventStore
from services.memory_consolidator import MemoryConsolidator
from services.memory_models import MemoryLayer
from services.memory_retriever import MemoryRelevanceRetriever
from services.memory_semantic_scorer import MemorySemanticScorerUnavailable


class FakePrimary:
    def __init__(self):
        self.events = []
        self.preferences = []
        self.inferred = []
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

    def remember_inferred_preference(self, user_id, record):
        self.inferred.append((user_id, record))
        return True

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


def test_memory_gateway_keeps_feedback_as_evidence_until_consolidated(tmp_path, monkeypatch):
    monkeypatch.setenv("MUSIC_FEEDBACK_DIR", str(tmp_path / "feedback"))
    store = MemoryEventStore(tmp_path / "memory.sqlite3")
    gateway = MemoryGateway(
        primary=FakePrimary(),
        enable_graphzep_sidecar=False,
        event_store=store,
        enable_event_ledger=True,
    )

    asyncio.run(gateway.remember_event(event_type="like", title="A", artist="Singer", user_id="u1"))
    gateway.remember_preference(user_id="u1", preferences={"add_moods": ["Warm"]})
    asyncio.run(gateway.remember_slate_feedback(
        exposure_id="exp-1", rating="too_sad", reasons=["太丧了"], user_id="u1",
    ))

    layers = {record["layer"] for record in gateway.list_memory_records(user_id="u1")}
    assert layers == {"L0", "L1"}
    assert gateway.list_memory_records(user_id="u2") == []


def test_memory_gateway_slate_feedback_does_not_mutate_hot_preferences_directly(tmp_path, monkeypatch):
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
    assert primary.preferences == []
    assert result.preference_update == {}


def test_memory_gateway_consolidates_validated_l2_and_projects_to_hot_path(tmp_path, monkeypatch):
    monkeypatch.setenv("MUSIC_FEEDBACK_DIR", str(tmp_path / "feedback"))
    store = MemoryEventStore(tmp_path / "memory.sqlite3")

    async def generator(_user_id, evidence):
        return {
            "candidates": [
                {
                    "field": "add_moods",
                    "value": "Warm",
                    "scope": "contextual",
                    "confidence": 0.88,
                    "evidence_ids": [evidence[0]["record_id"], evidence[1]["record_id"]],
                    "ttl_days": 45,
                    "retrieval_cues": ["温暖治愈的音乐", "warm healing music"],
                    "decision_summary": "Two independent positive signals support a warm-mood preference.",
                }
            ],
            "summary": "one preference",
        }

    primary = FakePrimary()
    gateway = MemoryGateway(
        primary=primary,
        event_store=store,
        enable_event_ledger=True,
        enable_consolidation=False,
        consolidator=MemoryConsolidator(store, generator=generator, min_evidence=2),
    )
    asyncio.run(gateway.remember_event(event_type="like", title="A", artist="Singer", user_id="u1"))
    asyncio.run(gateway.remember_event(event_type="full_play", title="B", artist="Singer", user_id="u1"))

    report = asyncio.run(gateway.consolidate_user(user_id="u1", force=True))

    assert report["skipped"] is False
    assert len(report["accepted"]) == 1
    assert primary.inferred[0][0] == "u1"
    assert primary.inferred[0][1]["field"] == "add_moods"
    l2 = [row for row in gateway.list_memory_records(user_id="u1") if row["layer"] == "L2"]
    assert len(l2) == 1
    assert l2[0]["expires_at"] > l2[0]["created_at"]


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


def test_structured_mode_does_not_call_optional_sidecars():
    episodic = FakeEpisodic("graphzep", "should not be used")
    gateway = MemoryGateway(
        primary=FakePrimary(),
        episodic_adapters=[episodic],
        memory_mode="structured",
    )

    context = asyncio.run(gateway.retrieve_context(query="rain", user_id="u1"))

    assert context["episodic"] == ""
    assert context["episodic_backends"] == {}


def test_off_mode_ignores_injected_primary_and_disables_profile():
    primary = FakePrimary()
    gateway = MemoryGateway(primary=primary, memory_mode="off")

    result = asyncio.run(
        gateway.remember_event(event_type="like", title="A", artist="Singer", user_id="u1")
    )
    context = asyncio.run(gateway.retrieve_context(query="rain", user_id="u1"))

    assert result.success is True
    assert primary.events == []
    assert context["profile"] == {}
    assert context["memory_trace"]["mode"] == "off"


def test_strict_sidecar_mode_fails_closed_without_adapter():
    try:
        MemoryGateway(
            primary=FakePrimary(),
            episodic_adapters=[],
            memory_mode="sidecar",
            strict_sidecar=True,
        )
    except RuntimeError as exc:
        assert "requires at least one" in str(exc)
    else:
        raise AssertionError("strict sidecar mode must not silently degrade")


def test_memory_trace_includes_owner_and_await_idle(tmp_path):
    store = MemoryEventStore(tmp_path / "memory.sqlite3")
    store.append(
        user_id="u1", layer=MemoryLayer.INFERRED, kind="preference",
        source="memory_consolidator", evidence_id="e1", confidence=0.9,
        payload={
            "field": "add_scenarios",
            "value": "Rainy evenings",
            "retrieval_cues": ["雨夜听歌"],
        },
        memory_key="preference:add_scenarios:rainy evenings",
    )
    episodic = FakeEpisodic("fake", "")
    gateway = MemoryGateway(
        primary=FakePrimary(),
        event_store=store,
        memory_mode="sidecar",
        episodic_adapters=[episodic],
        strict_sidecar=True,
        relevance_retriever=MemoryRelevanceRetriever(),
    )

    async def run():
        await gateway.remember_text(description="user likes rain", user_id="u1")
        idle = await gateway.await_idle(timeout_seconds=1)
        context = await gateway.retrieve_context(query="雨夜想听歌", user_id="u1")
        return idle, context

    idle, context = asyncio.run(run())

    assert idle["idle"] is True
    assert idle["failures"] == []
    assert episodic.writes
    assert context["retrieved_records"][0]["user_id"] == "u1"
    assert context["retrieved_records"][0]["memory_key"].startswith("preference:")


def test_memory_gateway_fails_closed_when_semantic_scorer_is_unavailable(tmp_path):
    class _UnavailableScorer:
        name = "unavailable"

        def score(self, query, documents):
            del query, documents
            raise MemorySemanticScorerUnavailable("missing local model")

    store = MemoryEventStore(tmp_path / "memory.sqlite3")
    store.append(
        user_id="u1", layer=MemoryLayer.EXPLICIT, kind="preference",
        source="user_explicit", evidence_id="e1", confidence=1.0,
        payload={"field": "add_moods", "value": "Calm"},
        memory_key="preference:add_moods:calm",
    )
    gateway = MemoryGateway(
        primary=FakePrimary(),
        event_store=store,
        relevance_retriever=MemoryRelevanceRetriever(
            semantic_scorer=_UnavailableScorer(),
        ),
    )

    context = asyncio.run(gateway.retrieve_context(query="quiet music", user_id="u1"))

    assert context["retrieved_records"] == []
    assert context["memory_trace"]["relevance_error"] == "semantic_scorer_unavailable"


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


def test_memory_profile_diagnostics_count_hot_and_episodic_signals():
    profile = FakePrimary().profile
    diagnostics = summarize_memory_profile(profile, ["mem0"])

    assert diagnostics["hot_path_has_signal"] is True
    assert diagnostics["positive_preference_count"] >= 6
    assert diagnostics["negative_preference_count"] >= 2
    assert diagnostics["context_preference_count"] >= 1
    assert diagnostics["episodic_enabled"] is True
    assert diagnostics["needs_more_feedback"] is False


def test_explain_memory_includes_privacy_preserving_diagnostics():
    gateway = MemoryGateway(primary=FakePrimary(), episodic_adapters=[FakeEpisodic("mem0")])

    report = gateway.explain_memory(user_id="u1")

    assert report["diagnostics"]["episodic_backends"] == ["mem0"]
    assert report["diagnostics"]["hot_path_has_signal"] is True
    assert any(section["field"] == "avoid_moods" for section in report["editable_sections"])


def test_editable_memory_sections_are_ui_ready_and_deduped():
    sections = editable_memory_sections(
        {
            "avoid_moods": ["Energetic", "energetic", "Party"],
            "activity_contexts": ["less_familiar"],
        }
    )
    by_field = {section["field"]: section for section in sections}

    assert by_field["avoid_moods"]["label"] == "避开情绪"
    assert by_field["avoid_moods"]["values"] == ["Energetic", "Party"]
    assert by_field["avoid_moods"]["deletable"] is True
    assert by_field["activity_contexts"]["tone"] == "context"
