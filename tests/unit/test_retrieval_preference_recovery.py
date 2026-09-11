import pytest

from retrieval import hybrid_retrieval as hybrid
from retrieval.retrieval_fusion import apply_hard_filters


def test_dislike_identity_does_not_exclude_other_recordings():
    songs = [{"song": {"music_id": i, "title": "Same", "artist": "Artist"}} for i in ("a", "b")]
    assert apply_hard_filters(songs, {}, disliked_songs=[songs[0]["song"]]) == [songs[1]]
    # A legacy dislike with a known artist remains usable without title-wide exclusion.
    other = {"song": {"title": "Same", "artist": "Other"}}
    assert apply_hard_filters(songs + [other], {}, disliked_songs=[
        {"title": "Same", "artist": "Artist"},
    ]) == [other]


def test_dislike_outage_is_not_cached_as_empty(monkeypatch):
    class Client:
        calls = 0
        def execute_query(self, *args):
            raise AssertionError("must use read path")
        def execute_read_query(self, *args):
            self.calls += 1
            if self.calls == 1:
                raise ConnectionError("offline")
            return [{"music_id": "a"}]
    client = Client()
    monkeypatch.setattr("retrieval.neo4j_client.get_neo4j_client", lambda: client)
    retriever = hybrid.MusicHybridRetrieval()
    with pytest.raises(ConnectionError):
        retriever._get_disliked_songs("u")
    assert retriever._get_disliked_songs("u") == [{"music_id": "a"}]


def test_profile_failure_does_not_poison_success_cache(monkeypatch):
    class Gateway:
        calls = 0
        def get_user_profile(self, *args):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("offline")
            return {"favorite_genres": ["ambient"]}
    gateway = Gateway()
    monkeypatch.setattr("services.memory_gateway.get_memory_gateway", lambda: gateway)
    monkeypatch.setattr(hybrid, "_user_pref_cache", {})
    assert hybrid._load_user_preferences("u")["unavailable"] is True
    assert "u" not in hybrid._user_pref_cache
    assert "ambient" in hybrid._load_user_preferences("u")["genres"]


def test_profile_cache_expires_and_has_bounded_owners(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(hybrid, "_user_pref_cache", {})
    monkeypatch.setattr(hybrid, "_USER_PREF_CACHE_CAPACITY", 2)
    now = [1.0]
    monkeypatch.setattr(hybrid.time, "monotonic", lambda: now[0])
    calls = []
    def read(user):
        calls.append(user)
        return {}
    monkeypatch.setattr("services.memory_gateway.get_memory_gateway", lambda: SimpleNamespace(get_user_profile=read))
    hybrid._load_user_preferences("a")
    hybrid._load_user_preferences("a")
    assert calls == ["a"]
    now[0] = 32.0
    hybrid._load_user_preferences("a")
    assert calls == ["a", "a"]
    hybrid._load_user_preferences("b")
    hybrid._load_user_preferences("c")
    assert len(hybrid._user_pref_cache) == 2


def test_invalidation_during_profile_read_cannot_repopulate_stale_cache(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(hybrid, "_user_pref_cache", {})
    def read(user):
        hybrid.invalidate_user_pref_cache(user)
        return {"favorite_genres": ["old"]}
    monkeypatch.setattr("services.memory_gateway.get_memory_gateway", lambda: SimpleNamespace(get_user_profile=read))
    hybrid._load_user_preferences("a")
    assert "a" not in hybrid._user_pref_cache
