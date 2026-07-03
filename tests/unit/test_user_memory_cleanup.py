from retrieval.user_memory import UserMemoryManager


class _Neo4jClient:
    def __init__(self, find_results):
        self.find_results = list(find_results)
        self.calls = []

    def execute_query(self, query, params=None):
        self.calls.append({"query": query, "params": params or {}})
        if "RETURN elementId(existing) AS song_id" in query:
            return self.find_results.pop(0) if self.find_results else []
        return []


def test_remove_like_deletes_by_song_element_id_when_artist_match_resolves():
    client = _Neo4jClient([[{"song_id": "song-1"}]])
    manager = UserMemoryManager(neo4j_client=client)

    manager.remove_like("u1", "同名歌", "歌手A")

    delete_call = client.calls[1]
    assert "MATCH (u:User {id: $user_id})-[r:LIKES]->(s:Song)" in delete_call["query"]
    assert "elementId(s) = $song_id" in delete_call["query"]
    assert "title: $song_title" not in delete_call["query"]
    assert delete_call["params"] == {"user_id": "u1", "song_id": "song-1"}


def test_remove_save_falls_back_to_title_and_artist_when_song_is_not_resolved():
    client = _Neo4jClient([[]])
    manager = UserMemoryManager(neo4j_client=client)

    manager.remove_save("u1", "同名歌", "歌手A")

    delete_call = client.calls[1]
    assert "MATCH (u:User {id: $user_id})-[r:SAVES]->(s:Song {title: $song_title})" in delete_call["query"]
    assert "s.artist = $artist" in delete_call["query"]
    assert "PERFORMED_BY" in delete_call["query"]
    assert delete_call["params"] == {"user_id": "u1", "song_title": "同名歌", "artist": "歌手A"}


def test_record_dislike_cleans_like_and_save_by_resolved_song_id_before_dislike():
    client = _Neo4jClient([[{"song_id": "song-1"}], [{"song_id": "song-1"}]])
    manager = UserMemoryManager(neo4j_client=client)

    manager.record_dislike("u1", "同名歌", "歌手A")

    cleanup_call = client.calls[1]
    assert "LIKES|SAVES" in cleanup_call["query"]
    assert "elementId(s) = $song_id" in cleanup_call["query"]
    assert cleanup_call["params"] == {"user_id": "u1", "song_id": "song-1"}

    dislike_call = client.calls[3]
    assert "MERGE (u)-[r:DISLIKES]->(s)" in dislike_call["query"]
    assert dislike_call["params"] == {"user_id": "u1", "song_id": "song-1"}


def test_semantic_preferences_support_mood_scenario_fields_and_dedupe_query():
    client = _Neo4jClient([])
    manager = UserMemoryManager(neo4j_client=client)

    manager.update_semantic_preferences(
        "u1",
        {
            "add_moods": ["Warm"],
            "avoid_moods": ["Energetic"],
            "add_scenarios": ["Rainy Day"],
            "avoid_scenarios": ["Party"],
        },
    )

    update_call = client.calls[1]
    assert "u.add_moods" in update_call["query"]
    assert "u.avoid_moods" in update_call["query"]
    assert "u.add_scenarios" in update_call["query"]
    assert "u.avoid_scenarios" in update_call["query"]
    assert "WHERE NOT (toLower(toString(x)) IN" in update_call["query"]
    assert "reduce(acc = []" in update_call["query"]
    assert update_call["params"]["add_moods"] == ["Warm"]


def test_remove_semantic_preference_allows_only_known_list_fields():
    client = _Neo4jClient([])
    manager = UserMemoryManager(neo4j_client=client)

    ok = manager.remove_semantic_preference("u1", "avoid_moods", "Energetic")
    bad = manager.remove_semantic_preference("u1", "profile_free_text", "secret")

    assert ok is True
    assert bad is False
    remove_call = client.calls[0]
    assert "u.avoid_moods" in remove_call["query"]
    assert "toLower(toString(x))" in remove_call["query"]
    assert remove_call["params"] == {"user_id": "u1", "value": "Energetic"}


def test_clear_semantic_preferences_only_resets_learned_fields():
    client = _Neo4jClient([])
    manager = UserMemoryManager(neo4j_client=client)

    ok = manager.clear_semantic_preferences("u1")

    assert ok is True
    clear_call = client.calls[0]
    assert "u.add_moods = []" in clear_call["query"]
    assert "u.avoid_moods = []" in clear_call["query"]
    assert "u.activity_contexts = []" in clear_call["query"]
    assert "u.preferred_genres" not in clear_call["query"]
    assert "u.mood_tendency = ''" in clear_call["query"]
    assert clear_call["params"] == {"user_id": "u1"}
