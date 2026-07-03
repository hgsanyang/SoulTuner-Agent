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
