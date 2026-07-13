from retrieval.user_memory import UserMemoryManager


class FakeNeo4j:
    def __init__(self):
        self.calls = []

    def execute_query(self, query, parameters=None):
        parameters = parameters or {}
        self.calls.append((query, parameters))
        if "LIKES|SAVES|LISTENED_TO" in query:
            return []
        if "properties(u) AS p" in query:
            return [
                {
                    "avoid_genres": [],
                    "add_genres": [],
                    "add_moods": [],
                    "avoid_moods": ["Sad"],
                    "add_scenarios": [],
                    "avoid_scenarios": [],
                    "add_artists": [],
                    "avoid_artists": [],
                    "mood_tendency": "",
                    "activity_contexts": [],
                    "language_preference": "",
                    "preferred_genres": None,
                    "preferred_moods": None,
                    "preferred_scenarios": None,
                    "preferred_languages": None,
                    "preferences_updated_at": 0,
                }
            ]
        if "RETURN count(p) AS expired_count" in query:
            return [{"expired_count": 0}]
        if "RETURN p.memory_key AS memory_key" in query and "ORDER BY" in query:
            return [
                {
                    "memory_key": "preference:add_moods:warm",
                    "field": "add_moods",
                    "value": "Warm",
                    "confidence": 0.88,
                    "expires_at": parameters["now"] + 1000,
                },
                {
                    "memory_key": "preference:add_moods:sad",
                    "field": "add_moods",
                    "value": "Sad",
                    "confidence": 0.9,
                    "expires_at": parameters["now"] + 1000,
                },
            ]
        if "MERGE (p:InferredPreference" in query:
            return [{"memory_key": parameters["memory_key"]}]
        return []


def test_active_l2_merges_into_profile_but_explicit_conflict_wins():
    manager = UserMemoryManager(neo4j_client=FakeNeo4j())

    profile = manager.get_user_preferences("u1")

    assert "Warm" in profile["add_moods"]
    assert "Sad" not in profile["add_moods"]
    assert profile["avoid_moods"] == ["Sad"]
    assert [item["value"] for item in profile["inferred_preferences"]] == ["Warm"]


def test_l2_projection_preserves_expiry_and_evidence():
    client = FakeNeo4j()
    manager = UserMemoryManager(neo4j_client=client)

    ok = manager.upsert_inferred_preference(
        "u1",
        {
            "memory_key": "preference:add_moods:warm",
            "field": "add_moods",
            "value": "Warm",
            "scope": "contextual",
            "confidence": 0.88,
            "evidence_ids": ["e1", "e2"],
            "expires_at": 999999,
            "ledger_record_id": "l2-1",
        },
    )

    assert ok is True
    projection = next(params for query, params in client.calls if "MERGE (p:InferredPreference" in query)
    assert projection["expires_at"] == 999999
    assert projection["evidence_ids"] == ["e1", "e2"]
