import json

from services.music_knowledge_graph import UPSERT_KNOWLEDGE_CARD_CYPHER, knowledge_card_params


def test_knowledge_card_params_are_graph_safe():
    params = knowledge_card_params(
        {
            "kind": "song",
            "title": "Anchor",
            "artist": "Teleman",
            "summary": "A gentle indie song.",
            "facts": ["Fact A"],
            "source": "web",
        }
    )

    assert params["key"] == "song::anchor::teleman"
    assert params["confidence"] == 0.68
    assert json.loads(params["facts_json"]) == ["Fact A"]


def test_knowledge_graph_query_does_not_create_song_nodes():
    assert "OPTIONAL MATCH (s:Song)" in UPSERT_KNOWLEDGE_CARD_CYPHER
    assert "MERGE (s:Song" not in UPSERT_KNOWLEDGE_CARD_CYPHER
    assert "MERGE (k:KnowledgeCard" in UPSERT_KNOWLEDGE_CARD_CYPHER
