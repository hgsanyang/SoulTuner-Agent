import json

from retrieval import recall_sources


class _Client:
    def execute_query(self, _query, _params=None):
        return [
            {
                "title": "晚风心里吹",
                "artists": ["阿梨粤"],
                "album": "Single",
                "audio_url": "/static/audio/cantonese.mp3",
                "cover_url": None,
                "lrc_url": None,
                "language": "Cantonese",
                "region": "Hong Kong",
                "genres": ["Pop"],
                "moods": ["Relaxing"],
                "themes": [],
                "scenarios": [],
                "updated_at": 2,
            },
            {
                "title": "普通话歌曲",
                "artists": ["歌手"],
                "album": "Single",
                "audio_url": "/static/audio/chinese.mp3",
                "cover_url": None,
                "lrc_url": None,
                "language": "Chinese",
                "region": "Mainland China",
                "genres": ["Pop"],
                "moods": ["Happy"],
                "themes": [],
                "scenarios": [],
                "updated_at": 1,
            },
        ]


def test_graph_language_constraint_is_a_recall_signal(monkeypatch):
    monkeypatch.setattr(recall_sources, "get_neo4j_client", lambda: _Client())

    raw = recall_sources.graph_candidate_recall(
        {"language": "Cantonese"},
        {},
        limit=10,
    )
    rows = json.loads(raw)

    assert [row["title"] for row in rows] == ["晚风心里吹"]
    assert rows[0]["language"] == "Cantonese"
    assert rows[0]["similarity_score"] == 4.0


def test_graph_recall_filters_unplayable_song_nodes():
    where = recall_sources._playable_song_where("s")

    assert "s.audio_url IS NOT NULL" in where
    assert "unplayable_stub" in where
