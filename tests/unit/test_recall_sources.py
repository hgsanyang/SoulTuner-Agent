import json

from retrieval import recall_sources


class _Client:
    def execute_query(self, query, _params=None):
        if "RETURN DISTINCT elementId(s) AS eid" in query:
            return [{"eid": "1"}, {"eid": "2"}]
        return [
            {
                "eid": "1",
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
                "eid": "2",
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


def test_artist_recall_falls_back_to_denormalized_song_artist(monkeypatch):
    queries = []

    class _ArtistFallbackClient:
        def execute_query(self, query, _params=None):
            queries.append(query)
            if "RETURN DISTINCT elementId(s) AS eid" in query:
                return [{"eid": "jay-1"}]
            return [{
                "eid": "jay-1",
                "title": "以父之名",
                "artist": "周杰伦",
                "artists": [],
                "audio_url": "/static/audio/jay.mp3",
                "language": "Chinese",
                "region": "Taiwan",
                "genres": ["Hip-Hop"],
                "moods": [],
                "themes": [],
                "scenarios": [],
            }]

    monkeypatch.setattr(recall_sources, "get_neo4j_client", lambda: _ArtistFallbackClient())
    rows = json.loads(recall_sources.graph_candidate_recall(
        {"artist_entities": ["周杰伦"]},
        {},
        limit=10,
    ))

    assert rows[0]["artist"] == "周杰伦"
    assert any("coalesce(s.artist" in query for query in queries)
