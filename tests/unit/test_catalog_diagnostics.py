from services.catalog_diagnostics import summarize_catalog_bias


def test_catalog_diagnostics_flags_concentrated_catalog_and_feedback():
    catalog_rows = [
        {
            "title": f"Song {idx}",
            "audio_url": f"/static/audio/{idx}.mp3",
            "language": "Chinese",
            "genres": ["Rock"] if idx < 8 else ["R&B"],
            "moods": ["Energetic"],
            "themes": [],
            "scenarios": [],
            "source": "local",
            "has_muq_embedding": idx < 5,
            "has_m2d2_embedding": True,
        }
        for idx in range(10)
    ]
    exposures = [{
        "ts": 10,
        "items": [
            {"source": "Neo4j", "recall_sources": ["graph"], "genres": ["Rock"], "artist": "A"},
            {"source": "Neo4j", "recall_sources": ["graph"], "genres": ["Rock"], "artist": "A"},
        ],
    }]
    slate_feedback = [{
        "rating": "too_familiar",
        "reasons": ["太像我的旧歌单", "想发现更多新歌"],
    }]

    report = summarize_catalog_bias(catalog_rows, exposures, slate_feedback)

    assert report["catalog"]["total_songs"] == 10
    assert report["catalog"]["coverage"]["genres"]["ratio"] == 1.0
    assert report["recent_recommendations"]["top_recall_sources"][0]["label"] == "Graph"
    assert report["slate_feedback"]["count"] == 1
    assert any(warning["code"] == "language_concentration" for warning in report["warnings"])
    assert any(warning["code"] == "user_wants_discovery" for warning in report["warnings"])
