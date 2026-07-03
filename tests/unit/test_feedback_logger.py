import json

from services import feedback_logger


def test_exposure_and_event_jsonl_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("MUSIC_FEEDBACK_DIR", str(tmp_path))

    exposure_id = feedback_logger.log_exposure(
        query="想听空灵女声",
        request_id="req-1",
        intent_type="vector_search",
        recommendations=[
            {
                "song": {"title": "A", "artist": "Singer", "language": "Chinese"},
                "similarity_score": 0.9,
                "_semantic_score": 0.8,
                "_acoustic_score": 0.7,
                "_personal_score": 0.6,
                "_source_ranks": {"dense": 1},
            }
        ],
    )
    feedback_logger.log_user_event(
        event_type="like",
        song_title="A",
        artist="Singer",
        exposure_id=exposure_id,
    )

    exposures = feedback_logger.load_jsonl(tmp_path / "exposures.jsonl")
    events = feedback_logger.load_jsonl(tmp_path / "events.jsonl")

    assert exposures[0]["exposure_id"] == "req-1"
    assert "query" not in exposures[0]
    assert len(exposures[0]["query_hash"]) == 64
    assert exposures[0]["items"][0]["semantic_score"] == 0.8
    assert exposures[0]["items"][0]["source_ranks"] == {"dense": 1}
    assert events[0]["event_type"] == "like"


def test_log_slate_feedback_jsonl_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("MUSIC_FEEDBACK_DIR", str(tmp_path))

    feedback_id = feedback_logger.log_slate_feedback(
        exposure_id="exp-1",
        rating="too_familiar",
        reasons=["太像我的旧歌单", "想发现更多新歌"],
        note="少一点已收藏的民谣",
    )

    rows = feedback_logger.load_jsonl(tmp_path / feedback_logger.SLATE_FEEDBACK_FILE)

    assert rows[0]["feedback_id"] == feedback_id
    assert rows[0]["exposure_id"] == "exp-1"
    assert rows[0]["rating"] == "too_familiar"
    assert rows[0]["reasons"] == ["太像我的旧歌单", "想发现更多新歌"]
    assert rows[0]["note"] == "少一点已收藏的民谣"


def test_estimate_tri_anchor_weights_prefers_positive_features():
    exposures = [
        {
            "exposure_id": "e1",
            "items": [
                {
                    "title": "A",
                    "artist": "Singer",
                    "semantic_score": 0.9,
                    "acoustic_score": 0.7,
                    "personal_score": 0.4,
                }
            ],
        }
    ]
    events = [{"event_type": "like", "title": "A", "artist": "Singer", "exposure_id": "e1"}]

    report = feedback_logger.estimate_tri_anchor_weights(exposures, events)

    assert report["matched_events"] == 1
    assert report["positive_events"] == 1
    assert report["weights"]["semantic"] > report["weights"]["personal"]


def test_build_feedback_training_rows_joins_by_exposure_and_song():
    exposures = [{
        "exposure_id": "e1",
        "items": [{
            "title": "A",
            "artist": "Singer",
            "semantic_score": 0.9,
            "acoustic_score": 0.2,
            "personal_score": 0.1,
        }],
    }]
    events = [{"event_type": "dislike", "title": "A", "artist": "Singer", "exposure_id": "e1"}]

    rows = feedback_logger.build_feedback_training_rows(exposures, events)

    assert rows[0]["label"] == 0
    assert rows[0]["features"] == {"semantic": 0.9, "acoustic": 0.2, "personal": 0.1}


def test_learn_tri_anchor_weights_requires_enough_labeled_data():
    report = feedback_logger.learn_tri_anchor_weights([], [], min_events=2)

    assert report["status"] == "insufficient_data"
    assert "weights" not in report


def test_learn_tri_anchor_weights_reduces_log_loss_on_audited_rows():
    exposures = []
    events = []
    for idx in range(6):
        exposure_id = f"p{idx}"
        exposures.append({
            "exposure_id": exposure_id,
            "items": [{
                "title": f"Like {idx}",
                "artist": "Singer",
                "semantic_score": 0.9,
                "acoustic_score": 0.8,
                "personal_score": 0.2,
            }],
        })
        events.append({"event_type": "like", "title": f"Like {idx}", "artist": "Singer", "exposure_id": exposure_id})
    for idx in range(6):
        exposure_id = f"n{idx}"
        exposures.append({
            "exposure_id": exposure_id,
            "items": [{
                "title": f"Skip {idx}",
                "artist": "Singer",
                "semantic_score": 0.1,
                "acoustic_score": 0.2,
                "personal_score": 0.8,
            }],
        })
        events.append({"event_type": "skip", "title": f"Skip {idx}", "artist": "Singer", "exposure_id": exposure_id})

    report = feedback_logger.learn_tri_anchor_weights(exposures, events, min_events=4)

    assert report["status"] == "ok"
    assert report["matched_events"] == 12
    assert report["learned_log_loss"] < report["baseline_log_loss"]
    assert abs(sum(report["weights"].values()) - 1.0) < 0.001


def test_load_learned_tri_anchor_weights_normalizes(tmp_path, monkeypatch):
    monkeypatch.setenv("MUSIC_FEEDBACK_DIR", str(tmp_path))
    (tmp_path / feedback_logger.WEIGHTS_FILE).write_text(
        json.dumps({"weights": {"semantic": 2, "acoustic": 1, "personal": 1}}),
        encoding="utf-8",
    )

    weights = feedback_logger.load_learned_tri_anchor_weights()

    assert weights == {"semantic": 0.5, "acoustic": 0.25, "personal": 0.25}
