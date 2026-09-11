from retrieval.candidate_identity import candidate_identity
from retrieval.post_recall_adjustments import _metadata_for


def test_same_title_recordings_keep_separate_metadata():
    a = {"title": "Same", "artist": "A", "music_id": "1"}
    b = {"title": "Same", "artist": "A", "music_id": "2"}
    metadata = {candidate_identity(a)["key"]: {"ts_beta": 9},
                candidate_identity(b)["key"]: {"ts_beta": 1}}
    assert _metadata_for({"song": a}, metadata)["ts_beta"] == 9
    assert _metadata_for({"song": b}, metadata)["ts_beta"] == 1
    assert candidate_identity({"title": "Same", "artist": "A"})["key"] != candidate_identity(
        {"title": "Same", "artist": "B"})["key"]
