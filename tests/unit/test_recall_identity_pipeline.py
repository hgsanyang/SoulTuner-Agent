"""Recording identity must survive recall formatting, parsing and fusion."""
import json

from retrieval.recall_sources import _records_to_json
from retrieval.hybrid_retrieval import MusicHybridRetrieval
from retrieval.retrieval_fusion import weighted_rrf
from tools.semantic_search import _fetch_song_details_for_ranked_eids


def test_graph_identity_survives_parser_and_fusion():
    records = [
        {"music_id": mid, "title": "Same", "artist": "Artist", "score": .8}
        for mid in ("studio", "live")
    ]
    raw = _records_to_json(records, "score")
    parsed = MusicHybridRetrieval._parse_engine_results(raw, "graph")
    assert len(parsed) == 2
    assert {item["song"]["music_id"] for item in parsed} == {"studio", "live"}
    fused = weighted_rrf({"graph": parsed, "dense": parsed}, {"graph": 1, "dense": 1})
    assert len(fused) == 2
    assert all(len(item["recall_sources"]) == 2 for item in fused)


def test_rrf_ignores_legacy_title_key_when_recording_id_exists():
    items = [{"key": "same_title", "song": {"title": "Same", "artist": "Artist", "music_id": mid}}
             for mid in ("a", "b")]
    assert len(weighted_rrf({"graph": items}, {"graph": 1})) == 2


def test_dense_metadata_projects_recording_id():
    class Client:
        def execute_query(self, query, params):
            assert "song.music_id AS music_id" in query
            return [{"music_id": "recording-1", "title": "Same", "_eid": "e1"}]

    rows = _fetch_song_details_for_ranked_eids(
        Client(), [{"_eid": "e1", "similarity_score": .8}], backend="muq", backend_name="MuQ")
    assert json.loads(json.dumps(rows))[0]["music_id"] == "recording-1"
