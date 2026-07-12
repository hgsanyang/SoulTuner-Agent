import pytest

from retrieval.acoustic_probe import percentile_scores, raw_probe_delta, score_catalog_embeddings


def test_raw_probe_delta_prefers_positive_anchor():
    audio = [1.0, 0.0]
    positive = [1.0, 0.0]
    negative = [0.0, 1.0]

    assert raw_probe_delta(audio, positive, negative) > 0


def test_score_catalog_embeddings_percentile_normalises_fields():
    probe_embeddings = {
        "vocalness": ([1.0, 0.0], [0.0, 1.0]),
        "drumness": ([0.0, 1.0], [1.0, 0.0]),
        "energy": ([1.0, 1.0], [-1.0, -1.0]),
    }
    scores = score_catalog_embeddings(
        {
            "vocal": [1.0, 0.0],
            "instrumental": [0.0, 1.0],
        },
        probe_embeddings,
    )

    assert scores["vocal"]["acoustic_vocalness"] > scores["instrumental"]["acoustic_vocalness"]
    assert scores["instrumental"]["acoustic_instrumentalness"] > scores["vocal"]["acoustic_instrumentalness"]
    assert scores["vocal"]["acoustic_energy"] == pytest.approx(0.5)


def test_percentile_scores_uses_rank_not_minmax_distance():
    scores = percentile_scores({"low": 0.0, "middle": 1.0, "outlier": 1000.0})

    assert scores == {"low": 0.0, "middle": 0.5, "outlier": 1.0}
