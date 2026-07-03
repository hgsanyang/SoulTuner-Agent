import json
import math

from retrieval.alignment_calibration import apply_alignment_calibration, build_bias_calibration
from tools.semantic_search import build_dense_query_variants, _mean_vectors, _should_use_dense_query_variants


def test_dense_query_variants_are_deterministic():
    variants = build_dense_query_variants("ethereal female vocals")

    assert len(variants) == 3
    assert variants[0] == "ethereal female vocals"
    assert "instrumentation" in variants[1]
    assert "mood trajectory" in variants[2]


def test_dense_query_variants_auto_targets_scene_not_precision():
    assert _should_use_dense_query_variants("需要安静温柔的雨天歌")
    assert not _should_use_dense_query_variants("歌手周杰伦")


def test_mean_vectors_normalizes_average():
    merged = _mean_vectors([[1.0, 0.0], [0.0, 1.0]])

    assert len(merged) == 2
    assert math.isclose(sum(x * x for x in merged), 1.0, rel_tol=1e-6)


def test_alignment_calibration_noops_without_file(monkeypatch):
    monkeypatch.delenv("MUSIC_ALIGNMENT_CALIBRATION_PATH", raising=False)

    assert apply_alignment_calibration([1.0, 2.0], "muq") == [1.0, 2.0]


def test_alignment_calibration_applies_backend_bias(tmp_path, monkeypatch):
    path = tmp_path / "calibration.json"
    path.write_text(
        json.dumps({"muq": {"scale": 1.0, "bias": [1.0, 0.0], "normalize": False}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("MUSIC_ALIGNMENT_CALIBRATION_PATH", str(path))

    assert apply_alignment_calibration([1.0, 2.0], "muq") == [2.0, 2.0]


def test_build_bias_calibration_averages_text_to_audio_delta():
    spec = build_bias_calibration([
        ([0.0, 0.0], [1.0, 0.0]),
        ([0.0, 0.0], [0.0, 1.0]),
    ], normalize=False)

    assert spec["num_pairs"] == 2
    assert spec["bias"] == [0.5, 0.5]
    assert spec["scale"] == 1.0
    assert spec["normalize"] is False
