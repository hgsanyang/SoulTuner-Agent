from __future__ import annotations

import numpy as np

from data.pipeline.backfill_muq_embeddings import (
    MUQ_EMBEDDING_DIM,
    candidate_audio_roots,
    load_npz_embeddings,
    resolve_audio_path,
)


def test_resolve_audio_path_from_static_url(tmp_path):
    audio_dir = tmp_path / "processed_audio" / "audio"
    audio_dir.mkdir(parents=True)
    audio_file = audio_dir / "song name.mp3"
    audio_file.write_bytes(b"fake")

    resolved = resolve_audio_path("/static/audio/song name.mp3", [audio_dir])

    assert resolved == audio_file


def test_candidate_audio_roots_deduplicates(monkeypatch, tmp_path):
    audio_dir = tmp_path / "audio"
    data_root = tmp_path / "data"
    monkeypatch.setenv("MUSIC_AUDIO_DATA_DIR", str(audio_dir))
    monkeypatch.setenv("MUSIC_DATA_PATH", str(data_root))

    roots = candidate_audio_roots(str(audio_dir))

    assert roots[0] == audio_dir
    assert roots.count(audio_dir) == 1
    assert data_root / "processed_audio" / "audio" in roots


def test_load_npz_embeddings_validates_512d(tmp_path):
    npz_path = tmp_path / "muq.npz"
    np.savez(
        npz_path,
        ids=np.array(["1", "2"]),
        vecs=np.zeros((2, MUQ_EMBEDDING_DIM), dtype=np.float32),
    )

    embeddings = load_npz_embeddings(npz_path)

    assert sorted(embeddings) == ["1", "2"]
    assert len(embeddings["1"]) == MUQ_EMBEDDING_DIM


def test_load_npz_embeddings_rejects_wrong_dim(tmp_path):
    npz_path = tmp_path / "bad.npz"
    np.savez(npz_path, ids=np.array(["1"]), vecs=np.zeros((1, 128), dtype=np.float32))

    try:
        load_npz_embeddings(npz_path)
    except ValueError as exc:
        assert "Expected 512d" in str(exc)
    else:
        raise AssertionError("wrong-dimension MuQ npz should fail")
