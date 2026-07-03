from data.pipeline.backfill_clamp3_embeddings import candidate_audio_roots, resolve_audio_path


def test_clamp3_backfill_resolves_static_audio_url(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    audio = audio_dir / "demo.mp3"
    audio.write_bytes(b"fake")

    assert resolve_audio_path("/static/audio/demo.mp3", [audio_dir]) == audio


def test_clamp3_backfill_audio_roots_deduplicate(monkeypatch, tmp_path):
    audio_dir = tmp_path / "audio"
    data_root = tmp_path / "data"
    monkeypatch.setenv("MUSIC_AUDIO_DATA_DIR", str(audio_dir))
    monkeypatch.setenv("MUSIC_DATA_PATH", str(data_root))

    roots = candidate_audio_roots(str(audio_dir))

    assert roots[0] == audio_dir
    assert roots.count(audio_dir) == 1
    assert data_root / "processed_audio" / "audio" in roots
