from pathlib import Path

from deploy.modelscope_space import retrieval_demo


def test_resolve_audio_source_keeps_local_audio_inside_configured_root(tmp_path, monkeypatch):
    audio_root = tmp_path / "audio"
    audio_root.mkdir()
    track = audio_root / "licensed-demo.mp3"
    track.write_bytes(b"ID3")
    monkeypatch.setenv("SOULTUNER_AUDIO_ROOT", str(audio_root))

    assert retrieval_demo.resolve_audio_source({"audio_relpath": track.name}) == str(track.resolve())


def test_resolve_audio_source_rejects_path_traversal(tmp_path, monkeypatch):
    audio_root = tmp_path / "audio"
    audio_root.mkdir()
    outside = tmp_path / "private.mp3"
    outside.write_bytes(b"ID3")
    monkeypatch.setenv("SOULTUNER_AUDIO_ROOT", str(audio_root))

    assert retrieval_demo.resolve_audio_source({"audio_relpath": "../private.mp3"}) is None


def test_resolve_audio_source_accepts_https_preview(monkeypatch, tmp_path):
    monkeypatch.setenv("SOULTUNER_AUDIO_ROOT", str(tmp_path))
    url = "https://media.example.test/cc-track.mp3"

    assert retrieval_demo.resolve_audio_source({"audio_url": url}) == url


def test_open_audio_schema_is_retrievable_with_visible_provenance(tmp_path, monkeypatch):
    audio_root = tmp_path / "audio"
    audio_root.mkdir()
    track = audio_root / "pan.mp3"
    track.write_bytes(b"original-archive-bytes")
    monkeypatch.setenv("SOULTUNER_AUDIO_ROOT", str(audio_root))
    row = {
        "song_id": "sdd-4883",
        "title": "Pan",
        "artist": "Tom La Meche",
        "release_date": "2005-01-01",
        "genres": ["easylistening", "lounge"],
        "moods_themes": ["meditative"],
        "captions": [{"text": "Floaty flute with hopeful synthesizer harmonies."}],
        "audio_relpath": "pan.mp3",
        "license_id": "CC-BY-NC-ND-2.5",
        "license_url": "https://creativecommons.org/licenses/by-nc-nd/2.5/",
        "attribution": "Pan by Tom La Meche",
        "source_url": "https://example.test/pan",
        "cover_url": "https://images.example.test/pan.jpg",
        "cover_attribution": "Pan cover from the source catalog",
    }
    monkeypatch.setattr(retrieval_demo, "load_catalog", lambda: (row,))
    plan = {
        "hints": {"mood": ["安静"], "scenario": ["居家"], "genre": []},
        "hard": {},
        "metadata": {},
    }
    route = {"graph_weight": 0.4, "dense_weight": 0.6}

    results = retrieval_demo.retrieve("安静但不压抑、有氛围感", plan, route, top_k=1)

    assert results[0]["title"] == "Pan"
    assert results[0]["decade"] == 2000
    assert results[0]["audio_available"] is True
    assert results[0]["dense_source"] == "catalog_descriptions"
    assert results[0]["license"] == "CC-BY-NC-ND-2.5"
    assert results[0]["attribution"] == "Pan by Tom La Meche"
    assert results[0]["cover_url"] == "https://images.example.test/pan.jpg"
    assert results[0]["cover_attribution"] == "Pan cover from the source catalog"
