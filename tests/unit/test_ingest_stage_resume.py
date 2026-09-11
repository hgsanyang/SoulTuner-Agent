import asyncio

import pytest

from tools import acquire_music as acquire


def test_retry_reuses_successful_family_and_invalidates_changed_audio(tmp_path, monkeypatch):
    audio = tmp_path / "track.flac"
    audio.write_bytes(b"audio-one")
    song = {"song_id": "42", "title": "Track", "artist": "Artist", "file_basename": "Track",
            "ext": "flac", "audio_path": str(audio), "tagging_mode": "deferred"}
    cache, calls = {}, []
    class Client:
        def execute_query(self, query, params):
            return [{"eid": "1", "muq": 512, "omar": 1024}]
    monkeypatch.setattr("retrieval.neo4j_client.get_neo4j_client", Client)
    monkeypatch.setattr(acquire, "_ingest_embedding_families", lambda: ("muq_embedding", "omar_embedding"))
    async def extract(path, *, families):
        calls.append(families)
        if len(calls) == 1:
            return acquire.EmbeddingExtraction(vectors={"muq_embedding": [0.1] * 512}, errors={"omar_embedding": "offline"})
        return acquire.EmbeddingExtraction(vectors={name: [0.1] * (512 if name == "muq_embedding" else 1024)
                                                    for name in families})
    async def save(key, result):
        cache[key] = result
    monkeypatch.setattr(acquire, "_extract_embeddings", extract)
    with pytest.raises(acquire.EnrichmentIncompleteError):
        asyncio.run(acquire._background_flywheel([song], stage_cache=cache, save_stage=save))
    asyncio.run(acquire._background_flywheel([song], stage_cache=cache, save_stage=save))
    assert calls == [("muq_embedding", "omar_embedding"), ("omar_embedding",)]
    asyncio.run(acquire._background_flywheel([song], stage_cache=cache, save_stage=save))
    assert len(calls) == 2
    audio.write_bytes(b"audio-two")
    asyncio.run(acquire._background_flywheel([song], stage_cache=cache, save_stage=save))
    assert calls[-1] == ("muq_embedding", "omar_embedding")


def test_stage_checkpoint_is_preserved_on_retry_and_old_claim_is_fenced(tmp_path, monkeypatch):
    from services import ingest_queue as queue
    for name in ("PENDING_DIR", "PROCESSING_DIR", "DONE_DIR", "FAILED_DIR"):
        monkeypatch.setattr(queue, name, tmp_path / name)
    job = queue.enqueue_songs([{"title": "Track", "artist": "Artist", "audio_url": "/audio/a.mp3"}])
    old_path, _ = queue.claim_next_job()
    queue.checkpoint_stage(old_path, 0, "digest", {"vectors": {"muq_embedding": [1.]}})
    queue.fail_job(old_path, "offline")
    assert queue.retry_failed_job(job)
    _, payload = queue.claim_next_job()
    assert payload["song_stages"]["0"]["digest"]["vectors"]["muq_embedding"] == [1.]
    with pytest.raises(FileNotFoundError):
        queue.checkpoint_stage(old_path, 0, "digest", {})


@pytest.mark.parametrize("mode", ["success", "empty", "replace_failure"])
def test_download_publishes_only_complete_files(tmp_path, monkeypatch, mode):
    from types import SimpleNamespace
    target = tmp_path / "audio.mp3"
    target.write_bytes(b"old-complete")
    class Response:
        status = 200
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        async def read(self):
            return b"" if mode == "empty" else b"new-complete"
    session = SimpleNamespace(get=lambda *args, **kwargs: Response())
    if mode == "replace_failure":
        def fail(*args):
            raise OSError("disk failure")
        monkeypatch.setattr(acquire.os, "replace", fail)
    result = asyncio.run(acquire.OnlineMusicAcquirer._download_file(None, "https://example.invalid/audio", str(target), session))
    assert result == (mode == "success")
    assert target.read_bytes() == (b"new-complete" if mode == "success" else b"old-complete")
    assert list(tmp_path.glob("*.part")) == []
