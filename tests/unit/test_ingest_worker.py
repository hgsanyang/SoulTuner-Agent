import asyncio

from scripts import ingest_worker


def test_worker_persists_success_result(tmp_path, monkeypatch):
    job_path = tmp_path / "job.json"
    payload = {"job_id": "job", "songs": [{"title": "Track"}]}
    completed = []

    monkeypatch.setattr(ingest_worker, "claim_next_job", lambda: (job_path, payload))

    async def enrich(_songs):
        return {"song_count": 1, "warnings": ["omar optional"]}

    monkeypatch.setattr(ingest_worker, "_background_flywheel", enrich)
    monkeypatch.setattr(
        ingest_worker,
        "complete_job",
        lambda path, result=None: completed.append((path, result)),
    )

    assert asyncio.run(ingest_worker.process_one()) is True
    assert completed == [(job_path, {"song_count": 1, "warnings": ["omar optional"]})]


def test_worker_marks_incomplete_enrichment_failed(tmp_path, monkeypatch):
    job_path = tmp_path / "job.json"
    payload = {"job_id": "job", "songs": [{"title": "Track"}]}
    failures = []

    monkeypatch.setattr(ingest_worker, "claim_next_job", lambda: (job_path, payload))

    async def enrich(_songs):
        raise RuntimeError("missing muq_embedding")

    monkeypatch.setattr(ingest_worker, "_background_flywheel", enrich)
    monkeypatch.setattr(
        ingest_worker,
        "fail_job",
        lambda path, error: failures.append((path, error)),
    )

    assert asyncio.run(ingest_worker.process_one()) is True
    assert failures == [(job_path, "missing muq_embedding")]
