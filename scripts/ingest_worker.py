"""Process queued music enrichment jobs outside the online API process."""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.logging_config import get_logger
from services.ingest_queue import claim_next_job, complete_job, fail_job
from tools.acquire_music import _background_flywheel

logger = get_logger(__name__)


async def process_one() -> bool:
    claimed = claim_next_job()
    if claimed is None:
        return False

    job_path, payload = claimed
    songs = payload.get("songs", [])
    job_id = payload.get("job_id", job_path.stem)
    try:
        logger.info("[IngestWorker] processing job=%s songs=%s", job_id, len(songs))
        await _background_flywheel(songs)
        complete_job(job_path)
        logger.info("[IngestWorker] completed job=%s", job_id)
    except Exception as exc:
        fail_job(job_path, str(exc))
        logger.exception("[IngestWorker] failed job=%s", job_id)
    return True


async def run(watch: bool, interval: float) -> None:
    while True:
        processed = await process_one()
        if not watch:
            if not processed:
                return
            continue
        if not processed:
            await asyncio.sleep(interval)


def main() -> None:
    parser = argparse.ArgumentParser(description="SoulTuner offline music ingestion worker")
    parser.add_argument("--watch", action="store_true", help="Keep polling for new jobs")
    parser.add_argument("--interval", type=float, default=3.0, help="Polling interval in seconds")
    args = parser.parse_args()
    started = time.time()
    asyncio.run(run(args.watch, args.interval))
    logger.info("[IngestWorker] stopped after %.1fs", time.time() - started)


if __name__ == "__main__":
    main()
