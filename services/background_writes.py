"""Bounded best-effort writes; not a durable queue for critical user data."""
import contextvars
import logging
import threading
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class BackgroundWrites:
    def __init__(self, *, workers=2, capacity=64):
        self._slots = threading.BoundedSemaphore(capacity)
        self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="soultuner-write")

    def submit(self, operation, *args, **kwargs):
        if not self._slots.acquire(blocking=False):
            logger.warning("Best-effort background write dropped: capacity reached")
            return False
        context = contextvars.copy_context()
        try:
            future = self._executor.submit(context.run, operation, *args, **kwargs)
        except BaseException:
            self._slots.release()
            raise

        def finished(result):
            try:
                result.result()
            except Exception:
                logger.exception("Best-effort background write failed")
            finally:
                self._slots.release()

        future.add_done_callback(finished)
        return True

    def close(self):
        self._executor.shutdown(wait=True)


background_writes = BackgroundWrites()
