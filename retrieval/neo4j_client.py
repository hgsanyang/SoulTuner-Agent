import os
import logging
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)


class Neo4jClient:
    """Neo4j 图数据库连接与查询客户端"""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Neo4jClient, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        self._lock = threading.RLock()
        self.driver = None
        self._uri = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
        self._user = os.getenv("NEO4J_USER", "neo4j")
        self._password = os.getenv("NEO4J_PASSWORD", "12345678")
        self._connect()

    def _connect(self, *, force: bool = False) -> bool:
        """Create a verified driver, and leave later calls able to retry."""

        with self._lock:
            if self.driver is not None and not force:
                return True
            if force:
                self._discard_driver()
            candidate = None
            try:
                from neo4j import GraphDatabase

                candidate = GraphDatabase.driver(
                    self._uri,
                    auth=(self._user, self._password),
                    connection_timeout=float(os.getenv("NEO4J_CONNECTION_TIMEOUT_SECONDS", "5")),
                )
                candidate.verify_connectivity()
                self.driver = candidate
                logger.info("Successfully connected to Neo4j database.")
                return True
            except Exception as exc:
                if candidate is not None:
                    try:
                        candidate.close()
                    except Exception:
                        pass
                self.driver = None
                logger.warning("Neo4j connection is unavailable; a later query will retry: %s", exc)
                return False

    def _discard_driver(self) -> None:
        driver, self.driver = self.driver, None
        if driver is not None:
            try:
                driver.close()
            except Exception:
                pass

    @staticmethod
    def _is_retryable_connection_error(exc: Exception) -> bool:
        try:
            from neo4j.exceptions import ServiceUnavailable, SessionExpired, TransientError

            return isinstance(exc, (ServiceUnavailable, SessionExpired, TransientError))
        except ImportError:  # pragma: no cover - dependency gate
            return False

    def close(self):
        with self._lock:
            self._discard_driver()

    def execute_query(self, query: str, parameters: dict[str, Any] | None = None):
        """Execute Cypher and reconnect once after a transient disconnect.

        A failed initial application start is recoverable too: the first query
        after Neo4j returns will create a fresh driver instead of keeping the
        process permanently degraded.
        """
        parameters = parameters or {}
        reconnect_backoff = max(0.0, float(os.getenv("NEO4J_RECONNECT_BACKOFF_SECONDS", "0.25")))
        for attempt in range(2):
            if self.driver is None and not self._connect():
                if attempt == 0 and reconnect_backoff:
                    time.sleep(reconnect_backoff)
                    continue
                return []
            driver = self.driver
            try:
                with driver.session() as session:
                    result = session.run(query, parameters)
                    return [record.data() for record in result]
            except Exception as exc:
                if attempt == 0 and self._is_retryable_connection_error(exc):
                    logger.warning("Neo4j query lost its connection; reconnecting once: %s", exc)
                    with self._lock:
                        if self.driver is driver:
                            self._discard_driver()
                    if reconnect_backoff:
                        time.sleep(reconnect_backoff)
                    continue
                logger.error("Error executing Neo4j query: %s", exc)
                return []
        return []


def get_neo4j_client() -> Neo4jClient:
    return Neo4jClient()
