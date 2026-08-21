from __future__ import annotations

from typing import Any

from neo4j.exceptions import ServiceUnavailable

from retrieval.neo4j_client import Neo4jClient


class _Record:
    def __init__(self, value: dict[str, Any]):
        self.value = value

    def data(self) -> dict[str, Any]:
        return self.value


class _Session:
    def __init__(self, result: list[dict[str, Any]] | None = None, error: Exception | None = None):
        self.result = result or []
        self.error = error

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def run(self, _query: str, _parameters: dict[str, Any]):
        if self.error:
            raise self.error
        return [_Record(row) for row in self.result]


class _Driver:
    def __init__(self, session: _Session):
        self._session = session
        self.closed = False

    def verify_connectivity(self) -> None:
        return None

    def session(self, *, database: str | None = None) -> _Session:
        self.database = database
        return self._session

    def close(self) -> None:
        self.closed = True


def _reset_singleton(monkeypatch) -> None:
    monkeypatch.setattr(Neo4jClient, "_instance", None)
    monkeypatch.setenv("NEO4J_RECONNECT_BACKOFF_SECONDS", "0")


def test_initial_outage_recovers_on_first_query(monkeypatch) -> None:
    import neo4j

    _reset_singleton(monkeypatch)
    healthy = _Driver(_Session([{"ok": True}]))
    calls = 0

    def create_driver(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ServiceUnavailable("neo4j is still starting")
        return healthy

    monkeypatch.setattr(neo4j.GraphDatabase, "driver", create_driver)

    client = Neo4jClient()
    assert client.driver is None
    assert client.execute_query("RETURN true AS ok") == [{"ok": True}]
    assert calls == 2


def test_runtime_disconnect_discards_driver_and_retries_once(monkeypatch) -> None:
    import neo4j

    _reset_singleton(monkeypatch)
    disconnected = _Driver(_Session(error=ServiceUnavailable("connection dropped")))
    healthy = _Driver(_Session([{"value": 42}]))
    drivers = iter((disconnected, healthy))
    monkeypatch.setattr(neo4j.GraphDatabase, "driver", lambda *_args, **_kwargs: next(drivers))

    client = Neo4jClient()
    assert client.execute_query("RETURN 42 AS value") == [{"value": 42}]
    assert disconnected.closed is True
    assert client.driver is healthy


def test_non_retryable_query_error_is_not_replayed(monkeypatch) -> None:
    import neo4j

    _reset_singleton(monkeypatch)
    driver = _Driver(_Session(error=ValueError("bad query")))
    calls = 0

    def create_driver(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return driver

    monkeypatch.setattr(neo4j.GraphDatabase, "driver", create_driver)

    client = Neo4jClient()
    assert client.execute_query("INVALID") == []
    assert calls == 1
    assert driver.closed is False


def test_explicit_database_is_used_for_aura_sessions(monkeypatch) -> None:
    import neo4j

    _reset_singleton(monkeypatch)
    monkeypatch.setenv("NEO4J_DATABASE", "55e3d095")
    driver = _Driver(_Session([{"ok": True}]))
    monkeypatch.setattr(neo4j.GraphDatabase, "driver", lambda *_args, **_kwargs: driver)

    client = Neo4jClient()
    assert client.execute_query("RETURN true AS ok") == [{"ok": True}]
    assert driver.database == "55e3d095"
