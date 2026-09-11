from threading import Lock
from concurrent.futures import ThreadPoolExecutor

import pytest

from services.memory_event_store import MemoryEventStore
from services.memory_gateway import MemoryGateway


class Projector:
    def __init__(self):
        self.receipts = set()
        self.applied = []
        self.fail_after_commit = False
        self.lock = Lock()

    def remember_preference_once(self, user_id, preferences, operation_id):
        with self.lock:
            key = (user_id, operation_id)
            if key not in self.receipts:
                self.applied.append((user_id, preferences))
                self.receipts.add(key)
            if self.fail_after_commit:
                raise ConnectionError('unconfirmed response')
        return True


def test_restart_after_remote_commit_recovers_without_reapplying(tmp_path):
    path = tmp_path / 'ledger.db'
    primary = Projector()
    primary.fail_after_commit = True
    store = MemoryEventStore(path)
    gateway = MemoryGateway(primary=primary, event_store=store)
    with pytest.raises(ConnectionError):
        gateway.remember_preference(user_id='alice', preferences={'add_moods': ['Warm']})
    assert store.effective_records(user_id='alice') == []
    assert gateway.retry_pending_preference_write(user_id='bob').success
    with pytest.raises(RuntimeError, match='pending'):
        gateway.remember_preference(user_id='alice', preferences={'avoid_moods': ['Warm']})
    with pytest.raises(RuntimeError, match='pending'):
        gateway.forget_preference_item(user_id='alice', field='add_moods', value='Warm')
    primary.fail_after_commit = False
    reopened = MemoryEventStore(path)
    recovered = MemoryGateway(primary=primary, event_store=reopened)
    assert recovered.retry_pending_preference_write(user_id='alice').success
    assert len(primary.applied) == 1
    assert len(reopened.effective_records(user_id='alice')) == 1
    assert reopened.pending_preference_write(user_id='alice') is None
    assert recovered.retry_pending_preference_write(user_id='alice').success
    assert len(primary.applied) == 1


def test_partial_ledger_failure_is_idempotent(tmp_path, monkeypatch):
    store = MemoryEventStore(tmp_path / 'ledger.db')
    primary = Projector()
    gateway = MemoryGateway(primary=primary, event_store=store)
    append = store.append
    calls = 0

    def fail_second(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError('disk unavailable')
        return append(**kwargs)

    monkeypatch.setattr(store, 'append', fail_second)
    with pytest.raises(OSError):
        gateway.remember_preference(user_id='alice', preferences={'add_moods': ['Warm', 'Calm']})
    assert len(store.effective_records(user_id='alice')) == 1
    monkeypatch.setattr(store, 'append', append)
    assert gateway.retry_pending_preference_write(user_id='alice').success
    assert len(primary.applied) == 1
    assert len(store.list_records(user_id='alice')) == 2


def test_concurrent_recovery_reuses_operation_and_records(tmp_path):
    store = MemoryEventStore(tmp_path / 'ledger.db')
    store.stage_preference_write(user_id='alice', preferences={'add_moods': ['Warm', 'Calm']})
    primary = Projector()
    gateway = MemoryGateway(primary=primary, event_store=store)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: gateway.retry_pending_preference_write(user_id='alice'), range(8)))
    assert all(result.success for result in results)
    assert len(primary.applied) == 1
    assert len(store.list_records(user_id='alice')) == 2


def test_completed_receipt_cannot_finish_new_operation(tmp_path):
    store = MemoryEventStore(tmp_path / 'ledger.db')
    old = store.stage_preference_write(user_id='alice', preferences={'add_moods': ['Warm']})
    store.finish_preference_write(user_id='alice', operation_id=old['operation_id'])
    new = store.stage_preference_write(user_id='alice', preferences={'avoid_moods': ['Warm']})
    store.finish_preference_write(user_id='bob', operation_id=new['operation_id'])
    store.finish_preference_write(user_id='alice', operation_id=old['operation_id'])
    assert store.pending_preference_write(user_id='alice')['operation_id'] == new['operation_id']


def test_visitor_recovery_uses_signed_owner_and_rejects_body_owner(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from types import SimpleNamespace
    from api.anonymous_sessions import AnonymousSessionMiddleware
    from api.visitor_memory import router
    from services import memory_gateway

    monkeypatch.setenv('SOULTUNER_SESSION_SECRET', 'a' * 64)
    monkeypatch.setenv('SOULTUNER_PUBLIC_ORIGIN', 'https://music.test')
    owners = []

    def retry(*, user_id):
        owners.append(user_id)
        return {"success": True}

    monkeypatch.setattr(memory_gateway, 'get_memory_gateway', lambda: SimpleNamespace(retry_pending_writes=retry))
    app = FastAPI()
    app.include_router(router)
    app.add_middleware(AnonymousSessionMiddleware, enabled=True)
    with TestClient(app, base_url='https://music.test') as client:
        assert client.post('/api/visitor/memory/retry-writes', json={}).status_code == 401
        client.get('/api/anonymous-session')
        assert client.post('/api/visitor/memory/retry-writes', json={'user_id': 'alice'}).status_code == 422
        assert client.post('/api/visitor/memory/retry-writes', json={}).json()['success']
    assert len(owners) == 1 and owners[0] != 'alice'


def test_retry_record_identity_cannot_change_owner_or_payload(tmp_path):
    from services.memory_models import MemoryLayer
    store = MemoryEventStore(tmp_path / 'ledger.db')
    kwargs = dict(user_id='alice', layer=MemoryLayer.EXPLICIT, kind='preference', source='user_explicit',
                  evidence_id='manual', payload={'field': 'add_moods', 'value': 'Warm'}, record_id='fixed')
    store.append(**kwargs)
    with pytest.raises(RuntimeError, match='identity_conflict'):
        store.append(**{**kwargs, 'user_id': 'bob'})
    with pytest.raises(RuntimeError, match='identity_conflict'):
        store.append(**{**kwargs, 'payload': {'value': 'Different'}})
