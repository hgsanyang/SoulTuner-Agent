import asyncio
import threading
from types import SimpleNamespace

import pytest

from services.memory_consolidator import InferredPreferenceCandidate, MemoryConsolidationReport
from services.memory_event_store import MemoryEventStore
from services.memory_gateway import MemoryGateway


@pytest.mark.parametrize('failure', [False, None, 'raise'])
def test_unconfirmed_projection_retains_ledger_and_reports_failure(tmp_path, failure):
    event_thread = threading.get_ident()
    store = MemoryEventStore(tmp_path / 'ledger.db')
    candidate = InferredPreferenceCandidate(field='add_moods', value='Warm', confidence=0.8,
                                           evidence_ids=['e'], ttl_days=45)

    async def consolidate(**kwargs):
        return MemoryConsolidationReport(user_id='alice', evidence_count=2, accepted=[candidate])

    def project(*args):
        assert threading.get_ident() != event_thread
        if failure == 'raise':
            raise RuntimeError('private endpoint')
        return failure

    gateway = MemoryGateway(primary=SimpleNamespace(remember_inferred_preference=project),
                            event_store=store, consolidator=SimpleNamespace(consolidate=consolidate))

    async def run():
        task = gateway._track_background(gateway.consolidate_user(user_id='alice', force=True))
        result = await task
        await asyncio.sleep(0)
        return result, await gateway.await_idle(timeout_seconds=1)

    result, idle = asyncio.run(run())
    assert result['success'] is False
    assert result['status'] == 'projection_pending'
    assert len(result['unprojected_record_ids']) == 1
    assert result['projected_memory_keys'] == []
    assert 'private endpoint' not in str(result)
    assert idle['failures'] == ['background_write_rejected']
    records = store.list_records(user_id='alice')
    assert len(records) == 2  # durable L2 plus audit, neither discarded on projection fault


def test_generator_failure_does_not_expose_internal_error(tmp_path):
    async def consolidate(**kwargs):
        raise RuntimeError('secret endpoint details')
    gateway = MemoryGateway(primary=SimpleNamespace(), event_store=MemoryEventStore(tmp_path / 'ledger.db'),
                            consolidator=SimpleNamespace(consolidate=consolidate))
    result = asyncio.run(gateway.consolidate_user(user_id='alice', force=True))
    assert result['success'] is False
    assert result['reason'] == 'memory_consolidation_unavailable'


def test_projection_job_survives_restart_and_partial_response(tmp_path):
    path = tmp_path / 'ledger.db'
    store = MemoryEventStore(path)
    receipts = set()
    applied = []
    fail = True

    def projector(owner, payload):
        key = (owner, payload['ledger_record_id'])
        if key not in receipts:
            receipts.add(key)
            applied.append(payload)
        if fail:
            raise ConnectionError('response lost')
        return True

    primary = SimpleNamespace(remember_inferred_preference_once=projector)
    gateway = MemoryGateway(primary=primary, event_store=store)
    candidate = InferredPreferenceCandidate(field='add_moods', value='Warm', confidence=0.8,
                                           evidence_ids=['e'], ttl_days=45)
    record = gateway._append_inferred_candidate(user_id='alice', candidate=candidate)
    assert store.pending_projection_ids(user_id='alice') == [record.record_id]
    assert not gateway.retry_pending_projections(user_id='alice')['success']
    reopened = MemoryEventStore(path)
    recovered = MemoryGateway(primary=primary, event_store=reopened)
    assert recovered.retry_pending_projections(user_id='bob')['success']
    fail = False
    assert recovered.retry_pending_projections(user_id='alice')['success']
    assert len(applied) == 1
    assert applied[0]['ledger_seq'] > 0
    assert reopened.pending_projection_ids(user_id='alice') == []
    assert len(reopened.list_records(user_id='alice')) == 1


def test_obsolete_pending_projection_is_not_replayed(tmp_path):
    store = MemoryEventStore(tmp_path / 'ledger.db')
    calls = []
    def project(owner, payload):
        calls.append(payload['ledger_record_id'])
        return True
    gateway = MemoryGateway(primary=SimpleNamespace(remember_inferred_preference_once=project), event_store=store)
    candidate = InferredPreferenceCandidate(field='add_moods', value='Warm', confidence=0.8, ttl_days=45)
    first = gateway._append_inferred_candidate(user_id='alice', candidate=candidate)
    second = gateway._append_inferred_candidate(user_id='alice', candidate=candidate)
    assert gateway.retry_pending_projections(user_id='alice')['success']
    assert calls == [second.record_id]
    assert not store.projection_pending(user_id='alice', record_id=first.record_id)


def test_pending_projection_requires_idempotent_adapter_before_deletion(tmp_path):
    store = MemoryEventStore(tmp_path / 'ledger.db')
    gateway = MemoryGateway(primary=SimpleNamespace(remember_inferred_preference=lambda *a: True), event_store=store)
    candidate = InferredPreferenceCandidate(field='add_moods', value='Warm', confidence=0.8, ttl_days=45)
    record = gateway._append_inferred_candidate(user_id='alice', candidate=candidate)
    with pytest.raises(RuntimeError, match='recovery_pending'):
        gateway.delete_memory_record(user_id='alice', record_id=record.record_id)
    assert store.is_effective_record(user_id='alice', record_id=record.record_id)
    assert store.projection_pending(user_id='alice', record_id=record.record_id)
