import asyncio
from types import SimpleNamespace

from services.memory_gateway import MemoryGateway


def test_background_work_is_bounded_and_cancellation_is_observed():
    gateway = MemoryGateway(primary=SimpleNamespace(), enable_graphzep_sidecar=False,
                            enable_event_ledger=False)
    async def run():
        gate = asyncio.Event()
        tasks = [gateway._track_background(gate.wait()) for _ in range(32)]
        rejected = gate.wait()
        assert gateway._track_background(rejected) is None
        assert rejected.cr_frame is None
        assert len(gateway._background_tasks) == 32
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await asyncio.sleep(0)
        assert not gateway._background_tasks
        assert "background_task_cancelled" in gateway._background_failures
        for _ in range(60):
            async def failed():
                return False
            await gateway._track_background(failed())
        await asyncio.sleep(0)
        assert len(gateway._background_failures) == 50
        assert gateway._background_failures[-1] == "background_write_rejected"
    asyncio.run(run())
