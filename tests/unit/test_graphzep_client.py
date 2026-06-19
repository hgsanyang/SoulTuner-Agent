import asyncio

import httpx

from services.graphzep_client import GraphZepClient


class _FailingHttpClient:
    def __init__(self):
        self.calls = 0

    async def request(self, method, path, **kwargs):
        self.calls += 1
        raise httpx.ConnectError("connection refused")

    async def get(self, path):
        raise httpx.ConnectError("connection refused")

    async def aclose(self):
        return None


class _OkResponse:
    status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        return {"facts": [{"fact": "用户喜欢 city pop", "valid_at": "2026-06-19"}]}


class _OkHttpClient:
    def __init__(self):
        self.calls = 0

    async def request(self, method, path, **kwargs):
        self.calls += 1
        return _OkResponse()

    async def get(self, path):
        return _OkResponse()

    async def aclose(self):
        return None


def test_graphzep_client_caches_unavailable_state():
    async def _run():
        http_client = _FailingHttpClient()
        client = GraphZepClient(
            base_url="http://graphzep.invalid",
            http_client=http_client,
            unavailable_ttl_seconds=300,
        )

        first = await client.search_facts("安静的歌")
        second = await client.search_facts("跑步听的歌")
        write_ok = await client.add_messages("hi", "hello")

        assert "暂时不可用" in first
        assert "暂时不可用" in second
        assert write_ok is False
        assert http_client.calls == 1

    asyncio.run(_run())


def test_graphzep_client_formats_successful_search_results():
    async def _run():
        http_client = _OkHttpClient()
        client = GraphZepClient(
            base_url="http://graphzep.invalid",
            http_client=http_client,
            unavailable_ttl_seconds=300,
        )

        result = await client.search_facts("city pop")

        assert "用户喜欢 city pop" in result
        assert "2026-06-19" in result
        assert http_client.calls == 1

    asyncio.run(_run())
