import asyncio
import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp", reason="Install requirements-mcp.txt in the isolated MCP environment")

from mcp import Client
from mcp.client.stdio import StdioServerParameters

from integrations.music_mcp.server import create_server


def test_real_sdk_tools_contract_and_multi_turn_context():
    captured = []
    async def events(query, **kwargs):
        captured.append((query, kwargs))
        yield {"type": "song", "song": {"music_id": "a", "title": "A"}}
        yield {"type": "response", "text": "A warmer selection"}
        yield {"type": "complete", "success": True, "dialog_state": {"scene": "rain"}}
    async def run():
        async with Client(create_server(event_source=events)) as client:
            tools = (await client.list_tools()).tools
            assert [tool.name for tool in tools] == ["recommend_music"]
            props = tools[0].input_schema['properties']
            assert set(props) == {'query', 'conversation_ref'}
            first = await client.call_tool('recommend_music', {'query': 'rain'})
            assert not first.is_error
            data = first.structured_content
            second = await client.call_tool('recommend_music', {'query': 'dreamier', 'conversation_ref': data['conversation_ref']})
            assert not second.is_error
            assert captured[1][1]['chat_history'][0]['content'] == 'rain'
            assert captured[1][1]['dialog_state'] == {'scene': 'rain'}
            assert captured[1][1]['personal'] is False and captured[1][1]['web_search_enabled'] is False
            unknown = await client.call_tool('recommend_music', {'query': 'rain', 'conversation_ref': 'invented'})
            assert unknown.is_error and len(captured) == 2
    asyncio.run(run())


def test_real_sdk_reports_failure_not_success_text():
    async def events(query, **kwargs):
        raise ConnectionError('private backend detail')
        yield
    async def run():
        async with Client(create_server(event_source=events)) as client:
            result = await client.call_tool('recommend_music', {'query': 'rain'})
            assert result.is_error
            assert 'private backend detail' not in str(result)
    asyncio.run(run())


def test_real_stdio_handshake_without_starting_backend():
    async def run():
        params = StdioServerParameters(command=sys.executable, args=['-m', 'integrations.music_mcp.server'],
                                       cwd=str(Path(__file__).resolve().parents[2]))
        async with Client(params) as client:
            tools = (await client.list_tools()).tools
            assert len(tools) == 1 and tools[0].name == 'recommend_music'
            assert tools[0].annotations.read_only_hint is False
            result = await client.call_tool('recommend_music', {'query': 'rain', 'conversation_ref': 'unknown'})
            assert result.is_error
    asyncio.run(asyncio.wait_for(run(), 15))


def test_invalid_query_never_reaches_backend():
    calls = []
    async def events(query, **kwargs):
        calls.append(query)
        yield {'type': 'complete', 'success': True}
    async def run():
        async with Client(create_server(event_source=events)) as client:
            result = await client.call_tool('recommend_music', {'query': 'x' * 8001})
            assert result.is_error
    asyncio.run(run())
    assert calls == []


def test_same_conversation_rejects_overlap_and_unconfirmed_replay():
    entered = asyncio.Event()
    release = asyncio.Event()
    calls = []

    async def events(query, **kwargs):
        calls.append(query)
        if query == 'second':
            entered.set()
            await release.wait()
            raise ConnectionError('unconfirmed')
        yield {'type': 'response', 'text': 'ready'}
        yield {'type': 'complete', 'success': True}

    async def run():
        async with Client(create_server(event_source=events)) as client:
            first = await client.call_tool('recommend_music', {'query': 'first'})
            ref = first.structured_content['conversation_ref']
            pending = asyncio.create_task(client.call_tool('recommend_music', {'query': 'second', 'conversation_ref': ref}))
            await asyncio.wait_for(entered.wait(), 5)
            overlap = await client.call_tool('recommend_music', {'query': 'overlap', 'conversation_ref': ref})
            assert overlap.is_error
            release.set()
            assert (await pending).is_error
            retry = await client.call_tool('recommend_music', {'query': 'retry', 'conversation_ref': ref})
            assert retry.is_error
            assert calls == ['first', 'second']
    asyncio.run(asyncio.wait_for(run(), 15))


@pytest.mark.parametrize('terminal', [{}, {'type': 'complete', 'success': False}])
def test_incomplete_result_is_never_returned_as_success(terminal):
    async def events(query, **kwargs):
        yield {'type': 'song', 'song': {'title': 'partial'}}
        yield terminal
    async def run():
        async with Client(create_server(event_source=events)) as client:
            result = await client.call_tool('recommend_music', {'query': 'rain'})
            assert result.is_error
    asyncio.run(run())
