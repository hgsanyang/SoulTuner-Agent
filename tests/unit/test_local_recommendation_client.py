import asyncio
import json

import httpx
import pytest

from services.local_recommendation_client import recommend_events, local_api_url, RecommendationClientError


@pytest.mark.parametrize('url', ['https://music.test', 'http://localhost:8000', 'http://127.0.0.1:8000/x',
                                 'http://user:secret@127.0.0.1', 'http://127.0.0.1?x=1'])
def test_remote_redirectable_or_credential_urls_rejected(url):
    with pytest.raises(ValueError):
        local_api_url(url)


def test_client_preserves_early_songs_and_context_without_replanning():
    async def run():
        received = []
        release = asyncio.Event()
        class Body(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield b'data: {"type":"song","song":{"music_id":"a"}}\n\n'
                await release.wait()
                yield b'data: {"type":"complete","success":true,"dialog_state":{"scene":"rain"}}\n\n'
        async def handler(request):
            received.append(json.loads(request.content))
            return httpx.Response(200, headers={'content-type': 'text/event-stream'}, stream=Body())
        events = recommend_events('dreamier', session_id='same-session',
                                  chat_history=[{'role': 'user', 'content': 'quiet music'}],
                                  transport=httpx.MockTransport(handler))
        first = await anext(events)
        assert first['type'] == 'song' and not release.is_set()
        release.set()
        tail = [event async for event in events]
        assert tail[-1]['dialog_state'] == {'scene': 'rain'}
        assert len(received) == 1
        assert received[0]['web_search_enabled'] is False
        assert received[0]['interaction_mode'] == 'developer'
        assert received[0]['session_id'] == 'same-session'
        assert received[0]['chat_history'][0]['content'] == 'quiet music'
        assert 'llm_provider' not in received[0]
    asyncio.run(asyncio.wait_for(run(), 3))


@pytest.mark.parametrize('body', [b'data: {"type":"error","error":"secret"}\n\n',
                                 b'data: {"type":"song"}\n\n', b'data: broken\n\n'])
def test_no_success_on_error_truncation_or_malformed_event(body):
    async def run():
        def handler(request):
            return httpx.Response(200, headers={'content-type': 'text/event-stream'}, content=body)
        with pytest.raises(RecommendationClientError) as error:
            _ = [event async for event in recommend_events('rain', session_id='s', transport=httpx.MockTransport(handler))]
        assert 'secret' not in str(error.value)
    asyncio.run(run())


def test_client_timeout_closes_connection_without_retry():
    async def run():
        closed = []
        class Body(httpx.AsyncByteStream):
            async def __aiter__(self):
                await asyncio.Event().wait()
                yield b''
            async def aclose(self):
                closed.append(True)
        def handler(request):
            return httpx.Response(200, headers={'content-type': 'text/event-stream'}, stream=Body())
        with pytest.raises(RecommendationClientError, match='timeout'):
            _ = [event async for event in recommend_events('rain', session_id='s', timeout_seconds=0.02,
                                                           transport=httpx.MockTransport(handler))]
        assert closed == [True]
    asyncio.run(run())


def test_client_does_not_follow_remote_redirect():
    requests = []
    def handler(request):
        requests.append(request)
        return httpx.Response(302, headers={'location': 'https://external.invalid'})
    async def run():
        with pytest.raises(RecommendationClientError, match='302'):
            _ = [event async for event in recommend_events('rain', session_id='s', transport=httpx.MockTransport(handler))]
    asyncio.run(run())
    assert len(requests) == 1


def test_local_device_and_read_budget_preserved():
    def handler(request):
        assert json.loads(request.content)['device'] == 'local-gradio'
        assert request.extensions['timeout']['read'] == 180
        assert request.extensions['timeout']['connect'] == 5
        return httpx.Response(200, headers={'content-type': 'text/event-stream'},
                              content=b'data: {"type":"complete","success":true}\n\n')
    async def run():
        events = [e async for e in recommend_events('rain', session_id='s', device='local-gradio',
                                                    transport=httpx.MockTransport(handler))]
        assert events[-1]['success'] is True
    asyncio.run(run())
