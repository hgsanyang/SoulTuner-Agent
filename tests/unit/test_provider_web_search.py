"""Native provider search must carry actual tool evidence, never legacy fallback."""
import asyncio
import json

import httpx
import pytest

from services.provider_web_search import native_web_search, sourced_response_text
from services.recommendation_execution import (
    RecommendationExecution, RecommendationModels, execution_scope,
)


def evidence():
    return {'output': [
        {'type': 'web_search_call', 'status': 'completed',
         'action': {'sources': [{'url': 'https://example.org/song'}]}},
        {'type': 'message', 'content': [{'type': 'output_text', 'text': 'Song by Artist'}]},
    ]}


def test_native_sources_are_retained():
    assert sourced_response_text(evidence()) == 'Song by Artist\nSources:\nhttps://example.org/song'


@pytest.mark.parametrize('data', [
    {'output': []},
    {'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': 'Guess'}]}]},
    {'output': [{'type': 'web_search_call', 'status': 'completed'}]},
])
def test_unsourced_model_text_is_not_web_evidence(data):
    with pytest.raises(RuntimeError):
        sourced_response_text(data)


def test_disabled_search_never_calls_provider():
    def reject(request):
        pytest.fail('Disabled search must not access network')
    with execution_scope(RecommendationExecution(False, RecommendationModels(None, None, None, None))):
        with pytest.raises(PermissionError):
            asyncio.run(native_web_search('rainy music', transport=httpx.MockTransport(reject)))


def test_native_endpoint_and_no_legacy_retry(monkeypatch):
    from config.settings import settings
    monkeypatch.setattr(settings, 'dashscope_api_key', 'synthetic-test-key')
    monkeypatch.setattr(settings, 'llm_default_model', 'qwen3.7-plus')
    monkeypatch.setenv('MUSIC_WEB_SEARCH_ENABLED', '1')
    calls = []

    def respond(request):
        calls.append(request)
        assert str(request.url) == 'https://dashscope.aliyuncs.com/compatible-mode/v1/responses'
        assert json.loads(request.content)['tools'] == [{'type': 'web_search'}]
        return httpx.Response(200, json=evidence())

    assert 'Song by Artist' in asyncio.run(native_web_search('rainy music', transport=httpx.MockTransport(respond)))
    assert len(calls) == 1
    calls.clear()

    def fail(request):
        calls.append(request)
        return httpx.Response(503, text='unavailable')

    with pytest.raises(RuntimeError, match='no legacy fallback attempted'):
        asyncio.run(native_web_search('rainy music', transport=httpx.MockTransport(fail)))
    assert len(calls) == 1
