"""API-native web discovery only; no standalone search engine fallback."""
from __future__ import annotations

import json
import os

import httpx

from services.recommendation_execution import web_search_allowed


def sourced_response_text(data: dict) -> str:
    output = data.get('output') or []
    searched = any(item.get('type') == 'web_search_call' and item.get('status') == 'completed'
                   for item in output if isinstance(item, dict))
    texts, urls = [], []
    for item in output:
        if not isinstance(item, dict):
            continue
        if item.get('type') == 'web_search_call' and item.get('status') == 'completed':
            for source in (item.get('action') or {}).get('sources') or []:
                if isinstance(source, dict):
                    url = str(source.get('url') or '')
                    if url.startswith(('https://', 'http://')):
                        urls.append(url)
        for content in item.get('content') or []:
            if not isinstance(content, dict):
                continue
            if content.get('type') == 'output_text':
                texts.append(str(content.get('text') or ''))
            for annotation in content.get('annotations') or []:
                if isinstance(annotation, dict) and annotation.get('type') == 'url_citation':
                    url = str(annotation.get('url') or '')
                    if url.startswith(('https://', 'http://')):
                        urls.append(url)
    # Do not label model recollections as verified web results.
    if not searched or not texts or not urls:
        raise RuntimeError('Provider did not supply completed web-search evidence and citations')
    return '\n'.join(texts + ['Sources:', *dict.fromkeys(urls)])[:20000]


async def native_web_search(query: str, *, transport=None) -> str:
    if not web_search_allowed():
        raise PermissionError('Web search is disabled for this request')
    if not isinstance(query, str) or not query.strip() or len(query) > 8000:
        raise ValueError('Invalid search query')
    from config.settings import settings
    key = str(settings.dashscope_api_key or os.getenv('DASHSCOPE_API_KEY') or '')
    if not key:
        raise RuntimeError('API-native search is not configured')
    model = str(settings.llm_default_model or '')
    if not model.startswith('qwen3.'):
        raise RuntimeError('Configured model is not supported by this native search adapter')
    async with httpx.AsyncClient(timeout=60, trust_env=False, follow_redirects=False, transport=transport) as client:
        try:
            response = await client.post('https://dashscope.aliyuncs.com/compatible-mode/v1/responses',
                headers={'Authorization': 'Bearer ' + key},
                json={'model': model, 'input': '实际联网搜索音乐资料，列出可核实的歌名、艺人和网页引用：\n' + query,
                      'tools': [{'type': 'web_search'}], 'enable_thinking': True, 'max_output_tokens': 1500})
            response.raise_for_status()
            return sourced_response_text(response.json())
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            raise RuntimeError('API-native search request failed; no legacy fallback attempted') from exc
