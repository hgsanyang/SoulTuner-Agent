import asyncio
from contextlib import aclosing

import pytest

from services.presentation_session import PresentationSession
from integrations.gradio_app import media_url, render_player, render_songs


def test_same_pipeline_context_and_early_songs():
    async def run():
        session = PresentationSession()
        captured = []
        release = asyncio.Event()
        async def source(query, **kwargs):
            captured.append((query, kwargs))
            yield {'type': 'recommendations_start'}
            yield {'type': 'song', 'song': {'music_id': query, 'title': query}}
            await release.wait()
            yield {'type': 'response', 'text': 'explanation'}
            yield {'type': 'complete', 'success': True, 'dialog_state': {'scene': 'rain'}}
        async with aclosing(session.turn('rain', event_source=source)) as events:
            await anext(events)
            await anext(events)
            first = await anext(events)
            assert first['songs'][0]['title'] == 'rain'
            assert not release.is_set()
            first['songs'][0]['title'] = 'client mutation'
            release.set()
            tail = [value async for value in events]
        assert tail[-1]['songs'][0]['title'] == 'rain'
        second = [value async for value in session.turn('梦幻一点的歌曲有没有', event_source=source)]
        assert second[-1]['songs'][0]['title'] == '梦幻一点的歌曲有没有'
        assert captured[1][1]['chat_history'][0]['content'] == 'rain'
        assert captured[1][1]['dialog_state'] == {'scene': 'rain'}
        assert captured[1][1]['session_id'] == captured[0][1]['session_id']
        assert captured[1][1]['web_search_enabled'] is False
    asyncio.run(asyncio.wait_for(run(), 3))


def test_chat_keeps_playlist_and_explicit_empty_recommendation_clears_it():
    async def run():
        session = PresentationSession(songs=[{'title': 'old'}])
        async def chat(query, **kwargs):
            yield {'type': 'response', 'text': 'hello'}
            yield {'type': 'complete', 'success': True}
        async def empty(query, **kwargs):
            yield {'type': 'recommendations_start'}
            yield {'type': 'complete', 'success': True}
        assert [v async for v in session.turn('hello', event_source=chat)][-1]['songs'] == [{'title': 'old'}]
        assert [v async for v in session.turn('different', event_source=empty)][-1]['songs'] == []
    asyncio.run(run())


def test_failure_is_partial_and_blocks_blind_retry():
    async def run():
        session = PresentationSession(songs=[{'title': 'old'}])
        async def fail(query, **kwargs):
            yield {'type': 'song', 'song': {'title': 'partial'}}
            raise RuntimeError('secret backend details')
        outputs = [v async for v in session.turn('rain', event_source=fail)]
        assert 'secret' not in str(outputs)
        assert '未确认' in outputs[-1]['status']
        assert session.history == [] and session.songs == [{'title': 'old'}]
        assert not session.busy and session.blocked
        with pytest.raises(ValueError):
            await anext(session.turn('retry', event_source=fail))
    asyncio.run(run())


def test_close_propagates_to_upstream_and_same_session_overlap_rejected():
    async def run():
        closed = []
        session = PresentationSession()
        async def source(query, **kwargs):
            try:
                yield {'type': 'song', 'song': {'title': 'A'}}
                await asyncio.Event().wait()
            finally:
                closed.append(True)
        stream = session.turn('rain', event_source=source)
        await anext(stream)
        await anext(stream)
        with pytest.raises(ValueError):
            await anext(session.turn('overlap', event_source=source))
        await stream.aclose()
        assert closed == [True] and session.blocked and not session.busy
    asyncio.run(run())


def test_media_rendering_escapes_text_and_rejects_local_or_active_urls():
    for value in ['javascript:alert(1)', 'file:///private/audio.mp3', 'C:/secret.mp3', 'http://user:pass@host/a']:
        assert media_url(value) == ''
    songs = [{'title': '<script>alert(1)</script>', 'artist': '<b>artist</b>',
              'audio_url': 'https://example.org/a.mp3', 'cover_url': 'javascript:bad'}, {'title': 'B'}]
    assert '<script>' not in render_songs(songs)
    assert '&lt;script&gt;' in render_songs(songs)
    assert 'javascript:' not in render_songs(songs)
    player, index = render_player(songs, -1)
    assert index == 1 and 'HTTP' in player
    assert '<audio controls' in render_player(songs, 2)[0]
