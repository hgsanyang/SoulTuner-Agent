"""Bounded, server-held state for local API-backed presentation clients."""
from __future__ import annotations

import asyncio
from contextlib import aclosing
from copy import deepcopy
from dataclasses import dataclass, field
import json
import uuid

from services.local_recommendation_client import recommend_events


@dataclass
class PresentationSession:
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    history: list = field(default_factory=list)
    dialog: dict = field(default_factory=dict)
    songs: list = field(default_factory=list)
    busy: bool = False
    blocked: bool = False

    async def turn(self, query, *, base_url='http://127.0.0.1:8000', event_source=None):
        if self.busy or self.blocked:
            raise ValueError('会话正在处理或上轮结果未确认，请勿重复提交。')
        if not isinstance(query, str) or not query.strip() or len(query) > 8000:
            raise ValueError('请输入 1–8000 字符的需求。')
        self.busy = True
        text, songs, completion = '', [], None
        replaced = False

        def snapshot(status):
            return deepcopy({'history': [*self.history, {'role': 'user', 'content': query},
                                         {'role': 'assistant', 'content': text}],
                             'songs': songs if replaced else self.songs, 'status': status})

        try:
            yield snapshot('正在理解需求，当前保留上一轮歌单。')
            events = (event_source or recommend_events)(
                query, base_url=base_url, session_id=self.session_id, profile_id='local_admin',
                personal=False, web_search_enabled=False, device='local-gradio', chat_history=deepcopy(self.history),
                dialog_state=deepcopy(self.dialog))
            async with aclosing(events):
                async for event in events:
                    kind = event.get('type')
                    if kind in {'recommendations_start', 'songs_start'}:
                        replaced = True
                        songs = []
                        yield snapshot('正在更新本轮歌单。')
                    elif kind == 'song':
                        replaced = True
                        song = event.get('song')
                        if not isinstance(song, dict) or len(songs) >= 50:
                            raise ValueError('Invalid song result')
                        candidate_songs = [*songs, deepcopy(song)]
                        if len(json.dumps(candidate_songs, allow_nan=False)) > 262144:
                            raise ValueError('Oversized song result')
                        songs = candidate_songs
                        yield snapshot('歌曲已返回，可先播放；推荐解释仍在生成。')
                    elif kind in {'response', 'clarification_required'}:
                        text = str(event.get('text') or '')
                        if len(text) > 16000:
                            raise ValueError('Oversized response')
                        yield snapshot('正在回复。')
                    elif kind == 'complete':
                        if event.get('success') is not True:
                            raise ValueError('Unconfirmed completion')
                        completion = event
                    elif kind == 'error':
                        raise ValueError('Backend error')
            if completion is None:
                raise ValueError('Truncated stream')
            dialog = completion.get('dialog_state') or {}
            if not isinstance(dialog, dict) or len(json.dumps(dialog, allow_nan=False)) > 16000:
                raise ValueError('Invalid dialog state')
            history = snapshot('本轮完成。')['history']
            while len(history) > 24 or sum(len(row['content']) for row in history) > 24000:
                history = history[2:]
            self.history, self.dialog = history, deepcopy(dialog)
            if replaced:
                self.songs = deepcopy(songs)
            yield {'history': deepcopy(self.history), 'songs': deepcopy(self.songs),
                   'status': '本轮完成。', 'complete': deepcopy(completion)}
        except (asyncio.CancelledError, GeneratorExit):
            self.blocked = True
            raise
        except Exception:
            self.blocked = True
            yield snapshot('本轮未确认完成，已显示的歌曲仅为部分结果；未自动重试。请新建会话后继续。')
        finally:
            self.busy = False
