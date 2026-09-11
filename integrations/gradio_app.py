"""Optional loopback-only presentation of the main SoulTuner API."""
from __future__ import annotations

from contextlib import aclosing
from copy import deepcopy
import html
import os
from urllib.parse import urlsplit

from services.local_recommendation_client import local_api_url
from services.presentation_session import PresentationSession


def media_url(value):
    if not isinstance(value, str) or len(value) > 4096:
        return ''
    try:
        parsed = urlsplit(value)
        if parsed.scheme not in {'http', 'https'} or not parsed.hostname or parsed.username or parsed.password:
            return ''
    except ValueError:
        return ''
    return html.escape(value, quote=True)


def render_songs(songs):
    cards = []
    for song in songs:
        cover = media_url(song.get('cover_url'))
        image = f'<img src="{cover}" width="64" height="64" loading="lazy" alt="封面">' if cover else ''
        title = html.escape(str(song.get('title') or song.get('name') or '未命名曲目'))
        artist = html.escape(str(song.get('artist') or '未知艺人'))
        cards.append(f'<article style="display:flex;gap:12px;padding:12px;border-bottom:1px solid #8884">'
                     f'{image}<div><strong>{title}</strong><div>{artist}</div></div></article>')
    return ''.join(cards) or '<p>本轮尚无歌曲。</p>'


def render_player(songs, index):
    if not songs:
        return '<p>暂无可播放曲目。</p>', 0
    index = int(index or 0) % len(songs)
    song = songs[index]
    source = media_url(song.get('audio_url') or song.get('preview_url'))
    title = html.escape(str(song.get('title') or song.get('name') or '未命名曲目'))
    player = f'<audio controls preload="none" src="{source}"></audio>' if source else '<p>本曲目未提供可直接播放的 HTTP 音频地址。</p>'
    return f'<strong>{title}</strong><br>{player}', index


def build_app(base_url='http://127.0.0.1:8000'):
    import gradio as gr

    local_api_url(base_url)
    with gr.Blocks(title='SoulTuner · 主 API 展示端') as app:
        gr.Markdown('# SoulTuner\n同一个推荐管道，一段连续对话。仅供本地单用户使用，联网补充关闭。')
        session = gr.State(value=lambda: PresentationSession())
        rows = gr.State([])
        index = gr.State(0)
        with gr.Row():
            with gr.Column():
                chat = gr.Chatbot(type='messages', sanitize_html=True, allow_tags=False)
                query = gr.Textbox(label='现在想听什么，或继续聊聊', max_lines=4)
                with gr.Row():
                    send = gr.Button('发送', variant='primary')
                    reset = gr.Button('新建会话')
                status = gr.Markdown()
            with gr.Column():
                cards = gr.HTML(padding=True)
                player = gr.HTML(padding=True)
                with gr.Row():
                    previous = gr.Button('上一首')
                    following = gr.Button('下一首')

        async def submit(message, state):
            last_songs = deepcopy(state.songs)
            try:
                async with aclosing(state.turn(message, base_url=base_url)) as events:
                    async for result in events:
                        # Do not reload the player for each token of explanation.
                        changed = result['songs'][:1] != last_songs[:1]
                        last_songs = deepcopy(result['songs'])
                        yield (result['history'], render_songs(result['songs']), result['songs'],
                               result['status'], gr.update(interactive=False), gr.update(interactive=False),
                               render_player(result['songs'], 0)[0] if changed else gr.skip(),
                               0 if changed else gr.skip())
            except ValueError as exc:
                yield (gr.skip(), gr.skip(), gr.skip(), str(exc), gr.skip(), gr.skip(), gr.skip(), gr.skip())
            yield (gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.update(interactive=True),
                   gr.update(interactive=True), gr.skip(), gr.skip())

        outputs = [chat, cards, rows, status, send, reset, player, index]
        send.click(submit, [query, session], outputs, concurrency_limit=1)
        previous.click(lambda songs, idx: render_player(songs, idx - 1), [rows, index], [player, index])
        following.click(lambda songs, idx: render_player(songs, idx + 1), [rows, index], [player, index])
        reset.click(lambda: (PresentationSession(), [], [], '', '', '', 0),
                    outputs=[session, rows, chat, cards, player, status, index])
    return app


if __name__ == '__main__':
    build_app(os.getenv('SOULTUNER_LOCAL_API_URL', 'http://127.0.0.1:8000')).queue(max_size=8).launch(
        server_name='127.0.0.1', share=False, inbrowser=False)
