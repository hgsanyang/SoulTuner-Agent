"""Pure HTML renderers for the public Gradio experience.

Keep these functions free of Gradio/model imports so the public UI can be
tested without starting the 35B endpoint or materialising the audio dataset.
"""

from __future__ import annotations

import html
import re
from typing import Any
from urllib.parse import urlparse


def _text(value: Any, fallback: str = "") -> str:
    clean = str(value or "").strip()
    return clean or fallback


def _safe_web_href(value: Any) -> str:
    text = _text(value)
    parsed = urlparse(text)
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        return html.escape(text, quote=True)
    if len(text) <= 512 * 1024 and re.fullmatch(
        r"data:image/(?:png|jpeg|webp|svg\+xml);base64,[A-Za-z0-9+/=]+",
        text,
    ):
        return text
    return ""


def _score(value: Any) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "—"


def _tags(row: dict[str, Any]) -> list[str]:
    raw = row.get("tags") or []
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        return []
    return list(dict.fromkeys(_text(item) for item in raw if _text(item)))[:4]


def _cover_html(row: dict[str, Any], index: int, title: str) -> str:
    for key in ("cover_url", "cover", "image_url", "artwork_url"):
        href = _safe_web_href(row.get(key))
        if href:
            return (
                '<div class="st-cover">'
                f'<img src="{href}" alt="{html.escape(title, quote=True)} 封面" loading="lazy">'
                "</div>"
            )
    initial = html.escape(title[:1].upper() or "♫")
    return f'<div class="st-cover st-cover-{index % 5}" aria-hidden="true">{initial}</div>'


def render_results(
    rows: list[dict[str, Any]] | None,
    active_song_id: str | None = None,
) -> str:
    if not rows:
        return (
            '<div class="st-empty"><span>♫</span><b>推荐会出现在这里</b>'
            "<p>描述此刻的场景、心情或想避开的听感，SoulTuner 会为你组织一组可试听结果。</p></div>"
        )

    cards: list[str] = []
    for index, row in enumerate(rows, start=1):
        title = _text(row.get("title"), "未命名曲目")
        artist = _text(row.get("artist"), "未知艺人")
        language = _text(row.get("language"), "语言未知")
        decade = _text(row.get("decade"))
        meta = " · ".join(part for part in (artist, language, f"{decade}s" if decade and decade != "0" else "") if part)
        tag_html = "".join(f'<span class="st-tag">{html.escape(tag)}</span>' for tag in _tags(row))
        playable = bool(row.get("audio_available") or row.get("audio_source"))
        play_state = "可试听" if playable else "暂无音频"
        play_class = "is-ready" if playable else ""
        reason = html.escape(_text(row.get("reason"), "与当前需求具有较高匹配度。"))

        attribution = html.escape(_text(row.get("attribution")))
        licence = html.escape(_text(row.get("license"), "逐曲许可证"))
        licence_url = _safe_web_href(row.get("license_url"))
        source_url = _safe_web_href(row.get("source_url"))
        licence_html = (
            f'<a href="{licence_url}" target="_blank" rel="noopener">{licence}</a>' if licence_url else licence
        )
        source_html = f'<a href="{source_url}" target="_blank" rel="noopener">来源</a>' if source_url else ""
        provenance = " · ".join(part for part in (attribution, licence_html, source_html) if part)
        provenance_html = f'<div class="st-provenance">{provenance}</div>' if provenance else ""
        song_id = html.escape(_text(row.get("song_id")), quote=True)
        is_current = bool(active_song_id and _text(row.get("song_id")) == active_song_id)
        card_class = "st-card is-current" if is_current else "st-card"
        playing_label = "正在播放" if is_current and playable else play_state
        equalizer = (
            '<span class="st-equalizer" aria-label="正在播放"><i></i><i></i><i></i><i></i></span>'
            if is_current and playable
            else ""
        )

        cards.append(
            f'<article class="{card_class}" data-song-id="{song_id}">'
            f"{_cover_html(row, index, title)}"
            '<div class="st-track-main">'
            '<div class="st-track-heading">'
            f'<span class="st-rank">{index:02d}</span>'
            f"<div><h3>{html.escape(title)}</h3><p>{html.escape(meta)}</p></div>"
            "</div>"
            f'<div class="st-track-tags">{tag_html}</div>'
            f'<p class="st-reason">{reason}</p>{provenance_html}'
            "</div>"
            '<div class="st-track-side">'
            f"{equalizer}"
            f'<span class="st-match">{_score(row.get("final_score"))}<small>匹配</small></span>'
            f'<span class="st-play-state {play_class}">▶ {playing_label}</span>'
            "</div></article>"
        )
    return f'<section class="st-grid">{"".join(cards)}</section>'


def render_conversation(
    *,
    query: str = "",
    plan: dict[str, Any] | None = None,
    route: dict[str, Any] | None = None,
    status: str = "",
    opening: str = "",
    elapsed: float = 0.0,
    result_count: int = 0,
) -> str:
    clean_query = _text(query)
    if not clean_query:
        return (
            '<section class="st-conversation">'
            '<div class="st-assistant-row"><span class="st-avatar">S</span><div class="st-bubble st-assistant">'
            "<b>晚上好，我是 SoulTuner。</b>"
            "<p>不用先想歌名。告诉我此刻的场景、心情、想要的声音，或者明确说出“不想听什么”。</p>"
            '<div class="st-guide-grid"><span>☔ 暴雨天宅家</span><span>🌙 安静夜晚</span>'
            "<span>💻 专注工作</span><span>🏃 节奏运动</span></div>"
            "</div></div></section>"
        )

    current_plan = plan or {}
    current_route = route or {}
    evidence = current_plan.get("evidence") if isinstance(current_plan.get("evidence"), dict) else {}
    reason = _text(evidence.get("brief_reason"), "我已经理解这次的听歌方向。")
    natural_opening = _text(opening)
    policy = current_plan.get("lane_policy") if isinstance(current_plan.get("lane_policy"), dict) else {}
    lane_labels = [
        label
        for label, key in (("Graph", "graph"), ("Dense", "dense"), ("Web", "web"))
        if _text(policy.get(key), "off") != "off"
    ]
    lane_text = " + ".join(lane_labels) or _text(current_route.get("profile"), "安全目录")
    count_text = f"我为你整理了 {result_count} 首" if result_count else "暂时没有找到合适曲目"
    safe_status = html.escape(_text(status, "请求已完成"))
    opening_html = f"<p>{html.escape(natural_opening)}</p>" if natural_opening else ""
    return "".join(
        (
            '<section class="st-conversation">'
            '<div class="st-user-row"><div class="st-bubble st-user">'
            f"{html.escape(clean_query)}</div></div>",
            '<div class="st-assistant-row"><span class="st-avatar">S</span><div class="st-bubble st-assistant">'
            f"<b>{html.escape(count_text)}。</b>",
            opening_html,
            f'<p class="st-understanding">需求理解：{html.escape(reason)}</p>',
            '<div class="st-route-line">'
            f"<span>{html.escape(lane_text)}</span><span>{safe_status}</span><span>{elapsed:.2f}s</span>",
            "</div></div></div></section>",
        )
    )
