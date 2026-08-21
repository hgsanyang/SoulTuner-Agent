from __future__ import annotations

import importlib.util
from pathlib import Path


SPACE = Path(__file__).resolve().parents[2] / "deploy" / "modelscope_space"
SPEC = importlib.util.spec_from_file_location("space_ui_render", SPACE / "ui_render.py")
assert SPEC and SPEC.loader
ui = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ui)


def _row(index: int, **overrides):
    row = {
        "song_id": f"licensed-{index}",
        "title": f"Rain Room {index}",
        "artist": "Open Artist",
        "language": "Instrumental",
        "decade": 2020,
        "tags": ["ambient", "calm", "rainy"],
        "reason": "安静、有空间感，同时保留一点明亮感。",
        "final_score": 0.91,
        "audio_available": True,
        "audio_source": f"/audio/{index}.mp3",
        "license": "CC BY 4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "source_url": "https://example.test/source",
        "attribution": "Open Artist",
    }
    row.update(overrides)
    return row


def test_compact_song_cards_render_five_rows_cover_and_play_state() -> None:
    rows = [_row(index) for index in range(1, 6)]
    rows[0]["cover_url"] = "https://images.example.test/rain-room.jpg"

    rendered = ui.render_results(rows)

    assert rendered.count('class="st-card"') == 5
    assert 'src="https://images.example.test/rain-room.jpg"' in rendered
    assert rendered.count("▶ 可试听") == 5
    assert "CC BY 4.0" in rendered
    assert "Rain Room 5" in rendered


def test_card_renderer_falls_back_without_cover_or_optional_metadata() -> None:
    rendered = ui.render_results([{"title": "Bare Track", "artist": "Someone"}])

    assert "Bare Track" in rendered
    assert "st-cover-1" in rendered
    assert "暂无音频" in rendered
    assert "与当前需求具有较高匹配度" in rendered


def test_card_renderer_rejects_script_cover_and_escapes_catalog_text() -> None:
    rendered = ui.render_results(
        [_row(1, title="<img src=x onerror=alert(1)>", cover_url="javascript:alert(1)")]
    )

    assert "javascript:alert" not in rendered
    assert "onerror=alert" in rendered  # visible escaped title, never executable markup
    assert "&lt;img src=x onerror=alert(1)&gt;" in rendered
    assert "st-cover-1" in rendered


def test_conversation_renders_natural_opening_and_public_route_summary() -> None:
    rendered = ui.render_conversation(
        query="外面下暴雨，想听安静但不压抑的音乐",
        opening="窗外的雨声正好给房间铺一层背景，我选了一组安静但仍有呼吸感的音乐。",
        plan={
            "evidence": {"brief_reason": "暴雨天居家，偏好安静、空间感和轻微希望感"},
            "lane_policy": {"graph": "off", "dense": "required", "web": "off"},
        },
        route={"profile": "dense_only"},
        status="35B 模型候选通过结构与策略守卫",
        elapsed=8.25,
        result_count=5,
    )

    assert "我为你整理了 5 首" in rendered
    assert "窗外的雨声" in rendered
    assert "需求理解：暴雨天居家" in rendered
    assert "Dense" in rendered
    assert "8.25s" in rendered


def test_initial_conversation_contains_recommendation_guide() -> None:
    rendered = ui.render_conversation()

    assert "晚上好，我是 SoulTuner" in rendered
    assert "暴雨天宅家" in rendered
    assert "专注工作" in rendered


def test_gradio_layout_keeps_stable_entry_player_feedback_and_five_track_default() -> None:
    source = (SPACE / "app.py").read_text(encoding="utf-8")

    assert "def build_app() -> gr.Blocks" in source
    assert "demo = build_app()" in source
    assert 'gr.Audio(' in source
    assert 'api_name="recommend"' in source
    assert 'api_name="feedback"' in source
    assert 'api_name="general_chat"' in source
    assert "gr.Chatbot(" in source
    assert '"交给 Planner 找音乐"' in source
    assert "gr.Slider(4, 12, value=8" in source
    assert "recommendation_opening(" in source
    assert "render_conversation(" in source
