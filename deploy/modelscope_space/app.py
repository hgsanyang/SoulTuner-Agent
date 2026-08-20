"""SoulTuner Agent public Creation Space.

The default mode is fully local and deterministic so the Space remains reviewable on CPU.
When an OpenAI-compatible Planner endpoint is configured, only the Planner provider changes;
retrieval, policy guard, fusion, memory and UI keep the same contract.
"""

from __future__ import annotations

import html
import json
import os
import time
from collections import Counter
from typing import Any
from urllib.parse import urlparse

import gradio as gr

from hardware import runtime_markdown
from open_audio_bootstrap import materialize_open_audio
from open_audio_bootstrap import startup_markdown as open_audio_startup_markdown
from planner_runtime import default_profile, plan_request, profile_choices
from retrieval_demo import retrieve
from retrieval_demo import audio_root
from space_bootstrap import (
    launch_local_planner_if_requested,
    live_startup_markdown,
    startup_markdown,
)


TITLE = "SoulTuner 智能音乐推荐 Agent"
PLANNER_STARTUP = launch_local_planner_if_requested()
OPEN_AUDIO_STARTUP = materialize_open_audio()
EXAMPLES = [
    "我今天心情有点差，想听温暖治愈、但不要太吵的歌",
    "想要低音更重、鼓点清晰，适合夜跑的音乐",
    "给我一些 90 年代英文摇滚，整体不要太沉重",
    "刚才那种氛围很好，再来一组更安静、更有空间感的",
    "周末小聚想听轻松明亮的中文流行",
]


CSS = """
:root { --st-green: #12a66a; --st-dark: #071a13; --st-soft: #eaf8f1; }
.gradio-container { max-width: 1220px !important; margin: auto !important; }
.st-hero { padding: 30px 34px; border-radius: 24px; color: white;
  background: radial-gradient(circle at 85% 10%, #2ac784 0, transparent 32%),
              linear-gradient(135deg, #071a13 0%, #0c402d 58%, #0f7650 100%);
  box-shadow: 0 18px 60px rgba(6, 54, 37, .22); margin-bottom: 18px; }
.st-hero h1 { font-size: 40px; margin: 0 0 8px; letter-spacing: -.5px; }
.st-hero p { max-width: 820px; opacity: .9; font-size: 16px; line-height: 1.8; }
.st-chip { display: inline-block; padding: 6px 12px; border: 1px solid rgba(255,255,255,.3);
  border-radius: 999px; margin: 8px 8px 0 0; font-size: 13px; }
.st-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }
.st-card { padding: 18px; border-radius: 18px; border: 1px solid #d9eee4;
  background: linear-gradient(145deg, #ffffff, #f4fbf7); color: #102a20 !important;
  box-shadow: 0 8px 30px rgba(7,50,35,.07); }
.st-rank { width: 30px; height: 30px; display: inline-grid; place-items: center; border-radius: 10px;
  background: #0d8d5d; color: white !important; font-weight: 700; margin-right: 9px; }
.st-card h3 { display: inline; color: #102a20 !important; font-size: 17px; font-weight: 750; }
.st-meta { color: #405f52 !important; font-size: 13px; margin: 9px 0; }
.st-tag { display: inline-block; padding: 3px 8px; margin: 2px; border-radius: 999px;
  background: #e2f7ed; color: #096743 !important; font-size: 12px; }
.st-reason { color: #1d342a !important; line-height: 1.65; font-size: 14px; min-height: 45px; }
.st-scores { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 11px;
  color: #315647 !important; font-size: 12px; }
.st-score { padding: 4px 8px; border-radius: 8px; background: #f8fcfa;
  border: 1px solid #bcd9cc; color: #315647 !important; font-weight: 600; }
.st-audio { margin-top: 12px; color: #536f62 !important; font-size: 12px; }
.st-empty { padding: 48px 24px; text-align: center; border-radius: 18px; color: #60756b;
  border: 1px dashed #b7d9c8; background: #f7fcf9; }
.st-flow { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin: 16px 0; }
.st-flow div { padding: 16px 12px; border-radius: 14px; background: #edf9f3; border: 1px solid #d8eee3;
  text-align: center; font-weight: 600; color: #125d42; }
@media (max-width: 760px) { .st-grid { grid-template-columns: 1fr; } .st-flow { grid-template-columns: 1fr 1fr; }
  .st-hero h1 { font-size: 31px; } }
"""


def _blank_memory() -> dict[str, Any]:
    return {"events": [], "positive_tags": {}, "negative_tags": {}}


def _preference_tags(memory: dict[str, Any] | None) -> set[str]:
    data = memory or _blank_memory()
    positives = Counter(data.get("positive_tags", {}))
    negatives = Counter(data.get("negative_tags", {}))
    return {
        tag
        for tag, score in positives.most_common(8)
        if score > negatives.get(tag, 0)
    }


def memory_markdown(memory: dict[str, Any] | None) -> str:
    data = memory or _blank_memory()
    events = data.get("events", [])
    likes = sum(event["action"] == "喜欢" for event in events)
    skips = sum(event["action"] == "跳过" for event in events)
    dislikes = sum(event["action"] == "不喜欢" for event in events)
    tags = list(_preference_tags(data))[:6]
    tag_text = "、".join(tags) if tags else "尚未形成偏好标签"
    return (
        "### 当前会话记忆\n"
        f"喜欢 **{likes}** · 跳过 **{skips}** · 不喜欢 **{dislikes}**  \n"
        f"偏好摘要：{tag_text}  \n"
        "仅保存在当前浏览器会话，刷新页面后清空。"
    )


def _safe_web_href(value: Any) -> str:
    text = str(value or "").strip()
    parsed = urlparse(text)
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        return html.escape(text, quote=True)
    return ""


def _render_results(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return '<div class="st-empty">输入一句话，SoulTuner 会生成检索计划并返回推荐。</div>'
    cards: list[str] = []
    for index, row in enumerate(rows, start=1):
        tags = "".join(
            f'<span class="st-tag">{html.escape(str(tag))}</span>' for tag in row["tags"][:5]
        )
        audio_text = "可播放公开试听" if row["audio_available"] else "当前曲目暂无可播放的授权音频"
        attribution = html.escape(str(row.get("attribution") or ""))
        licence = html.escape(str(row.get("license") or "逐曲许可证"))
        licence_url = _safe_web_href(row.get("license_url"))
        source_url = _safe_web_href(row.get("source_url"))
        licence_html = f'<a href="{licence_url}" target="_blank" rel="noopener">{licence}</a>' if licence_url else licence
        source_html = f'<a href="{source_url}" target="_blank" rel="noopener">上游曲目</a>' if source_url else ""
        provenance = " · ".join(part for part in (attribution, licence_html, source_html) if part)
        cards.append(
            '<article class="st-card">'
            f'<span class="st-rank">{index}</span><h3>{html.escape(row["title"])}</h3>'
            f'<div class="st-meta">{html.escape(row["artist"])} · {html.escape(row["language"])} · '
            f'{row["decade"]}s · {html.escape(row["song_id"])}</div>'
            f'<div>{tags}</div><p class="st-reason">{html.escape(row["reason"])}</p>'
            '<div class="st-scores">'
            f'<span class="st-score">融合 {row["final_score"]:.3f}</span>'
            f'<span class="st-score">Graph {row["graph_score"]:.3f}</span>'
            f'<span class="st-score">Dense {row["dense_score"]:.3f}</span>'
            f'<span class="st-score">偏好 {row["preference_score"]:.3f}</span>'
            f'</div><div class="st-audio">♫ {audio_text}</div>'
            f'<div class="st-meta">{provenance}</div></article>'
        )
    return f'<section class="st-grid">{"".join(cards)}</section>'


def _route_markdown(route: dict[str, Any], plan: dict[str, Any], status: str, elapsed: float) -> str:
    policy = plan["lane_policy"]
    reason = plan["evidence"]["brief_reason"]
    return (
        f"**检索策略：{route['profile']}** · Graph `{policy['graph']}` · "
        f"Dense `{policy['dense']}` · Web `{policy['web']}`  \n"
        f"{reason}  \n"
        f"运行状态：{status} · 端到端 {elapsed:.2f} 秒"
    )


def recommend(
    query: str,
    profile: str,
    top_k: int,
    memory: dict[str, Any] | None,
) -> tuple[str, str, list[list[Any]], dict[str, Any], dict[str, Any], Any, list, str, str | None]:
    clean_query = (query or "").strip()
    if not clean_query:
        empty = _render_results([])
        return (
            "请输入一句音乐需求。",
            empty,
            [],
            {},
            {},
            gr.Dropdown(choices=[], value=None),
            [],
            memory_markdown(memory),
            None,
        )
    started = time.perf_counter()
    plan, route, status = plan_request(profile, clean_query)
    rows = retrieve(
        clean_query,
        plan,
        route,
        top_k=int(top_k),
        preference_tags=_preference_tags(memory),
    )
    elapsed = time.perf_counter() - started
    table = [
        [
            row["title"],
            row["artist"],
            row["graph_score"],
            row["dense_score"],
            row["preference_score"],
            row["final_score"],
        ]
        for row in rows
    ]
    choices = [(f"{row['title']} — {row['artist']}", row["song_id"]) for row in rows]
    return (
        _route_markdown(route, plan, status, elapsed),
        _render_results(rows),
        table,
        plan,
        route,
        gr.Dropdown(choices=choices, value=choices[0][1] if choices else None),
        rows,
        memory_markdown(memory),
        rows[0].get("audio_source") if rows else None,
    )


def select_audio(song_id: str | None, rows: list[dict[str, Any]] | None) -> str | None:
    selected = next((row for row in (rows or []) if row.get("song_id") == song_id), None)
    return selected.get("audio_source") if selected else None


def record_feedback(
    song_id: str | None,
    action: str | None,
    rows: list[dict[str, Any]] | None,
    memory: dict[str, Any] | None,
) -> tuple[str, dict[str, Any], str]:
    if not song_id or action not in {"喜欢", "跳过", "不喜欢"}:
        data = memory or _blank_memory()
        return "请先选择歌曲和反馈类型。", data, memory_markdown(data)
    selected = next((row for row in (rows or []) if row["song_id"] == song_id), None)
    if selected is None:
        data = memory or _blank_memory()
        return "当前推荐中未找到这首歌，请重新检索。", data, memory_markdown(data)

    data = json.loads(json.dumps(memory or _blank_memory(), ensure_ascii=False))
    data["events"].append(
        {"song_id": song_id, "title": selected["title"], "action": action}
    )
    key = "positive_tags" if action == "喜欢" else "negative_tags"
    weight = 2 if action in {"喜欢", "不喜欢"} else 1
    for tag in selected["tags"]:
        data[key][tag] = int(data[key].get(tag, 0)) + weight
    return (
        f"已记录：{action}《{selected['title']}》。下一次推荐会使用当前会话偏好。",
        data,
        memory_markdown(data),
    )


def reset_memory() -> tuple[dict[str, Any], str, str]:
    data = _blank_memory()
    return data, "当前会话记忆已清空。", memory_markdown(data)


def build_app() -> gr.Blocks:
    with gr.Blocks(title=TITLE, css=CSS, theme=gr.themes.Soft(primary_hue="emerald")) as demo:
        memory_state = gr.State(_blank_memory())
        result_state = gr.State([])

        gr.HTML(
            """
            <section class="st-hero">
              <h1>SoulTuner Agent</h1>
              <p>用一句普通的话描述你想听什么。系统会规划 Graph / Dense / Web 的职责，
              在受控目录中融合召回结果，并通过会话反馈逐步理解你的偏好。</p>
              <span class="st-chip">35B Music Planner</span>
              <span class="st-chip">Graph + Dense</span>
              <span class="st-chip">Memory & Feedback</span>
              <span class="st-chip">AMD ROCm Ready</span>
            </section>
            """
        )
        gr.Markdown(runtime_markdown())
        planner_status = gr.Markdown(startup_markdown(PLANNER_STARTUP))
        gr.Markdown(open_audio_startup_markdown(OPEN_AUDIO_STARTUP))
        planner_status_timer = gr.Timer(value=5, active=bool(PLANNER_STARTUP["requested"]))
        planner_status_timer.tick(
            live_startup_markdown,
            outputs=planner_status,
            api_name=False,
            show_progress="hidden",
        )

        with gr.Tabs():
            with gr.Tab("发现音乐"):
                with gr.Row():
                    with gr.Column(scale=7):
                        query = gr.Textbox(
                            label="现在想听什么？",
                            placeholder="例如：我今天心情有点差，想听温暖治愈、但不要太吵的歌",
                            lines=3,
                        )
                    with gr.Column(scale=3):
                        profile = gr.Dropdown(
                            choices=profile_choices(),
                            value=default_profile(),
                            label="Planner 档位",
                        )
                        top_k = gr.Slider(4, 12, value=8, step=1, label="推荐数量")
                run_button = gr.Button("生成我的推荐", variant="primary")
                gr.Examples(EXAMPLES, inputs=query, label="试试这些需求")
                route_status = gr.Markdown("等待请求。")
                cards = gr.HTML(_render_results([]))
                audio_player = gr.Audio(
                    label="公开授权音频试听",
                    type="filepath",
                    interactive=False,
                )

                with gr.Row():
                    selected_song = gr.Dropdown(label="选择一首歌反馈", choices=[])
                    feedback_action = gr.Radio(
                        ["喜欢", "跳过", "不喜欢"], label="本次反馈"
                    )
                feedback_button = gr.Button("记录反馈")
                feedback_status = gr.Markdown("反馈只保存在当前会话。")

            with gr.Tab("偏好与记忆"):
                memory_view = gr.Markdown(memory_markdown(_blank_memory()))
                gr.Markdown(
                    "生产版会把经过用户授权的偏好写入可撤销长期记忆；本公开空间只展示会话级闭环。"
                )
                reset_button = gr.Button("清空当前会话记忆")
                reset_status = gr.Markdown()

            with gr.Tab("检索诊断"):
                gr.Markdown(
                    "这里展示 Planner 的公开短理由、通道角色和融合分，便于复现推荐路径；不展示隐藏思维链。"
                )
                score_table = gr.Dataframe(
                    headers=["歌曲", "艺人", "Graph", "Dense", "偏好", "融合"],
                    datatype=["str", "str", "number", "number", "number", "number"],
                    interactive=False,
                    label="候选评分",
                )
                with gr.Accordion("结构化 PlannerDecisionV5", open=False):
                    plan_json = gr.JSON(label="Planner 决策")
                with gr.Accordion("确定性路由编译结果", open=False):
                    route_json = gr.JSON(label="Route")

            with gr.Tab("完整项目与部署"):
                gr.HTML(
                    """
                    <h2>从一句话到可学习的推荐闭环</h2>
                    <div class="st-flow">
                      <div>自然语言与会话</div><div>35B Planner + Guard</div>
                      <div>Graph / Dense / Web</div><div>融合、反馈与记忆</div>
                    </div>
                    <p><b>Graph</b> 对接歌曲、艺人、流派、年代、语言和标签关系；
                    <b>Dense</b> 对接 MuQ 等音频/语义向量；<b>Memory</b> 保存经授权且可撤销的偏好；
                    <b>Data flywheel</b> 只接纳脱敏并经审核的失败样本。</p>
                    <p>当前创空间使用逐曲核验许可的开放音频实现同一数据流。完整工程通过 Neo4j、Qdrant、
                    音频向量服务、长期记忆和 Next.js 前端替换相应适配器，PlannerDecisionV5 契约保持不变。</p>
                    <h3>部署档位</h3>
                    <ul>
                      <li>CPU：确定性 Planner 与开放音频演示集，保证审核时可启动；</li>
                      <li>API：使用兼容 OpenAI 协议的 Planner，业务代码无需变化；</li>
                      <li>AMD MI308X：ROCm/HIP + SoulTuner V4.2 35B LoRA 本地 endpoint。</li>
                    </ul>
                    <p><a href="https://github.com/hgsanyang/SoulTuner-Agent" target="_blank">完整 GitHub 工程</a>
                    · <a href="https://modelscope.cn/learn/435660" target="_blank">技术文章</a>
                    · <a href="https://modelscope.cn/gallery/hgsanyang/soultuner-v4-2-35b-music-planner" target="_blank">Notebook</a></p>
                    """
                )

        run_outputs = [
            route_status,
            cards,
            score_table,
            plan_json,
            route_json,
            selected_song,
            result_state,
            memory_view,
            audio_player,
        ]
        run_button.click(
            recommend,
            inputs=[query, profile, top_k, memory_state],
            outputs=run_outputs,
            api_name="recommend",
        )
        query.submit(
            recommend,
            inputs=[query, profile, top_k, memory_state],
            outputs=run_outputs,
            api_name=False,
        )
        feedback_button.click(
            record_feedback,
            inputs=[selected_song, feedback_action, result_state, memory_state],
            outputs=[feedback_status, memory_state, memory_view],
            api_name="feedback",
        )
        selected_song.change(
            select_audio,
            inputs=[selected_song, result_state],
            outputs=audio_player,
            api_name=False,
        )
        reset_button.click(
            reset_memory,
            outputs=[memory_state, reset_status, memory_view],
            api_name="reset_memory",
        )
    return demo


demo = build_app()


if __name__ == "__main__":
    permitted_audio_root = audio_root()
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        show_error=True,
        allowed_paths=[str(permitted_audio_root)] if permitted_audio_root.is_dir() else None,
    )
