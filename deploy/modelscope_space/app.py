"""SoulTuner Agent public Creation Space.

The default mode is fully local and deterministic so the Space remains reviewable on CPU.
When an OpenAI-compatible Planner endpoint is configured, only the Planner provider changes;
retrieval, policy guard, fusion, memory and UI keep the same contract.
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from typing import Any

import gradio as gr

from conversation_runtime import recommendation_opening
from conversation_ui import continue_general_chat, reset_general_chat
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
from ui_render import render_conversation, render_results


TITLE = "SoulTuner 智能音乐推荐 Agent"
PLANNER_STARTUP = launch_local_planner_if_requested()
OPEN_AUDIO_STARTUP = materialize_open_audio()
PUBLIC_TRACK_COUNT = int(OPEN_AUDIO_STARTUP.get("tracks") or 0)
EXAMPLES = [
    "外面下暴雨，窝在家里想听氛围感强、安静但不压抑的音乐",
    "我今天心情有点差，想听温暖治愈、但不要太吵的歌",
    "想要低音更重、鼓点清晰，适合夜跑的音乐",
    "给我一些 90 年代英文摇滚，整体不要太沉重",
    "刚才那种氛围很好，再来一组更安静、更有空间感的",
]


CSS = """
:root { --st-green: #24d184; --st-green-2: #11a76a; --st-ink: #eef6f2;
  --st-muted: #92a49d; --st-panel: rgba(16, 24, 34, .92); --st-line: rgba(255,255,255,.09); }
html, body, .gradio-container { background: #080d15 !important; color: var(--st-ink) !important; overflow-x: hidden !important; }
body { background-image: radial-gradient(circle at 12% 18%, rgba(36,209,132,.08), transparent 24%),
  radial-gradient(circle at 84% 5%, rgba(39,121,255,.07), transparent 26%) !important; }
.gradio-container { width: 100% !important; max-width: 1460px !important; box-sizing: border-box !important;
  margin: auto !important; padding: 18px 24px 80px !important; }
.contain, .panel, .block { border-color: var(--st-line) !important; }
.st-hero { position: relative; overflow: hidden; padding: 26px 30px; border-radius: 24px;
  border: 1px solid rgba(56,232,154,.2); color: white;
  background: radial-gradient(circle at 88% 8%, rgba(43,221,139,.82), transparent 30%),
    linear-gradient(118deg, #073323 0%, #0b5b3b 54%, #0d8e58 100%);
  box-shadow: 0 22px 70px rgba(0,0,0,.3); margin-bottom: 12px; }
.st-hero:after { content: ""; position: absolute; inset: 0; opacity: .22; pointer-events: none;
  background-image: radial-gradient(#fff 1px, transparent 1px); background-size: 36px 36px; }
.st-hero h1 { position: relative; z-index: 1; font-size: clamp(34px, 4vw, 52px); margin: 0 0 4px;
  letter-spacing: -1.8px; font-weight: 850; }
.st-hero p { position: relative; z-index: 1; max-width: 860px; opacity: .9; margin: 0;
  font-size: 15px; line-height: 1.7; }
.st-hero-badges { position: relative; z-index: 1; margin-top: 14px; }
.st-chip { display: inline-block; padding: 6px 11px; border: 1px solid rgba(255,255,255,.3);
  border-radius: 999px; margin: 4px 6px 0 0; font-size: 12px; background: rgba(2,17,12,.2); }
.st-system { margin: 8px 0 14px !important; padding: 10px 14px !important; border-radius: 14px !important;
  background: rgba(14,22,31,.82) !important; border: 1px solid var(--st-line) !important; }
.st-system p { margin: 2px 0 !important; color: #b9c7c1 !important; font-size: 12px !important; }
.st-tabs > .tab-nav { border-bottom: 1px solid var(--st-line) !important; }
.st-shell { width: 100% !important; max-width: 100% !important; gap: 14px !important; align-items: stretch !important; }
.st-pane { min-width: 0 !important; padding: 16px !important; border-radius: 18px !important;
  border: 1px solid var(--st-line) !important; background: var(--st-panel) !important;
  box-shadow: 0 16px 42px rgba(0,0,0,.18); }
.st-section-title { display: flex; align-items: center; justify-content: space-between; margin: 0 0 12px; }
.st-section-title h2 { margin: 0; color: #f5fbf8; font-size: 17px; }
.st-kicker { color: var(--st-green); font-size: 11px; font-weight: 800; letter-spacing: .08em; text-transform: uppercase; }
.st-conversation { min-height: 286px; padding: 6px 2px 14px; display: flex; flex-direction: column; gap: 14px; }
.st-assistant-row, .st-user-row { display: flex; gap: 10px; align-items: flex-start; }
.st-user-row { justify-content: flex-end; }
.st-avatar { width: 30px; height: 30px; border-radius: 10px; display: grid; place-items: center;
  flex: 0 0 auto; color: #04150e; background: linear-gradient(135deg,#27df8e,#1caf70); font-weight: 900; }
.st-bubble { max-width: 88%; border-radius: 16px; padding: 13px 15px; line-height: 1.65; font-size: 14px; }
.st-bubble p { margin: 5px 0 0; color: #bac8c2; }
.st-bubble .st-understanding { color: #7f938a; font-size: 11px; }
.st-assistant { background: #111c27; border: 1px solid rgba(255,255,255,.08); color: #edf7f2; }
.st-user { background: linear-gradient(135deg,#146744,#0f4f36); border: 1px solid rgba(71,238,163,.25); }
.st-guide-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 7px; margin-top: 12px; }
.st-guide-grid span { padding: 7px 9px; border-radius: 10px; color: #b8c9c1; background: rgba(255,255,255,.04); font-size: 12px; }
.st-route-line { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }
.st-route-line span { padding: 3px 7px; border-radius: 999px; color: #84e9ba; background: rgba(36,209,132,.09);
  border: 1px solid rgba(36,209,132,.16); font-size: 10px; }
.st-input textarea { background: #151f2b !important; color: #eef7f3 !important; border-radius: 14px !important; }
.st-chatbot { min-height: 292px !important; border-radius: 15px !important; background: #0c141d !important;
  border: 1px solid var(--st-line) !important; }
.st-chat-status p { color: #78d9a9 !important; font-size: 11px !important; margin: 4px 0 !important; }
.st-primary button { min-height: 44px !important; border: 0 !important; border-radius: 13px !important;
  color: #03150e !important; font-weight: 850 !important; background: linear-gradient(90deg,#19b970,#2add91) !important; }
.st-grid { display: grid; grid-template-columns: 1fr; gap: 8px; max-height: 650px; overflow-y: auto; padding-right: 4px; }
.st-card { display: grid; grid-template-columns: 62px minmax(0,1fr) 64px; gap: 11px; align-items: center;
  padding: 10px; border-radius: 15px; border: 1px solid var(--st-line); background: #0d151f; color: #eaf4ef;
  transition: border-color .18s ease, transform .18s ease, background .18s ease; }
.st-card:hover { transform: translateY(-1px); border-color: rgba(42,221,145,.35); background: #101b26; }
.st-cover { width: 62px; height: 62px; border-radius: 12px; overflow: hidden; display: grid; place-items: center;
  color: rgba(255,255,255,.9); font-size: 24px; font-weight: 900; background: linear-gradient(145deg,#215b47,#132d28); }
.st-cover img { width: 100%; height: 100%; display: block; object-fit: cover; }
.st-cover-0 { background: linear-gradient(145deg,#205b4a,#0b3028); }.st-cover-1 { background: linear-gradient(145deg,#31527f,#182844); }
.st-cover-2 { background: linear-gradient(145deg,#76533a,#312316); }.st-cover-3 { background: linear-gradient(145deg,#603d71,#2a1835); }
.st-cover-4 { background: linear-gradient(145deg,#566a35,#28341a); }
.st-track-main { min-width: 0; }.st-track-heading { display: flex; gap: 8px; align-items: flex-start; }
.st-rank { color: #64746e; font-size: 10px; font-weight: 800; padding-top: 3px; }
.st-track-heading h3 { color: #f3f9f6; font-size: 14px; margin: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.st-track-heading p { color: #879a92; font-size: 11px; margin: 2px 0 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.st-track-tags { margin: 6px 0 4px; white-space: nowrap; overflow: hidden; }.st-tag { display: inline-block; padding: 2px 6px;
  margin-right: 4px; border-radius: 999px; background: rgba(36,209,132,.08); color: #7bdcac; font-size: 9px; }
.st-reason { color: #aab9b3; line-height: 1.45; font-size: 10px; margin: 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.st-provenance { margin-top: 3px; color: #62736c; font-size: 9px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.st-provenance a { color: #6eaf91 !important; text-decoration: none; }.st-track-side { display: flex; flex-direction: column; gap: 7px; align-items: flex-end; }
.st-match { color: #e7f7ef; font-size: 15px; font-weight: 800; }.st-match small { display: block; color: #61736b; font-size: 8px; text-align: right; }
.st-play-state { padding: 3px 6px; border-radius: 999px; color: #778880; background: rgba(255,255,255,.04); font-size: 8px; white-space: nowrap; }
.st-play-state.is-ready { color: #65dea3; background: rgba(36,209,132,.1); }
.st-empty { min-height: 340px; padding: 52px 26px; text-align: center; border-radius: 16px; color: #81928b;
  border: 1px dashed rgba(255,255,255,.12); background: #0c141d; display: grid; place-content: center; }
.st-empty span { color: var(--st-green); font-size: 30px; }.st-empty b { color: #dce9e3; margin-top: 8px; }.st-empty p { max-width: 360px; font-size: 12px; }
.st-player, .st-feedback { margin-top: 10px !important; padding: 10px !important; border-radius: 14px !important;
  max-width: 100% !important; overflow: hidden !important; box-sizing: border-box !important;
  background: #0c141d !important; border: 1px solid var(--st-line) !important; }
.st-flow { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin: 16px 0; }
.st-flow div { padding: 14px 10px; border-radius: 12px; background: #111c27; border: 1px solid var(--st-line);
  text-align: center; font-weight: 650; color: #77d7aa; }
@media (max-width: 980px) { .st-shell { flex-direction: column !important; } .st-grid { max-height: none; }
  .st-conversation { min-height: auto; }.st-empty { min-height: 240px; } }
@media (max-width: 620px) { .gradio-container { padding: 10px 10px 70px !important; }.st-hero { padding: 22px 20px; }
  .st-card { grid-template-columns: 52px minmax(0,1fr); }.st-cover { width: 52px; height: 52px; }.st-track-side { display: none; }
  .st-guide-grid, .st-flow { grid-template-columns: 1fr 1fr; } }
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


_render_results = render_results


def _route_markdown(
    query: str,
    route: dict[str, Any],
    plan: dict[str, Any],
    status: str,
    elapsed: float,
    opening: str = "",
    conversation_status: str = "",
    result_count: int = 0,
) -> str:
    display_status = status
    if conversation_status:
        display_status = f"{status} · 自然语言 {conversation_status}"
    return render_conversation(
        query=query,
        plan=plan,
        route=route,
        status=display_status,
        opening=opening,
        elapsed=elapsed,
        result_count=result_count,
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
            render_conversation(),
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
    opening, conversation_status = recommendation_opening(
        clean_query,
        plan,
        rows,
        memory,
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
        _route_markdown(
            clean_query,
            route,
            plan,
            status,
            elapsed,
            opening,
            conversation_status,
            len(rows),
        ),
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
              <p>把场景、情绪和声音偏好说给我听。35B 会把自然对话与可执行检索计划衔接起来，
              再从开放授权目录中组织可试听推荐，并用反馈记住你这一轮的取舍。</p>
              <div class="st-hero-badges">
                <span class="st-chip">35B Planner + Conversation</span>
                <span class="st-chip">Graph + Dense</span>
                <span class="st-chip">Open Audio</span>
                <span class="st-chip">AMD ROCm</span>
              </div>
            </section>
            """
        )
        with gr.Row():
            gr.Markdown(runtime_markdown(), elem_classes=["st-system"])
            planner_status = gr.Markdown(
                startup_markdown(PLANNER_STARTUP), elem_classes=["st-system"]
            )
            gr.Markdown(
                open_audio_startup_markdown(OPEN_AUDIO_STARTUP), elem_classes=["st-system"]
            )
        planner_status_timer = gr.Timer(value=5, active=bool(PLANNER_STARTUP["requested"]))
        planner_status_timer.tick(
            live_startup_markdown,
            outputs=planner_status,
            api_name=False,
            show_progress="hidden",
        )

        with gr.Tabs(elem_classes=["st-tabs"]):
            with gr.Tab("发现音乐"):
                with gr.Row(elem_classes=["st-shell"]):
                    with gr.Column(scale=6, elem_classes=["st-pane"]):
                        gr.HTML(
                            '<div class="st-section-title"><div><span class="st-kicker">Conversation</span>'
                            '<h2>对话记录</h2></div><span class="st-kicker">会话级记忆</span></div>'
                        )
                        with gr.Tabs():
                            with gr.Tab("找音乐"):
                                route_status = gr.HTML(render_conversation())
                                query = gr.Textbox(
                                    label="现在想听什么？",
                                    placeholder="例如：外面下暴雨，想听安静但不压抑、有空间感的音乐",
                                    lines=3,
                                    elem_classes=["st-input"],
                                )
                                with gr.Accordion("推荐设置", open=False):
                                    profile = gr.Dropdown(
                                        choices=profile_choices(),
                                        value=default_profile(),
                                        label="Planner 档位",
                                    )
                                    top_k = gr.Slider(4, 12, value=8, step=1, label="推荐数量")
                                run_button = gr.Button(
                                    "交给 Planner 找音乐", variant="primary", elem_classes=["st-primary"]
                                )
                                gr.Examples(EXAMPLES, inputs=query, label="你也可以这样说")

                            with gr.Tab("聊音乐"):
                                chat_history = gr.Chatbot(
                                    value=[],
                                    type="messages",
                                    label="35B 基座自然对话",
                                    height=292,
                                    allow_tags=False,
                                    elem_classes=["st-chatbot"],
                                )
                                chat_status = gr.Markdown(
                                    "自然对话：等待消息。", elem_classes=["st-chat-status"]
                                )
                                chat_message = gr.Textbox(
                                    label="继续聊聊",
                                    placeholder="比如：为什么下雨天会更想听有空间感的音乐？",
                                    lines=2,
                                    elem_classes=["st-input"],
                                )
                                with gr.Row():
                                    chat_send = gr.Button("发送", variant="primary")
                                    chat_clear = gr.Button("清空对话")

                    with gr.Column(scale=7, elem_classes=["st-pane"]):
                        gr.HTML(
                            '<div class="st-section-title"><div><span class="st-kicker">Recommendations</span>'
                            '<h2>为你挑选</h2></div>'
                            f'<span class="st-kicker">{PUBLIC_TRACK_COUNT or "开放"} 首曲库</span></div>'
                        )
                        cards = gr.HTML(_render_results([]))
                        with gr.Group(elem_classes=["st-player"]):
                            audio_player = gr.Audio(
                                label="公开授权音频试听",
                                type="filepath",
                                interactive=False,
                            )
                            selected_song = gr.Dropdown(label="当前曲目", choices=[])
                        with gr.Group(elem_classes=["st-feedback"]):
                            with gr.Row():
                                feedback_action = gr.Radio(
                                    ["喜欢", "跳过", "不喜欢"], label="这首歌怎么样？"
                                )
                                feedback_button = gr.Button("记录反馈", variant="secondary")
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
        chat_send.click(
            continue_general_chat,
            inputs=[chat_message, chat_history, memory_state],
            outputs=[chat_message, chat_history, chat_status],
            api_name="general_chat",
        )
        chat_message.submit(
            continue_general_chat,
            inputs=[chat_message, chat_history, memory_state],
            outputs=[chat_message, chat_history, chat_status],
            api_name=False,
        )
        chat_clear.click(
            reset_general_chat,
            outputs=[chat_history, chat_status],
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
