"""Gradio entrypoint for the self-hosted SoulTuner 35B Planner demo."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import urllib.error
import urllib.request
from typing import Any

import gradio as gr

from model_profiles import (
    PROFILE_LABELS,
    default_profile,
    profile_choices,
    resolve_profile,
)
from planner_guard import format_route_markdown, guard_candidate, parse_candidate_content
from prompt_v42 import STUDENT_SYSTEM_PROMPT_V4_2


def _runtime_status(profile: str | None = None) -> str:
    has_kfd = os.path.exists("/dev/kfd")
    rocminfo = shutil.which("rocminfo")
    nvidia_smi = shutil.which("nvidia-smi")
    gpu_line = "未检测到本机 GPU（可使用远程端点或安全演示）"
    if has_kfd:
        gpu_line = "已检测到 ROCm GPU"
        if rocminfo:
            try:
                output = subprocess.run(
                    [rocminfo], capture_output=True, text=True, timeout=4, check=False
                ).stdout
                names = [
                    name
                    for line in output.splitlines()
                    if "Marketing Name:" in line
                    and (name := line.split(":", 1)[1].strip())
                    and "intel" not in name.casefold()
                ]
                if names:
                    gpu_line += "：" + " / ".join(dict.fromkeys(names))
            except (OSError, subprocess.SubprocessError):
                pass
    elif nvidia_smi:
        try:
            output = subprocess.run(
                [nvidia_smi, "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=4,
                check=False,
            ).stdout
            names = [line.strip() for line in output.splitlines() if line.strip()]
            gpu_line = "已检测到 CUDA GPU"
            if names:
                gpu_line += "：" + " / ".join(dict.fromkeys(names))
        except (OSError, subprocess.SubprocessError):
            pass
    selected = profile or default_profile()
    config, _ = resolve_profile(selected)
    mode = PROFILE_LABELS.get(selected, selected)
    if config is None:
        mode += " → 安全回退"
    return f"**运行模式：** {mode}  \n**GPU：** {gpu_line}  \n**Python：** {platform.python_version()}"


def _call_candidate(
    user_text: str, context: dict[str, Any], profile: str
) -> tuple[dict[str, Any] | None, str]:
    config, profile_note = resolve_profile(profile)
    if config is None:
        return None, profile_note

    protocol = str(config["protocol"])
    if protocol == "openai":
        user_context = {"current_input": user_text, **context}
        request_payload = {
            "model": str(config["model"]),
            "messages": [
                {"role": "system", "content": STUDENT_SYSTEM_PROMPT_V4_2},
                {"role": "user", "content": json.dumps(user_context, ensure_ascii=False)},
            ],
            "temperature": 0,
            "max_tokens": 1024,
            "stream": False,
            "enable_thinking": False,
        }
    else:
        request_payload = {"current_input": user_text, "context": context}
    body = json.dumps(request_payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    token = str(config["token"])
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(
        str(config["endpoint"]), data=body, headers=headers, method="POST"
    )
    try:
        timeout = max(1.0, float(os.getenv("SOULTUNER_PLANNER_TIMEOUT", "30")))
    except ValueError:
        timeout = 30.0
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if protocol == "openai" and isinstance(payload, dict):
            choices = payload.get("choices") or []
            content = choices[0].get("message", {}).get("content", "") if choices else ""
            payload = parse_candidate_content(str(content))
        elif isinstance(payload, dict) and isinstance(payload.get("decision"), dict):
            payload = payload["decision"]
        return (payload if isinstance(payload, dict) else None), f"{profile_note}调用成功"
    except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
        return None, f"候选端点失败：{type(exc).__name__}"


def plan_music_request(
    user_text: str, reference_title: str, reference_artist: str, model_profile: str
):
    context = {
        "reference_title": str(reference_title or "").strip(),
        "reference_artist": str(reference_artist or "").strip(),
    }
    candidate, endpoint_note = _call_candidate(
        str(user_text or ""), context, str(model_profile or default_profile())
    )
    plan, findings = guard_candidate(str(user_text or ""), candidate, context)
    diagnostics = "\n".join([f"- {endpoint_note}", *[f"- {item}" for item in findings]])
    return plan, format_route_markdown(plan), diagnostics, _runtime_status(model_profile)


ABOUT = """
### 部署方式

在页面上切换一次模型即可，无需修改业务代码：

1. **Qwen3.7 Plus**：4070 或无本地大模型环境时直接使用云端 API；
2. **SoulTuner V4.2 35B**：接入自托管或托管 GPU 上的训练模型 endpoint；
3. **安全演示**：不调用模型，仍可展示 Graph / Dense 路由。

两种模型输出都经过相同的结构校验、Lane 角色检查与确定性编译。`brief_reason` 是不超过 80 字的公开依据，不是隐藏思维链。
"""


with gr.Blocks(title="SoulTuner · Evidence-first Music Planner") as demo:
    gr.Markdown(
        "# SoulTuner · 证据优先音乐检索规划器\n"
        "领域微调的 35B Planner 候选 + 确定性安全守卫 + Graph/Dense 分工。"
    )
    with gr.Row():
        with gr.Column(scale=3):
            request_text = gr.Textbox(
                label="告诉 SoulTuner 你想听什么",
                value="我今天心情很差，想听一些温暖、治愈但不要太吵的歌",
                lines=4,
            )
            model_profile = gr.Dropdown(
                label="Planner 模型",
                choices=profile_choices(),
                value=default_profile(),
                interactive=True,
            )
            with gr.Accordion("可选：上一首参考歌曲", open=False):
                reference_title = gr.Textbox(label="歌名")
                reference_artist = gr.Textbox(label="歌手")
            submit = gr.Button("生成受保护的检索计划", variant="primary")
            gr.Examples(
                examples=[
                    ["我今天心情很差，想听温暖治愈的歌", "", ""],
                    ["低音更重、鼓点更大的歌", "", ""],
                    ["找一些适合深夜学习的爵士", "", ""],
                    ["我要和刚刚那首歌听感相似的", "Dreams", "Fleetwood Mac"],
                    ["怎么把网易云歌单导进来？", "", ""],
                    ["介绍一下《Bohemian Rhapsody》的创作背景", "", ""],
                ],
                inputs=[request_text, reference_title, reference_artist],
            )
        with gr.Column(scale=2):
            runtime = gr.Markdown(value=_runtime_status(default_profile()))
            route = gr.Markdown()
            diagnostics = gr.Markdown()
    plan_json = gr.JSON(label="可审计 PlannerDecision")
    gr.Markdown(ABOUT)

    submit.click(
        fn=plan_music_request,
        inputs=[request_text, reference_title, reference_artist, model_profile],
        outputs=[plan_json, route, diagnostics, runtime],
    )
    demo.load(
        fn=plan_music_request,
        inputs=[request_text, reference_title, reference_artist, model_profile],
        outputs=[plan_json, route, diagnostics, runtime],
    )
    model_profile.change(
        fn=_runtime_status,
        inputs=[model_profile],
        outputs=[runtime],
    )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=8).launch(server_name="0.0.0.0", server_port=7860)
