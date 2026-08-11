# SoulTuner AMD Creation Space

这是 SoulTuner 的 ModelScope 创空间最小部署包。页面只有一个 **Planner 模型**下拉框，业务代码无需随模型变化：4070/普通电脑使用 Qwen3.7 Plus 云端 API，获得 AMD 创空间后切换到 SoulTuner V4.2 35B；两种模型共用同一套策略守卫和执行编译器。

## 三个部署档位

| 页面选择 | 需要什么 | 适用位置 |
|---|---|---|
| `Qwen3.7 Plus（云端，4070 可用）` | `DASHSCOPE_API_KEY` | 当前 4070 电脑、CPU 创空间 |
| `SoulTuner V4.2 35B（AMD 创空间）` | 35B OpenAI 兼容端点 | AMD MI308X 创空间/服务器 |
| `安全演示（无需模型）` | 无 | 无 Key 的公开预览 |

统一规则：模型只提出 `PlannerDecisionV5` 候选；`planner_guard.py` 检查结构、任务类型和 Graph/Dense/Web 通道角色，再确定性编译。端点超时、缺字段或策略冲突时自动回退。公开 `brief_reason` 最长 80 字；演示不包含私有训练集或 sealed 原文。

## 当前 4070：直接运行，不加载 35B

```powershell
cd deploy/modelscope_space
python -m pip install -r requirements.txt
# 也可参考 .env.example；真实 Key 不要写入仓库
$env:DASHSCOPE_API_KEY="你的 DashScope Key"
$env:SOULTUNER_MODEL_PROFILE="qwen3.7-plus"
python app.py
```

打开 `http://127.0.0.1:7860`，下拉框保持 Qwen3.7 Plus。RTX 4070 不需要、也不建议加载 35B 基座和 adapter。

## 启动并接入 35B 候选端点

在完成训练的 AMD 实例上，指向本次验证过的 `checkpoint-450`：

```bash
export SOULTUNER_BASE_MODEL=/path/to/Qwen3.6-35B-A3B
export SOULTUNER_ADAPTER=/path/to/checkpoint-450
export SOULTUNER_SERVE_API_KEY='replace-with-a-random-secret'
bash start_35b_endpoint.sh
```

脚本调用官方 `swift deploy`，默认在 8000 端口提供 OpenAI 兼容接口，关闭 thinking、temperature=0、max_new_tokens=1024。先使用训练/评测已验证的原生后端；确认 ROCm 版 vLLM 与 LoRA 兼容后，才设置 `SOULTUNER_INFER_BACKEND=vllm`。

创空间 Secret / Variable 中配置：

- `SOULTUNER_PLANNER_ENDPOINT`：完整 OpenAI 地址，例如 `https://host/v1/chat/completions`；
- `SOULTUNER_PLANNER_TOKEN`：可选 Bearer Token，必须放 Secret，不得写进仓库。
- `SOULTUNER_PLANNER_MODEL`：默认 `soultuner-planner-v4.2-35b`；
- `SOULTUNER_PLANNER_PROTOCOL`：默认 `openai`；自建轻量包装服务时可设为 `native`。
- `SOULTUNER_MODEL_PROFILE=soultuner-v4.2-35b`：把页面默认档位切到训练模型。
- `SOULTUNER_PLANNER_TIMEOUT`：默认 30 秒；可按创空间冷启动时间调整。

OpenAI 模式会使用与训练相同的冻结 student system prompt。`native` 模式可以直接返回 V5 JSON，也可以返回 `{"decision": {...}}`。超时、非 JSON、非法 schema、任务冲突和缺少必需 lane 都会触发降级。

## ModelScope 创空间发布

活动使用国内站 `modelscope.cn`。先在网页创建公开 Gradio 创空间，或使用官方 CLI：

```bash
python -m pip install modelscope-hub
ms-hub login --token YOUR_MODELSCOPE_TOKEN
ms-hub create YOUR_NAME/soultuner-amd-planner --repo-type studio --sdk-type gradio --visibility public
```

Gradio 创空间入口必须是仓库根目录的 `app.py`。把本目录中的 `app.py`、`model_profiles.py`、`planner_guard.py`、`prompt_v42.py`、`requirements.txt` 复制到创空间仓库根目录后提交并推送，再执行：

```bash
ms-hub deploy YOUR_NAME/soultuner-amd-planner --repo-type studio
ms-hub logs YOUR_NAME/soultuner-amd-planner --log-type run
```

不要把 Token 直接粘进文章、Notebook 或公开仓库。网页已经登录时优先使用网页创建；CLI 登录只在你本人持有并能安全保存访问令牌时使用。

申请 AMD 专属资源后，在创空间硬件设置中选择批准的 AMD MI308X 规格。付费规格会产生真实费用；没有明确确认前不要选择 `paid/*` 资源。

## 审核材料

- 训练硬件：AMD MI308X，192 GB HBM；
- 模型：35B-A3B，LoRA/adapter 训练，总目标 2 epoch；
- 核心原创点：证据、lane 角色、执行计划三层分离；
- 公开评测：regression 412 + canonical sealed 500；
- 安全演示：候选模型永远经过 fail-closed 守卫；
- 活动标签：`AMD GPU激励计划`。

## 已完成训练表现

| 指标 | Regression 412 | Canonical sealed 500 |
|---|---:|---:|
| Schema valid | 99.51% | 99.40% |
| Compilable | 99.51% | 99.20% |
| Intent / route | 99.03% | 95.60% |
| Lane policy satisfaction | 91.75% | 86.80% |
| Lane role exact match | 78.88% | 76.20% |
| Required lane recall | — | 89.98% |
| HyDE present when Dense | 98.88% | 84.44% |

`Lane policy satisfaction` 检查必需通道是否打开、禁止通道是否关闭；`Lane role exact match` 更严格，要求 Graph、Dense、Web 的 `required / optional / off` 三个角色逐项完全一致。
