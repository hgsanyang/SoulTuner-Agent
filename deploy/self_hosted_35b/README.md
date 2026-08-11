# SoulTuner V4.2 35B 自托管部署

这个目录只解决一件事：把训练后的 SoulTuner Planner 作为私有或自托管服务接入主项目。它不绑定 ModelScope 创空间，也不绑定 AMD、NVIDIA 或某一家云厂商。

主工程的 LangGraph、Neo4j、Dense 音频检索、联网发现、长期记忆、反馈与前端保持不变；模型只负责提出 `PlannerDecisionV5` 候选，`planner_guard.py` 再做结构校验、Lane 角色检查和确定性编译。

```text
SoulTuner 前端 / Agent
        │
        ├─ Qwen3.7 Plus API
        │
        └─ SoulTuner V4.2 35B OpenAI-compatible endpoint
                          │
                          └─ Qwen3.6-35B-A3B base + SoulTuner LoRA adapter
```

## 一次切换模型

| 页面选择 | 模型运行位置 | 当前电脑需要什么 |
|---|---|---|
| `Qwen3.7 Plus（云端，4070 可用）` | DashScope | API Key，不加载 35B |
| `SoulTuner V4.2 35B（自托管）` | 自有 GPU 服务器或托管 GPU 工作区 | 只访问 HTTPS 端点 |
| `安全演示（无需模型）` | 本机确定性规则 | 无 Key、无 GPU |

本机启动 UI：

```powershell
cd deploy/self_hosted_35b
python -m pip install -r requirements.txt
$env:DASHSCOPE_API_KEY="你的 DashScope Key"
$env:SOULTUNER_MODEL_PROFILE="qwen3.7-plus"
python app.py
```

打开 `http://127.0.0.1:7860`。RTX 4070 只运行应用和客户端，不承载 35B 权重。

## 硬件边界

官方 BF16 基座文件约 72 GB；推理还需要 KV cache、临时张量和运行时空间。因此原生 BF16 自托管建议至少约 96 GB 可用显存/HBM，192 GB 单卡环境最宽松。本项目已经在 AMD MI308X 192 GB 上完成端点验证，但同一 OpenAI 兼容接口也可以由其他满足显存和软件兼容条件的服务器提供。

12 GB RTX 4070 不能直接承载这个 BF16 基座。4-bit 量化可能显著减小权重，但当前训练评测没有验证量化后的 Planner 指标，所以它是后续优化项，不作为默认部署路径。

## 推荐的模型发布结构

不要把 72 GB 基座重复上传到 SoulTuner 仓库。发布 **LoRA adapter** 即可，使用者分别下载官方基座和 SoulTuner adapter：

```text
SoulTuner-Planner-V4.2-35B-LoRA/
├── adapter_model.safetensors
├── adapter_config.json
├── README.md
├── LICENSE
├── NOTICE
└── SHA256SUMS
```

公开模型仓库不要包含 `optimizer.pt`、`scheduler.pt`、`rng_state*.pth`、`trainer_state.json`、私有训练集、regression/sealed 原文或任何访问令牌。这些文件只属于私有断点恢复与审计归档。

推荐发布位置：

1. **ModelScope 模型库**：面向国内下载和后续创空间，作为主发布源；
2. **Hugging Face Hub**：作为国际镜像和标准 PEFT 分发源；
3. **GitHub**：只放代码、文档、模型卡和 Hub 链接，不放模型权重。

`prepare_adapter_release.py` 会从私有 checkpoint 创建一个干净发布副本，并把训练机绝对路径改成公开基座 ID；它不会修改原 checkpoint：

```bash
python prepare_adapter_release.py \
  --checkpoint /private/path/checkpoint-450 \
  --output /private/path/SoulTuner-Planner-V4.2-35B-LoRA \
  --base-model Qwen/Qwen3.6-35B-A3B
```

发布前把 `MODEL_CARD_TEMPLATE.md` 复制为发布目录的 `README.md`，把 `NOTICE_TEMPLATE` 复制为 `NOTICE`，再加入 Apache-2.0 `LICENSE`。模板已经包含本次训练身份和聚合指标；正式公开前仍需确认 adapter 许可证同时满足训练数据授权。

## 从模型 Hub 下载

### ModelScope（国内与创空间优先）

正式 LoRA Adapter 已发布在：

- ModelScope：<https://modelscope.cn/models/hgsanyang/SoulTuner-Planner-V4.2-35B-LoRA>
- 模型 ID：`hgsanyang/SoulTuner-Planner-V4.2-35B-LoRA`

在 GPU 服务器执行下面的脚本即可下载官方基座、下载 Adapter，并核对 Adapter SHA-256：

```bash
python -m pip install -U modelscope
bash download_modelscope_assets.sh
```

默认下载到当前目录的 `models/`。可通过 `SOULTUNER_MODEL_ROOT` 修改位置。等价的手动命令是：

```bash
python -m pip install -U modelscope
modelscope download Qwen/Qwen3.6-35B-A3B \
  --repo-type model --revision master \
  --local-dir /models/qwen3.6-35b-a3b
modelscope download hgsanyang/SoulTuner-Planner-V4.2-35B-LoRA \
  --repo-type model --revision master \
  --local-dir /models/soultuner-v4.2-adapter
cd /models/soultuner-v4.2-adapter
sha256sum --check SHA256SUMS
```

ModelScope 已提供 `Qwen/Qwen3.6-35B-A3B` 官方模型页，因此创空间可以直接从国内 Hub 拉取基座和 SoulTuner adapter。生产环境应固定 revision/commit，并在启动前核对 SHA-256。

### Hugging Face

```bash
python -m pip install -U "huggingface_hub[hf_xet]"
hf download Qwen/Qwen3.6-35B-A3B --local-dir /models/qwen3.6-35b-a3b
hf download YOUR_NAME/SoulTuner-Planner-V4.2-35B-LoRA \
  --local-dir /models/soultuner-v4.2-adapter
```

模型 Hub 是权重分发位置；你的电脑不需要先完整下载再转传。可以直接在训练实例上生成干净 adapter 发布目录并上传，创空间或下一台 GPU 服务器再从 Hub 拉取。

## 启动 35B 端点

```bash
export SOULTUNER_BASE_MODEL=/models/qwen3.6-35b-a3b
export SOULTUNER_ADAPTER=/models/soultuner-v4.2-adapter
export SOULTUNER_SERVE_API_KEY='replace-with-a-random-secret'
bash start_35b_endpoint.sh
```

脚本使用 `swift deploy`，默认在 `0.0.0.0:8000` 提供 OpenAI 兼容接口，关闭 thinking、temperature=0、max_new_tokens=1024。先使用训练与评测已验证的原生后端；其他后端或量化方式需要重新做相同评测后再作为默认值。

服务端只应通过 HTTPS、VPN 或私网暴露。不要把 8000 端口和无鉴权服务直接公开到互联网。

## 让应用切到 35B

在运行 `app.py` 或主应用的机器上配置：

```bash
export SOULTUNER_MODEL_PROFILE=soultuner-v4.2-35b
export SOULTUNER_PLANNER_ENDPOINT=https://your-host/v1/chat/completions
export SOULTUNER_PLANNER_MODEL=soultuner-planner-v4.2-35b
export SOULTUNER_PLANNER_PROTOCOL=openai
export SOULTUNER_PLANNER_TOKEN='same-secret'
export SOULTUNER_PLANNER_TIMEOUT=60
python app.py
```

切回 API 只需把 `SOULTUNER_MODEL_PROFILE` 改成 `qwen3.7-plus`。超时、非 JSON、schema 不合法、任务冲突或缺少必需 Lane 时都会降级到安全计划。

## 已完成训练表现

这些数据来自同一固定评分器；只陈述领域 Planner 的结构化任务表现，不代表通用模型能力排名。

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

## 与完整 SoulTuner 项目的关系

本目录是可独立运行的 Planner 接入切片，不替代完整系统：

- **Graph**：Neo4j 中的歌曲、艺人、标签与关系约束；
- **Dense**：音频/语义向量相似度与听感召回；
- **Web**：本地库不足时的受控发现；
- **Memory**：用户偏好、会话状态和可撤销反馈；
- **Data flywheel**：经用户授权和人工审核的失败样本进入独立候选池，再经过版本化数据准入与离线评测。

公开部署只展示获得授权的元数据和示例，不应把私有歌库、用户记忆或训练评测数据打包进镜像。
