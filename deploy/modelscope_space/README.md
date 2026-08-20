---
license: MIT License
domain:
  - audio
  - natural-language-processing
tags:
  - AMD GPU
  - ROCm
  - music-recommendation
  - hybrid-retrieval
  - graph-rag
  - llm-agent
models:
  - hgsanyang/SoulTuner-Planner-V4.2-35B-LoRA
datasets:
  test:
    - hgsanyang/SoulTuner-Open-Audio-Demo
---

# SoulTuner Agent

SoulTuner 是一个自然语言音乐推荐 Agent。用户用一句话描述想听的内容，系统将请求规划为 Graph、Dense、Web 或免检索任务，再通过受控检索、结果融合、会话记忆和反馈闭环返回推荐。

本创空间以一个可直接运行的单体应用展示完整用户路径：

1. 自然语言请求与上下文理解；
2. SoulTuner Planner 生成结构化检索决策；
3. Graph 处理歌曲、艺人、流派、年代、标签和目录约束；
4. Dense 处理听感、氛围、音色与参考歌曲相似性；
5. 结果融合、推荐理由与受控输出；
6. 喜欢、跳过和不喜欢反馈写入当前会话记忆；
7. 同一应用可切换公开演示、远程 API 和 AMD MI308X 本地 35B 三种 Planner 档位。

公开演示使用 `SoulTuner-Open-Audio-Demo` 中 5 首逐曲核验的 Song Describer/Jamendo 原始音频，不包含个人数据、训练集或 sealed 评测答案。每首歌均保留上游地址、归属文本、许可证和 SHA-256；其中的 NoDerivatives 文件只按原始字节播放，不做转码、剪辑或重封装。

## 一键运行

```bash
python -m pip install -r requirements.txt
python app.py
```

默认使用 `demo-heuristic`，无需 GPU、模型权重或 API Key，适合创空间自动审核和 CPU 预览。

## Planner 档位

| 档位 | 用途 | 必要配置 |
|---|---|---|
| `demo-heuristic` | CPU 与自动审核 | 无 |
| `qwen-api` | 复用兼容 OpenAI 协议的云端 Planner | `SOULTUNER_PLANNER_BASE_URL`、`SOULTUNER_PLANNER_API_KEY` |
| `soultuner-v4.2-35b` | AMD MI308X 上运行训练后的 35B LoRA | 启动本地 endpoint 后配置相同的 Planner URL |

启动应用时可设置默认档位：

```bash
export SOULTUNER_MODEL_PROFILE=demo-heuristic
python app.py
```

## AMD MI308X / ROCm

AMD MI308X 并不是创建创空间后自动可用。完整顺序是：完成 AMD 开发者注册 →
申请加入 `AMD_Dev` 组织 → 审核通过 → 在现有 SoulTuner 创空间的部署设置中选择
AMD MI308X 和对应 ROCm 镜像。在组织审核完成前，继续保留 CPU 档位即可。

获批 AMD GPU 资源后，在对应 ROCm 镜像中把
`SOULTUNER_MODEL_PROFILE` 设为 `soultuner-v4.2-35b`。ModelScope 的 Gradio
创空间仍从 `app.py` 启动；应用会在后台运行 `start_amd_35b.sh`，让 Gradio 先
绑定公开端口，再完成依赖检查、基座与 LoRA 下载及 vLLM 启动。首次下载期间，
界面保持可用并对 Planner 请求进行安全回退。

在普通终端中也可以显式执行一体化入口：

```bash
python -m pip install -r requirements-amd.txt
bash start_space_amd.sh
```

这个入口会按顺序完成 ROCm/HIP 预检、基座与 LoRA 下载、可用时校验
`SHA256SUMS`、启动本地 OpenAI 兼容端点、等待 `/v1/models` 健康，然后启动现有
Gradio 界面。任何关键步骤失败都会停止，不会悄悄退回 CPU 承载 35B。

创空间自动启动路径会把同样的端点流程放到后台，并将日志写到
`soultuner-35b-endpoint.log`。若 ROCm 镜像缺少 `vllm`，启动脚本会在下载模型前
明确失败；不要在 AMD 资源上改用 CUDA wheel。

也可以分两个进程调试：

```bash
SOULTUNER_REQUIRE_ROCM=1 bash start_amd_35b.sh
# 另一个终端
SOULTUNER_MODEL_PROFILE=soultuner-v4.2-35b \
SOULTUNER_PLANNER_BASE_URL=http://127.0.0.1:8000/v1 python app.py
```

应用会显示 ROCm、HIP 和 GPU 探测结果。PyTorch 在 AMD GPU 上仍通过 `torch.cuda` 命名空间访问设备，是否为 ROCm 运行时以 `torch.version.hip` 为准。

### GPU 获批后的部署变量

| 变量 | 推荐值 | 说明 |
|---|---|---|
| `SOULTUNER_REQUIRE_ROCM` | `1` | 未检测到 ROCm/HIP 时拒绝启动 35B |
| `SOULTUNER_MODEL_PROFILE` | `soultuner-v4.2-35b` | 默认选择训练后的 Planner |
| `SOULTUNER_BASE_MODEL_ID` | `Qwen/Qwen3.6-35B-A3B` | 官方基座，不重复上传到项目仓库 |
| `SOULTUNER_ADAPTER_MODEL_ID` | `hgsanyang/SoulTuner-Planner-V4.2-35B-LoRA` | 公开 LoRA 仓库 |
| `SOULTUNER_MODEL_CACHE` | 平台持久化目录 | 避免每次休眠后重复下载 72 GB 基座 |
| `SOULTUNER_INFER_BACKEND` | `vllm` | 首次部署沿用当前启动方式，优化后再重新评测 |

ModelScope Access Token 仅在模型仍为私有或受限下载时作为“密文变量”配置；模型公开后不应再要求令牌。任何令牌都不得写入仓库、README 或普通明文变量。

未显式设置 `SOULTUNER_MODEL_CACHE` 时，启动脚本会在可写的
`/mnt/workspace` 上自动使用 `/mnt/workspace/soultuner/model_cache`；只有非创空间
环境才退回仓库旁的 `./model_cache`。

### 挂接公开授权音频目录

Gradio 稳定版不需要切换为 Docker 也能播放真实音频。`start_space_amd.sh`
会在 35B 基座下载的同时下载并校验公开音频数据集，默认物化到持久盘。也可以显式配置：

```bash
SOULTUNER_CATALOG_PATH=/mnt/workspace/soultuner/open_audio/catalog.jsonl
SOULTUNER_AUDIO_ROOT=/mnt/workspace/soultuner/open_audio/audio
```

目录行可使用 `audio_relpath`（推荐）、`audio_path`、`audio_url` 或 `preview_url`。
本地文件必须位于 `SOULTUNER_AUDIO_ROOT` 内，路径穿越和不支持的扩展名会被拒绝；
只有这个目录会加入 Gradio 文件白名单。检索后页面会显示音频播放器，选择另一首歌曲
会同步切换试听。音频来源、许可证、校验和应保留在独立数据集清单中。

### 先用训练实例完成推理实验

AMD 创空间资源是否获批不影响推理验证。之前完成训练的 AMD 实例已经缓存基座、
LoRA 和 ROCm 环境，是当前最快的验证机。它只承担短期实验，不承担长期公网服务：

1. 保留训练和评测归档，不重训、不改 checkpoint；
2. 从 `checkpoint-450` 的干净发布副本启动本地 OpenAI 兼容端点；
3. 用公开代表性请求测量加载时间、结构契约通过率、端到端延迟和并发吞吐；
4. 记录 GPU/HBM 峰值与运行参数；
5. 获批 MI308X 创空间后原样迁移通过验证的参数。

端点只通过 SSH 隧道、VPN 或私网访问，不把无鉴权的 8000 端口暴露到公网。实例
到期、SSH 地址变化或平台回收都可能中断服务，因此 CPU 创空间继续作为公开演示入口。

在端点健康后运行公开基准（不读取 regression 或 sealed 数据）：

```bash
python ../self_hosted_35b/benchmark_endpoint.py \
  --endpoint http://127.0.0.1:8000/v1/chat/completions \
  --model soultuner-planner-v4.2-35b \
  --warmup 1 --repeat 3 \
  --json reports/inference-baseline.json
```

报告只保存匿名场景 ID、契约通过率和延迟统计，不保存请求正文或模型原始回答。

### 推理加速的实施顺序

加速与公开部署并行推进，但每次只改变一组参数，使用同一批公开请求比较：

1. **基线**：BF16 LoRA、vLLM、thinking 关闭、`max_model_len=4096`、
   `gpu_memory_utilization=0.90`、`max_num_seqs=16`、prefix cache 开启；
2. **并发档位**：分别测试 1、4、8、16 并发，选择无 OOM 且契约保持有效的档位；
3. **AMD 优化对照**：镜像确实安装 AITER 后，单独设置
   `VLLM_ROCM_USE_AITER=1`，与基线比较吞吐和延迟；
4. **后续研究**：SGLang 和 4-bit/FP8 量化需要重新验证 Planner 结构准确率，当前不
   直接替换已经可用的 BF16 默认路径。

`SOULTUNER_MAX_MODEL_LEN`、`SOULTUNER_GPU_MEMORY_UTILIZATION`、
`SOULTUNER_MAX_NUM_SEQS` 和 `SOULTUNER_ENABLE_PREFIX_CACHING` 均可通过环境变量调整。
只有端点健康、契约通过且显存稳定的组合才进入创空间配置。

当前 MI308X 实测中，标准 vLLM 与 AITER 两组各完成 60/60 条公开请求，结构、守卫和
安全编译通过率均为 100%。16 并发下标准路径为 1.640 请求/秒，AITER 为 1.671
请求/秒；由于增益较小且当前版本仍出现算子回退，创空间默认继续使用标准 vLLM，
AITER 仅作为可复测的可选优化。完整公开汇总见
`../self_hosted_35b/INFERENCE_BENCHMARK.md`。

## 公开资源

- 完整工程：[SoulTuner-Agent](https://github.com/hgsanyang/SoulTuner-Agent)
- 35B LoRA：[SoulTuner-Planner-V4.2-35B-LoRA](https://modelscope.cn/models/hgsanyang/SoulTuner-Planner-V4.2-35B-LoRA)
- 开放音频演示集：[SoulTuner-Open-Audio-Demo](https://modelscope.cn/datasets/hgsanyang/SoulTuner-Open-Audio-Demo)
- Notebook：[SoulTuner 35B 音乐推荐规划器与混合检索实践](https://modelscope.cn/gallery/hgsanyang/soultuner-v4-2-35b-music-planner)
- 技术文章：[从一句话到可执行检索计划](https://modelscope.cn/learn/435660)

## 数据与安全边界

- 不在前端或仓库保存 API Key、ModelScope 令牌和用户凭据；
- 公开演示只使用逐曲许可的开放音频；页面始终显示归属、许可证和上游链接；
- 演示曲目均为非商业许可，不得将其扩大为商业音乐库；NoDerivatives 曲目不改写原始音频；
- Planner 输出必须通过结构校验和 Policy Guard 后才能触发检索；
- 生产部署需另外配置鉴权、限流、内容安全和持久化数据库；
- 35B LoRA 需与其基座模型许可、模型卡和使用约束一并遵守。

## License

应用代码采用 MIT License。音频数据不使用一个统一代码许可：每首曲目按数据集 manifest 中的 CC BY-NC / BY-NC-ND / BY-NC-SA 等逐曲条款使用。模型与基座模型分别遵循各自模型卡许可。
