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
  train:
    - hgsanyang/SoulTuner-Demo-Catalog
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

公开演示目录仅包含合成元数据和演示向量，不包含版权音频、个人数据、训练集或 sealed 评测答案。生产工程中的 Neo4j、Qdrant、MuQ 音频向量、长期记忆与前端播放器由同一接口接入。

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

获批 AMD GPU 资源后，在对应 ROCm 镜像中执行：

```bash
python -m pip install -r requirements-amd.txt
bash start_amd_35b.sh
```

另开一个进程启动界面：

```bash
export SOULTUNER_MODEL_PROFILE=soultuner-v4.2-35b
export SOULTUNER_PLANNER_BASE_URL=http://127.0.0.1:8000/v1
python app.py
```

应用会显示 ROCm、HIP 和 GPU 探测结果。PyTorch 在 AMD GPU 上仍通过 `torch.cuda` 命名空间访问设备，是否为 ROCm 运行时以 `torch.version.hip` 为准。

## 公开资源

- 完整工程：[SoulTuner-Agent](https://github.com/hgsanyang/SoulTuner-Agent)
- 35B LoRA：[SoulTuner-Planner-V4.2-35B-LoRA](https://modelscope.cn/models/hgsanyang/SoulTuner-Planner-V4.2-35B-LoRA)
- 演示目录：[SoulTuner-Demo-Catalog](https://modelscope.cn/datasets/hgsanyang/SoulTuner-Demo-Catalog)
- Notebook：[SoulTuner 35B 音乐推荐规划器与混合检索实践](https://modelscope.cn/gallery/hgsanyang/soultuner-v4-2-35b-music-planner)
- 技术文章：[从一句话到可执行检索计划](https://modelscope.cn/learn/435660)

## 数据与安全边界

- 不在前端或仓库保存 API Key、ModelScope 令牌和用户凭据；
- 公开演示只使用合成目录，所有反馈只保留在当前浏览器会话；
- Planner 输出必须通过结构校验和 Policy Guard 后才能触发检索；
- 生产部署需另外配置鉴权、限流、内容安全和持久化数据库；
- 35B LoRA 需与其基座模型许可、模型卡和使用约束一并遵守。

## License

应用代码采用 MIT License。公开演示数据采用其数据集页面所列的 CC BY 4.0；模型与基座模型分别遵循各自模型卡许可。
