# 🎵 SoulTuner Agent

<p align="center">
  <img src="assets/logo.png" alt="logo" width="200" />
</p>

<p align="center">
  <strong>基于自然语言的音乐推荐智能体</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue?logo=python" alt="Python" />
  <img src="https://img.shields.io/badge/LangGraph-Agent_Framework-orange?logo=langchain" alt="LangGraph" />
  <img src="https://img.shields.io/badge/Neo4j-Graph_Database-008CC1?logo=neo4j" alt="Neo4j" />
  <img src="https://img.shields.io/badge/Next.js_14-Frontend-black?logo=next.js" alt="Next.js" />
  <img src="https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker" alt="Docker" />
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License" />
  <br/>
  <img src="https://github.com/hgsanyang/SoulTuner-Agent/actions/workflows/ci.yml/badge.svg" alt="CI" />
  <img src="https://img.shields.io/badge/tests-CI_passing-brightgreen?logo=pytest" alt="Tests" />
  <img src="https://img.shields.io/badge/code_style-ruff-261230?logo=ruff" alt="Ruff" />
</p>

<p align="center">
  <a href="README.ch.md">中文</a> | <a href="README.md">English</a>
</p>

## 🎯 这是什么

SoulTuner 是一个音乐推荐智能体。你用一句人话描述想听什么，它负责听懂，然后给你歌。

- 🗣️ **说人话就行** — "今天心情特别差，想一个人静一静"，不需要你先想好流派和关键词
- 🧠 **越用越懂你** — 点赞、收藏、跳过和每次对话都会更新你的偏好画像，影响之后的排序
- 🌐 **库里没有就去网上找** — 找有榜单或口碑支撑的歌（可一键关闭）
- ♻️ **发现→试听→入库** — 遇到好歌先进暂存区试听，确认后再入库
- 🧪 **日常 / 开发两种模式** — 开发模式的数据独立存放，不参与个性化学习，也不进训练集

> 📖 完整功能与交互细节见 [Feature_Walkthrough.md](Feature_Walkthrough.md)

---

## 🖼️ 功能预览

<p align="center">
  <a href="https://www.bilibili.com/video/BV11dQLBDEeF/">
    <img src="https://img.shields.io/badge/▶_演示视频_—_B站观看_|_BV11dQLBDEeF-00A1D6?style=for-the-badge&logo=bilibili&logoColor=white&labelColor=FB7299" alt="演示视频" />
  </a>
</p>

### 🏠 首页 · 💬 对话 · 🎵 推荐 · 🎧 播放

<table>
  <tr>
    <td><img src="assets/首页.png" alt="首页" /></td>
    <td><img src="assets/对话页面.png" alt="对话" /></td>
  </tr>
  <tr>
    <td><img src="assets/音乐推荐.png" alt="推荐" /></td>
    <td><img src="assets/播放页1.png" alt="播放" /></td>
  </tr>
</table>

---

## 🚀 快速启动

```powershell
cd <你的项目目录>
Copy-Item .env.example .env
notepad .env
```

`.env` 里至少填这几项（默认用 DashScope / Qwen）：

```env
MAIN_LLM_PROVIDER=dashscope
MODEL_NAME=qwen3.7-plus
DASHSCOPE_API_KEY=你的 DashScope Key
NEO4J_PASSWORD=你的 Neo4j 密码
MUSIC_DATA_PATH=../data
```

然后启动，打开 `http://localhost:3003`：

```powershell
.\soultuner.ps1 up gpu
```

没有 NVIDIA 显卡就用 `.\soultuner.ps1 up cpu`。

想换模型厂商（SiliconFlow / Google / 火山 / 本地 SGLang、VLLM、Ollama），改 `MAIN_LLM_PROVIDER` 和 `MODEL_NAME` 并填对应 Key 即可，也可以启动后在前端「系统设置」里改。

### 自托管训练后的 SoulTuner 35B Planner

仓库已提供独立的 35B 自托管部署包：[deploy/self_hosted_35b](deploy/self_hosted_35b)。Planner 通过 OpenAI 兼容端点接入；在 Qwen3.7 Plus 与微调后的 SoulTuner 模型之间切换时，检索、记忆、排序和前端代码都不用修改。

| 档位 | 运行位置 | 本地显卡要求 |
|---|---|---|
| Qwen3.7 Plus API | 当前电脑或任意 CPU 主机 | 不加载大模型，RTX 4070 足够运行其余服务 |
| SoulTuner V4.2 35B | 自有 GPU 服务器或托管 GPU 工作区 | 35B 基座与 LoRA adapter 留在推理服务器 |
| 安全演示 | 任意环境 | 无 |

当前电脑直接这样启动：

```powershell
cd deploy/self_hosted_35b
python -m pip install -r requirements.txt
$env:DASHSCOPE_API_KEY="你的 Key"
$env:SOULTUNER_MODEL_PROFILE="qwen3.7-plus"
python app.py
```

在合适的 GPU 服务器上，从所选模型仓库分别下载官方 `Qwen/Qwen3.6-35B-A3B` 基座与 SoulTuner PEFT adapter，启动端点，再选择 **SoulTuner V4.2 35B**。通用部署包支持本地路径、Hugging Face 兼容仓库和任意 OpenAI 兼容推理端点；特定平台镜像放在独立适配目录。部署包 README 已包含硬件规格、完整 Agent 接入、完整性校验、服务启动和实测推理数据。已验证环境是 AMD MI308X，但应用契约不绑定某个云平台或 GPU 品牌。

完整服务也提供 AMD ROCm Compose 覆盖层。它保留原有 CPU 和 NVIDIA CUDA
档位，同时使用 AMD 官方 PyTorch 镜像与 ROCm 设备映射：

```bash
docker compose -f docker-compose.yml -f docker-compose.amd.yml --profile gpu up -d --build
```

主机要求与运行时验证见 [AMD_ROCM_DEPLOYMENT.md](docs/AMD_ROCM_DEPLOYMENT.md)。

<details>
<summary>其它常用命令</summary>

| 命令 | 用途 |
|---|---|
| `.\soultuner.ps1 doctor` | 检查各服务是否正常 |
| `.\soultuner.ps1 down` | 停止所有容器 |
| `.\soultuner.ps1 logs` | 查看服务日志 |
| `.\soultuner.ps1 test` | 运行单元测试 |
| `.\soultuner.ps1 ingest gpu` | 用 GPU Worker 处理待入库歌曲 |
| `python scripts/dev/start_backend.py` | 仅启动后端，供本地调试 |

</details>

---

## 🏗️ 架构

一次推荐请求会走完这条链路：

```
你说的一句话
     │
     ▼
┌──────────────────────────────────────────────────┐
│  Agent（LangGraph）                               │
│  召回记忆 → LLM 规划 → 按意图分流                  │
│  找歌 / 闲聊 / 获取歌曲 / 澄清追问                 │
└──────────────────────┬───────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────┐
│  本地检索与曲库扩展                                │
│  图谱召回 ＋ MuQ 向量召回 → 融合 → 按需联网补充    │
└──────────────────────┬───────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────┐
│  存储：Neo4j（图谱＋向量＋行为热路径）             │
│        SQLite（长期记忆账本 ＋ 反馈事件）          │
└──────────────────────┬───────────────────────────┘
                       ▼
        SSE 流式推送 → 前端（Next.js）
                       │
                       ▼
        你的反馈 ──────┘  记录下来，更新你的偏好画像
```

### 技术栈

| 层 | 用什么 |
|---|---|
| 前端 | Next.js 14 + React 18 |
| 后端 | FastAPI + SSE 流式推送 |
| Agent | LangGraph StateGraph |
| 图数据库 | Neo4j 5.x（图谱关系 + 原生向量索引） |
| 文搜音 | MuQ-MuLan（M2D-CLAP 回退） |
| 大语言模型 | 默认 `dashscope / qwen3.7-plus`，可换 provider |
| 长期记忆 | 本地 SQLite 账本 + Neo4j 热路径 |
| 排序 | 多路召回融合 → 精排 → 多样性 |
| 部署 | Docker Compose（CPU / GPU 两种入口） |

> 📖 推荐质量与对齐评测怎么跑，见 [tests/eval/README.md](tests/eval/README.md)

---

## 📁 项目结构

```
agent/       LangGraph 工作流与意图路由
retrieval/   混合检索、融合排序、音频编码、上下文管线
tools/       图谱检索 / 文搜音 / 联网发现 / 歌曲获取
services/    记忆网关、反馈事件、排序策略、外部服务客户端
schemas/     Pydantic 契约（状态、查询计划、反馈事件）
llms/        Provider 注册表与 Prompts
api/         FastAPI 接口层
data/        数据管线与 Planner 蒸馏 harness
web/         Next.js 前端
tests/       单元测试 + 结果导向评测
```

Planner 支持蒸馏为本地学生模型。公开仓库只提供可复现的训练 harness，私有训练数据不会进入 Git；详见 [data/sft/README.md](data/sft/README.md)。

---

## ⚙️ 配置

| 变量 | 说明 |
|---|---|
| `DASHSCOPE_API_KEY` | 默认模型的调用密钥（换 provider 就填对应厂商的） |
| `NEO4J_PASSWORD` | 本地 Neo4j 密码 |
| `MUSIC_DATA_PATH` | 音频、缓存、待入库队列、反馈日志的存放目录 |
| `MUSIC_WEB_SEARCH_ENABLED` | 是否允许联网补充候选 |
| `ADMIN_API_KEY` | 可选。设了它，删除/改配置/重建这些危险接口才需要带 key |

更多高级选项见 `.env.example`，普通使用不需要动。

服务只监听 `127.0.0.1`。要远程访问走 VPN 或 SSH 隧道。

---

## 🙏 致谢

本项目初始架构参考自 [imagist13/Muisc-Research](https://github.com/imagist13/Muisc-Research)，在此基础上做了大规模重构与功能扩展。

| 项目 | 用途 |
|---|---|
| [OpenMuQ/MuQ](https://github.com/OpenMuQ/MuQ) | MuQ-MuLan 文搜音主模型（CC-BY-NC 4.0） |
| [nttcslab/m2d](https://github.com/nttcslab/m2d) | M2D-CLAP 回退与辅助语义模型 |
| [MTG/omar-rq](https://github.com/MTG/omar-rq) | OMAR-RQ 音频表示模型 |
| [aexy-io/graphzep](https://github.com/aexy-io/graphzep) | legacy 记忆适配器（可选、非默认） |

---

## 📚 参考文献

- Palumbo, E. et al. (2025). *You Say Search, I Say Recs.* RecSys 2025.
- Zhu, H. et al. (2025). *MuQ / MuQ-MuLan: Self-Supervised Music Representation Learning with Mel Residual Vector Quantization.* [arXiv:2501.01108](https://arxiv.org/abs/2501.01108)
- Niizumi, D. et al. (2025). *M2D-CLAP: Exploring General-purpose Audio-Language Representations Beyond CLAP.* IEEE Access. [arXiv:2503.22104](https://arxiv.org/abs/2503.22104)
- Alonso-Jiménez, P. et al. (2025). *OMAR-RQ: Open Music Audio Representation Model.* ACM MM 2025. [arXiv:2507.03482](https://arxiv.org/abs/2507.03482)
- Gao, L. et al. (2023). *Precise Zero-Shot Dense Retrieval without Relevance Labels.* ACL 2023.
- Xu, W. et al. (2025). *A-MEM: Agentic Memory for LLM Agents.* [arXiv:2502.12110](https://arxiv.org/abs/2502.12110)
- Wang, Y. et al. (2023). *RecMind: Large Language Model Powered Agent for Recommendation.* [arXiv:2308.14296](https://arxiv.org/abs/2308.14296)
- Wu, D. et al. (2025). *LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory.* ICLR 2025.
- Manco, I. et al. (2023). *The Song Describer Dataset.* [arXiv:2311.10057](https://arxiv.org/abs/2311.10057)
- Rasmussen, P. et al. (2025). *Zep: A Temporal Knowledge Graph Architecture for Agent Memory.*

---

## 📄 许可证

- **SoulTuner 源码**：MIT（见 [LICENSE](LICENSE)）。
- **MuQ-MuLan 模型权重**：CC-BY-NC 4.0，**仅限非商业用途**。默认配置会下载这份权重，因此**若要商用默认配置，必须替换这些权重或另行取得受限模型的授权**。M2D-CLAP、OMAR-RQ 各自遵循其上游许可证。

⚠️ **免责声明**：本项目仅供学习与架构研究。不提供、不包含也不分发任何受版权保护的音频或歌词资源，音频数据需用户自行通过合法渠道获取。
