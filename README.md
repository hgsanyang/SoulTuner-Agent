# 🎵 SoulTuner Agent

<p align="center">
  <img src="assets/logo.png" alt="logo" width="200" />
</p>

<p align="center">
  <strong>多模态音乐推荐智能体 — Hybrid RAG × Knowledge Graph × Long-term Memory</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue?logo=python" alt="Python" />
  <img src="https://img.shields.io/badge/LangGraph-Agent_Framework-orange?logo=langchain" alt="LangGraph" />
  <img src="https://img.shields.io/badge/Neo4j-Graph_Database-008CC1?logo=neo4j" alt="Neo4j" />
  <img src="https://img.shields.io/badge/Next.js_14-Frontend-black?logo=next.js" alt="Next.js" />
  <img src="https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker" alt="Docker" />
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License" />
</p>

> 融合知识图谱（Neo4j）、双模型音频向量（M2D-CLAP + OMAR-RQ）、大语言模型和 GraphZep 长期记忆，通过 LangGraph 编排的多节点 Agent 工作流，实现多路混合检索、加权 RRF 融合、Neo4j 图距离加权、SSE 流式推荐、联网搜索回退、音乐旅程编排和用户行为数据飞轮。

---

## ✨ 核心特性

| 特性 | 说明 |
|------|------|
| 🔀 **Hybrid RAG** | GraphRAG + Semantic Search 并发检索，加权 RRF 融合排序 |
| 🎵 **双模型音频向量** | M2D-CLAP 跨模态语义 × 0.7 + OMAR-RQ 声学特征 × 0.3 |
| 🧠 **长期记忆** | GraphZep 双阶段召回，跨会话保留用户偏好 |
| 📊 **图距离加权** | Neo4j 图亲和力评分，用户历史偏好歌手/流派的歌曲获得排序加成 |
| 🌐 **联网搜索回退** | 本地库不足时自动触发 SearxNG 联邦搜索 + LLM 摘要提取 |
| 🎼 **音乐旅程** | LLM 故事→情绪拆解→逐段检索，SSE 实时推送 |
| ♻️ **数据飞轮** | 用户一键入库：搜索→发现→下载→标签提取→向量编码→Neo4j |
| 📡 **SSE 流式** | 前端实时渲染 thinking → 歌曲卡片 → 推荐理由 |
| ⚙️ **运行时配置** | 前端设置面板可实时调整 LLM、检索参数、RRF 权重 |
| 🐳 **Docker 部署** | `docker compose up` 一键启动全栈 |

---

## 🖼️ 功能预览

### 🏠 首页 · 💬 对话 · 🎵 推荐 · 🎧 播放 · 🗺️ 旅程

<table>
  <tr>
    <td><img src="assets/首页.png" alt="首页" /></td>
    <td><img src="assets/对话页面.png" alt="对话" /></td>
  </tr>
  <tr>
    <td><img src="assets/音乐推荐.png" alt="推荐" /></td>
    <td><img src="assets/播放页1.png" alt="播放" /></td>
  </tr>
  <tr>
    <td colspan="2"><img src="assets/音乐旅程.png" alt="旅程" /></td>
  </tr>
</table>

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────────────┐
│  Frontend (Next.js :3003)                                           │
│  React UI  ·  Global Audio Player  ·  Music Journey  ·  Settings   │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ SSE
┌──────────────────────────────▼──────────────────────────────────────┐
│  Backend (FastAPI :8501)                                            │
│  SSE Streaming API  ·  Settings API  ·  Static Audio Server        │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│  LangGraph Agent (StateGraph)                                       │
│                                                                     │
│  start → GraphZep Recall → Unified Planner (LLM) → 5-way Router   │
│                                                                     │
│     ┌──────────┬──────────┬──────────┬──────────┐                  │
│     ▼          ▼          ▼          ▼          ▼                  │
│   search    chat     play_song   acquire   journey                 │
│     │                                                               │
│     ▼                                                               │
│  Hybrid Retrieval ──→ LLM Explainer ──→ GraphZep Write → end      │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│  Hybrid Retrieval Engine                                            │
│                                                                     │
│  ┌─────────────┐  ┌──────────────────┐  ┌──────────────┐          │
│  │  GraphRAG   │  │  Semantic Search  │  │  Web Search  │          │
│  │  Neo4j      │  │  M2D-CLAP+OMAR   │  │  SearxNG     │          │
│  └──────┬──────┘  └────────┬─────────┘  └──────┬───────┘          │
│         └──────────────────┼───────────────────┘                   │
│                            ▼                                        │
│              Weighted RRF Fusion (α·Vector + β·Graph)              │
│                            ▼                                        │
│              Neo4j Graph Affinity Scoring                           │
│                            ▼                                        │
│              Artist Diversity Filter (指定歌手豁免)                  │
│                            ▼                                        │
│              MMR Genre-aware Rerank (λ=0.7)                        │
└─────────────────────────────────────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│  Storage Layer                                                      │
│  Neo4j (Graph + Vectors)  ·  GraphZep Memory (:3100)               │
└─────────────────────────────────────────────────────────────────────┘
```

### 技术栈

| 层 | 技术 |
|---|---|
| **前端** | Next.js 14 + React 18 + Framer Motion |
| **Agent** | LangGraph StateGraph（11 节点 + 5 条件路由） |
| **后端** | FastAPI + Uvicorn，SSE 流式推送，运行时设置 API |
| **图数据库** | Neo4j（原生向量索引 + 图谱关系） |
| **音频嵌入** | M2D-CLAP 2025（跨模态，768d）+ OMAR-RQ（声学，768d） |
| **大语言模型** | DeepSeek-V3 / 通义千问 / Gemini（可切换，支持微调模型） |
| **长期记忆** | GraphZep（Hono 微服务，双阶段召回） |
| **联网搜索** | SearxNG 联邦搜索 + Tavily + 智谱 WebSearch |
| **容器化** | Docker Compose（Neo4j + GraphZep + Backend + Frontend） |

---

## 🔬 技术深度

### RAG 混合检索

```
用户查询 → Unified Planner (LLM Structured Output · 一次调用完成意图+路由+HyDE)
                    ↓
        ┌──────────┼──────────┐
        ▼          ▼          ▼
   GraphRAG    VectorKNN   WebSearch
  (Cypher)    (M2D+OMAR)  (SearxNG)
        └──────────┼──────────┘
                   ▼
     加权 RRF 融合 (α·向量 + β·图谱，可前端调节)
                   ▼
     Neo4j 图距离亲和力加权 (用户偏好歌手/流派加分)
                   ▼
     Artist 多样性过滤 (每歌手≤3首，指定歌手豁免)
                   ▼
     MMR 流派多样性重排 (λ=0.7)
                   ▼
              Top-K 结果
```

**关键设计决策**：

- **GraphRAG**：五维标签过滤（genre / scenario / mood / language / region），200+ 中英文别名映射
- **双模型向量**：M2D-CLAP 跨模态语义 + OMAR-RQ 纯声学，三阶段流水线融合
- **HyDE**：Planner 一次调用生成 80-120 词英文声学描述，7 种情形自适应模板
- **RRF 加权**：向量/图谱权重前端可调，避免单一通道主导
- **图距离加权**：用户历史偏好的歌手、流派在 Neo4j 图谱中的距离影响排序
- **歌手豁免**：用户指定"推荐周杰伦的歌"时，`graph_entities` 中的歌手自动豁免多样性限制

### Agent 工作流

```mermaid
stateDiagram-v2
    [*] --> recall_memory: 入口
    recall_memory --> plan_query: GraphZep 粗/精召回
    plan_query --> route_intent: LLM Structured Output

    route_intent --> hybrid_retrieve: search / recommend
    route_intent --> chat_response: general_chat
    route_intent --> play_specific: play_specific_song
    route_intent --> acquire_music: acquire_music
    route_intent --> music_journey: journey

    hybrid_retrieve --> explain_results: 多路融合
    explain_results --> persist_memory: LLM 推荐解释

    chat_response --> persist_memory
    play_specific --> persist_memory
    acquire_music --> persist_memory
    music_journey --> persist_memory

    persist_memory --> [*]: GraphZep 异步写入
```

**统一查询规划器（Unified Query Planner）**

一次 LLM 调用同时完成：意图识别（9 种）→ 检索路由（Graph/Vector/Web）→ 过滤参数提取 → HyDE 声学描述。确定性后处理兜底自动补充 LLM 漏填的字段。

### 记忆系统

| 组件 | 说明 |
|------|------|
| **GraphZep 双阶段** | Stage 1 粗召回 20 条 → Stage 2 精排 5 条（相似度+时间衰减） |
| **GSSC Token 预算** | facts + chat_history 在 3000 token 内动态分配 |
| **Neo4j 偏好图谱** | 每轮对话 LLM 提取用户偏好，fire-and-forget 写入 User 节点 |

### 数据飞轮

用户搜索 → 发现新歌 → 一键"加入本地" → 下载音频/封面/歌词 → LLM 标签提取 + 双模型向量编码 → Neo4j 入库 → 下次检索可命中

---

## 📊 Neo4j 知识图谱

```mermaid
erDiagram
    Song ||--|| Artist : PERFORMED_BY
    Song }o--o{ Genre : HAS_GENRE
    Song }o--o{ Mood : HAS_MOOD
    Song }o--o{ Scenario : FITS_SCENARIO
    Song ||--|| Language : IN_LANGUAGE
    Song ||--|| Region : IN_REGION
    User }o--o{ Song : "LIKED / DISLIKED"
    User }o--o{ Genre : PREFERS_GENRE

    Song {
        string title
        string music_id
        float_arr m2d2_embedding
        float_arr omar_embedding
        string audio_url
    }
    Artist { string name }
    Genre { string name }
    Mood { string name }
    Scenario { string name }
    User { string user_id }
```

**向量索引**：`song_m2d2_index`（768d, cosine）+ `song_omar_index`（768d, cosine）

---

## 🚀 快速开始

### 方式一：Docker Compose（推荐）

```bash
# 1. 复制并配置环境变量
cp .env.example .env
# 编辑 .env 填入 API Key

# 2. 一键启动
docker compose up -d

# 3. 访问
# 前端: http://localhost:3003
# 后端: http://localhost:8501
# Neo4j: http://localhost:7474
```

### 方式二：本地开发（Conda）

```bash
# 环境准备
conda create -n music_agent python=3.11
conda activate music_agent
pip install -r requirements.txt
cd web && npm install && cd ..

# 一键启动所有服务
python startup_all.py

# 或前后端分离开发
python startup_all.py --no-web    # 终端 A：后端
cd web && npm run dev             # 终端 B：前端（热更新）
```

cd web ; npm run dev

> ⚠️ 启动前需先打开 Neo4j Desktop 并启动数据库。

<details>
<summary>手动分步启动</summary>

| 终端 | 命令 | 端口 |
|------|------|------|
| 0 | Neo4j Desktop 启动数据库 | `:7687` |
| 1 | `cd graphzep_service/server && npm run dev` | `:3100` |
| 2 | `python start.py --mode api` | `:8501` |
| 3 | `cd web && npm run dev` | `:3003` |
| 4 | `docker compose -f docker-compose.searxng.yml up -d` | `:8888` |

</details>

<details>
<summary>前置服务安装</summary>

**NeteaseCloudMusicApi**

```bash
git clone --depth 1 https://github.com/NeteaseCloudMusicApiEnhanced/api-enhanced.git NeteaseCloudMusicApi
cd NeteaseCloudMusicApi && npm install && npm start  # :3000
```

**SearxNG 联网搜索**

```bash
docker compose -f docker-compose.searxng.yml up -d  # :8888
```

</details>

---

## 📁 项目结构

```
.
├── agent/                      # LangGraph Agent
│   ├── music_agent.py          # Agent 主入口
│   └── music_graph.py          # StateGraph 工作流定义
│
├── api/server.py               # FastAPI 主服务 + Settings API
├── config/settings.py          # 全局配置（支持运行时修改）
│
├── retrieval/                  # 检索引擎层
│   ├── hybrid_retrieval.py     # 多路检索融合 + RRF + MMR
│   ├── audio_embedder.py       # M2D-CLAP 跨模态编码
│   ├── neo4j_client.py         # Neo4j 连接封装
│   ├── music_journey.py        # 音乐旅程编排器
│   └── user_memory.py          # 用户偏好 Neo4j 记忆
│
├── tools/                      # 工具层
│   ├── graphrag_search.py      # 知识图谱检索（Neo4j Cypher）
│   ├── semantic_search.py      # 向量检索（M2D-CLAP + OMAR）
│   ├── web_search_aggregator.py # 联网搜索聚合
│   └── acquire_music.py        # 数据飞轮（下载入库）
│
├── llms/                       # LLM 接口 + Prompts
├── schemas/                    # Pydantic 数据模型
├── services/                   # 外部服务客户端（GraphZep）
│
├── data/pipeline/              # 数据管线
│   ├── ingest_to_neo4j.py      # Neo4j 入库
│   └── lyrics_analyzer.py      # LLM 歌词标签分析
│
├── web/                        # Next.js 前端
│   ├── components/Settings/    # ⚙️ 运行时设置面板
│   └── components/Navigation/  # 导航、侧边栏
│
├── graphzep_service/           # GraphZep 微服务
├── docker-compose.yml          # Docker 全栈编排
├── Dockerfile                  # 后端镜像
├── web/Dockerfile              # 前端镜像
├── .env.example                # 环境变量模板
├── startup_all.py              # 本地一键启动器
└── requirements.txt            # Python 依赖
```

---

## 🔧 数据管线

首次部署或新增音乐时执行：

```bash
# 1. 下载音频和元数据
python data/pipeline/ncm_pipeline.py

# 2. 歌词标签提取（LLM 自动化）
python data/pipeline/lyrics_analyzer.py

# 3. 入库 Neo4j
python data/pipeline/ingest_to_neo4j.py              # 完整入库
python data/pipeline/ingest_to_neo4j.py --skip-embeddings   # 仅元数据
python data/pipeline/ingest_to_neo4j.py --update-embeddings # 仅补充向量
```

---

## ⚙️ 配置

### 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `OPENAI_BASE_URL` | LLM API 地址 | `https://api.siliconflow.cn/v1` |
| `OPENAI_API_KEY` | LLM API 密钥 | — |
| `MODEL_NAME` | 主推理模型 | `deepseek-ai/DeepSeek-V3` |
| `NEO4J_URI` | Neo4j 连接 | `neo4j://127.0.0.1:7687` |
| `NEO4J_PASSWORD` | Neo4j 密码 | — |
| `TAVILY_API_KEY` | 联网搜索 | 可选 |
| `GOOGLE_API_KEY` | Gemini API | 可选 |

### 运行时设置（前端）

前端设置面板（⚙️ 系统设置）支持实时调整：

| 分类 | 可调参数 |
|------|----------|
| **模型配置** | 主 LLM 提供商/模型、意图分析模型、HyDE 模型、微调模型路径、超时 |
| **检索参数** | 图谱/向量/歌单数量、RRF 权重、图距离加权开关/权重/跳数 |
| **音乐数据** | 本地音乐/MTG/联网获取/模型导出目录 |
| **记忆系统** | 上下文保留轮数、用户 ID |

修改后点击保存即时生效，关闭面板则丢弃未保存修改。支持「↩ 还原默认」。

---

## 🔌 API 端点

### SSE 流式

| 端点 | 说明 |
|------|------|
| `POST /api/recommendations/stream` | 音乐推荐（流式） |
| `POST /api/journey/stream` | 音乐旅程（流式） |

### REST

| 端点 | 说明 |
|------|------|
| `POST /api/search` | 歌曲搜索 |
| `POST /api/acquire-song` | 加入本地曲库 |
| `POST /api/user-event` | 用户行为上报 |
| `GET /api/settings` | 获取当前配置 |
| `POST /api/settings` | 更新配置 |
| `POST /api/settings/reset` | 还原默认配置 |
| `GET /health` | 健康检查 |

---

## 🙏 致谢

本项目初始架构参考自 [imagist13/Muisc-Research](https://github.com/imagist13/Muisc-Research)，在此基础上进行了大规模重构与功能扩展。

| 项目 | 用途 |
|------|------|
| [aexy-io/graphzep](https://github.com/aexy-io/graphzep) | GraphZep 长期记忆 |
| [nttcslab/m2d](https://github.com/nttcslab/m2d) | M2D-CLAP 跨模态模型 |
| [MTG/omar](https://github.com/MTG/omar) | OMAR-RQ 音频模型 |
| [langchain-ai/langgraph](https://github.com/langchain-ai/langgraph) | Agent 编排 |
| [searxng/searxng](https://github.com/searxng/searxng) | 联网元搜索 |

---

## 📄 许可证

MIT License

⚠️ **免责声明**：本项目仅供学习与架构研究，**严禁商业用途**。不提供、不包含也不分发任何受版权保护的音频或歌词资源。音频数据需用户自行通过合法渠道获取。
