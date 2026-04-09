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
  <br/>
  <img src="https://github.com/hgsanyang/SoulTuner-Agent/actions/workflows/ci.yml/badge.svg" alt="CI" />
  <img src="https://img.shields.io/badge/tests-51_passed-brightgreen?logo=pytest" alt="Tests" />
  <img src="https://img.shields.io/badge/code_style-ruff-261230?logo=ruff" alt="Ruff" />
</p>

<p align="center">
  <a href="README.md">中文</a> | <a href="README_EN.md">English</a>
</p>

## 🎯 用自然语言发现音乐，让 AI 真正听懂你

SoulTuner 是一款**本地部署**的 AI 音乐推荐智能体。它不是简单的"搜歌→播放"工具，而是一个能**持续学习你音乐品味**的私人 DJ：

- 🗣️ **用自然语言描述你想听的** — "我今天心情特别差，想一个人静一静"，系统自动识别情绪与场景，推荐契合当下状态的音乐
- 🧠 **越用越懂你** — 每一次点赞、收藏、跳过和对话，都在无声构建你的个性化音乐画像，下次推荐更精准
- 🌐 **本地曲库不够？实时联网补充** — 当本地库无法满足需求时，自动联网搜索最新音乐资讯
- 🗺️ **沉浸式音乐旅程** — 描述一段故事或场景，AI 为你编排一整段有起承转合的音乐旅程
- ♻️ **发现即入库** — 推荐中遇到好歌？一键下载并自动进行声学分析和入库，下次就能被检索到

> 📖 完整功能与交互细节请参阅 [Feature_Walkthrough.md](Feature_Walkthrough.md)
>
> 融合知识图谱（Neo4j）、双模型音频向量（M2D-CLAP + OMAR-RQ）、大语言模型和 GraphZep 长期记忆，通过 LangGraph 编排的多节点 Agent 工作流，实现多路混合检索、加权 RRF 融合、Neo4j 图距离加权、SSE 流式推荐、联网搜索回退、音乐旅程编排和用户行为数据飞轮。

---

## ✨ 核心特性

| 特性 | 说明 |
|------|------|
| 🔀 **Hybrid RAG** | GraphRAG + Semantic Search 并发检索，加权 RRF 融合排序 |
| 🎵 **双模型音频向量** | M2D-CLAP 跨模态语义 × 0.7 + OMAR-RQ 声学特征 × 0.3 |
| 🧠 **长期记忆** | GraphZep 双阶段召回，跨会话保留用户偏好 |
| 📊 **Graph Affinity** | Neo4j 图距离 + 用户画像偏好 Jaccard 双重个性化排序 |
| 🤖 **智能意图识别** | 7 类意图分类，支持 API 大模型 + 本地 Qwen3-4B 双模式 |
| 👤 **用户画像** | 前端可视化画像面板，流派/情绪/场景/语言偏好 → Neo4j + GraphZep 双写 |
| 🌐 **联网搜索回退** | 本地库不足时自动触发 SearxNG 联邦搜索 + LLM 摘要 |
| 🎼 **音乐旅程** | LLM 故事→情绪拆解→逐段检索，SSE 实时推送 |
| ♻️ **数据飞轮** | 用户一键入库：搜索→发现→下载→标签提取→向量编码→Neo4j |
| 📡 **SSE 流式** | 前端实时渲染 thinking → 歌曲卡片 → 推荐理由 |
| 🐳 **Docker 部署** | `docker compose up` 一键启动全栈 |

---

## 🖼️ 功能预览

📺 [**演示视频**](https://www.bilibili.com/video/BV1ZzSDBaEhj/) — 完整功能演示（B站）

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
│  start → GraphZep Recall → Planner (LLM) → Intent Router          │
│                                                                     │
│     ┌─────────┬─────────┬─────────┬──────────┐                     │
│     ▼         ▼         ▼         ▼          ▼                     │
│  search_songs  chat  acquire  gen_reco  journey                    │
│     │                                                               │
│     ▼                                                               │
│  Hybrid Retrieval ──→ LLM Explainer ──→ Pref Extract ──→ GraphZep Write → end │
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
│              Merge & Dedup (平等合并去重)                            │
│                            ▼                                        │
│              Graph Affinity (图距离 + 画像 Jaccard 加分)             │
│                            ▼                                        │
│              Dual-Anchor Rerank (M2D-CLAP + OMAR-RQ)               │
│                            ▼                                        │
│              MMR Multi-dim Diversity (λ=0.7)                       │
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
| **前端** | Next.js 14 + React 18 |
| **Agent** | LangGraph StateGraph（7 类意图路由） |
| **后端** | FastAPI + SSE 流式推送 |
| **图数据库** | Neo4j 5.x（原生向量索引 + 图谱关系 + 用户行为直写） |
| **音频嵌入** | M2D-CLAP 2025（跨模态语义，768d）+ OMAR-RQ（纯声学特征，1024d） |
| **大语言模型** | DeepSeek-V3.2 / Gemini / 豆包（火山引擎）/ 通义千问（API）+ Qwen3-4B（SGLang 本地部署） |
| **长期记忆** | GraphZep 时序记忆（双阶段召回） |
| **联网搜索** | SearxNG 联邦搜索 + Tavily + 智谱 WebSearch |
| **排序算法** | 双锚精排（cosine）+ Graph Affinity（shortestPath + Jaccard）+ MMR |
| **上下文管理** | GSSC Token 预算管线（Gather/Select/Structure/Compress + 异步预压缩缓存） |
| **容器化** | Docker Compose（Neo4j + GraphZep + Backend + Frontend） |

> 📖 完整技术栈与前端工程实现细节请参阅 [Technical_Report.md](Technical_Report.md)

---

## 🔬 技术深度

### RAG 混合检索流水线

```
用户查询 → Planner (LLM)
              ↓  intent_type + retrieval_plan
   ┌──────────┼──────────┐
   ▼          ▼          ▼
GraphRAG   VectorKNN  WebSearch       ← Step 1: 多路并发召回
(Cypher)  (M2D+OMAR) (SearxNG)
   └──────────┼──────────┘
              ▼
  Step 2: 平等合并去重                 ← 替代旧版加权 RRF
              ▼
  Step 2.5: DISLIKES 过滤             ← 排除用户明确不喜欢的歌
              ▼
  Step 3: Artist 多样性初筛            ← 每歌手 ≤ N 首（指定歌手豁免）
              ▼
  Step 4: Graph Affinity              ← 图距离 + Jaccard 偏好加分
              ▼
  Step 5: 双锚精排                     ← M2D-CLAP 语义锚 + OMAR-RQ 声学锚
              ▼
  Step 6: MMR 多维多样性重排           ← genre + mood + theme + scenario
              ▼
  Step 7: FinalCut (≤ 15 首)          ← 安全去重 + 截断
```

**关键设计决策**：

- **GraphRAG**：五维标签过滤（genre / scenario / mood / language / region），200+ 中英文别名映射
- **双模型向量**：M2D-CLAP 跨模态语义 + OMAR-RQ 纯声学，三阶段流水线融合
- **双锚精排**：M2D-CLAP 语义锚（query text → cosine）+ OMAR-RQ 声学锚（候选质心 → cosine），加权合并为最终排序分
- **MMR Jaccard**：利用候选歌的 `{genre, mood, theme, scenario}` 多维标签计算 Jaccard 相似度实现多样性重排

### Agent 工作流

```mermaid
stateDiagram-v2
    [*] --> recall_memory: 入口
    recall_memory --> plan_query: GraphZep 粗/精召回
    plan_query --> route_intent: LLM Structured Output (7 类意图)

    route_intent --> search_songs: graph_search / hybrid_search / vector_search / web_search
    route_intent --> chat_response: general_chat
    route_intent --> acquire_music: acquire_music
    route_intent --> gen_recommendations: recommend_by_favorites

    search_songs --> explain_results: 多路融合检索
    acquire_music --> explain_results
    gen_recommendations --> explain_results

    explain_results --> extract_preferences: LLM 推荐解释
    extract_preferences --> persist_memory: LLM 偏好提取（异步）

    chat_response --> persist_memory: 闲聊无推荐，跳过偏好提取

    persist_memory --> [*]: GraphZep 异步写入
```

> 意图识别支持 API 大模型（DeepSeek-V3.2 等）和本地 Qwen3-4B（SGLang 部署）双模式。本地模式下 HyDE 声学描述由独立模块生成。

> 偏好提取为独立 LangGraph 节点 `extract_preferences`，闲聊意图自动跳过。

### 记忆系统

| 组件 | 说明 |
|------|------|
| **GraphZep 双阶段** | Stage 1 粗召回 → Stage 2 精排（相似度 + 时间衰减） |
| **GSSC Token 预算** | facts + chat_history 动态分配，支持 LLM 摘要压缩 + 异步预压缩缓存 |
| **Neo4j 偏好图谱** | 每轮对话自动提取用户偏好，异步写入 User 节点 |
| **用户画像双写** | 前端画像面板 → 同时写入 Neo4j + GraphZep 长期记忆 |

### 用户画像系统

前端画像面板保存用户偏好（流派/情绪/场景/语言），同时写入 Neo4j User 节点属性和 GraphZep 长期记忆。检索排序时通过 Graph Affinity 读取偏好，计算 Jaccard 相似度为候选歌加分。

### 数据飞轮

用户搜索 → 发现新歌 → 一键"加入本地" → 下载音频/封面/歌词 → LLM 标签提取 + 双模型向量编码 → Neo4j 入库 → 下次检索可命中

### 工程质量

| 维度 | 说明 |
|------|------|
| **CI/CD** | GitHub Actions — 每次 push 自动运行 `ruff` 代码检查 + `pytest` 单元测试 |
| **单元测试** | 51 tests / 5 模块（key 标准化、Token 预算、标签映射、合并去重、Schema 校验）|
| **意图评测** | 55 条手工标注覆盖全部 7 种意图类型，批量测试准确率 **98.2%**（54/55） |
| **Token 追踪** | GSSC 管线内置结构化 Token 消耗报告（Before/After/Savings 对比） |
| **状态持久化** | LangGraph MemorySaver Checkpoint（内存级，可替换为 Sqlite/Postgres） |
| **代码规范** | Ruff 静态检查 + pyproject.toml 统一配置 |

<details>
<summary>意图分类评测详情</summary>

```
评测日期: 2026-04-09
模型: DeepSeek-V3.2 (SiliconFlow)
测试集: 55 条手工标注 (tests/eval/intent_test_queries.json)

Intent Type          Correct   Total   Accuracy
────────────────────────────────────────────────
graph_search              15      15     100.0%
hybrid_search             19      20      95.0%
vector_search              6       6     100.0%
web_search                 4       4     100.0%
general_chat               4       4     100.0%
acquire_music              3       3     100.0%
recommend_by_favorites     3       3     100.0%
────────────────────────────────────────────────
TOTAL                     54      55      98.2%
平均延迟: 11.55s/query（含意图分类 + 实体提取 + 标签推断 + HyDE 声学描述生成，单次 LLM 调用）
```

运行评测：
```bash
python -m tests.eval.evaluate_intent --provider siliconflow
```

</details>

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
    User }o--o{ Song : "LIKES / SAVES / LISTENED_TO"
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
    User {
        string user_id
        string preferred_genres
        string preferred_moods
        string preferred_scenarios
        string preferred_languages
        string profile_free_text
    }
```

**向量索引**：`song_m2d2_index`（768d, cosine）+ `song_omar_index`（1024d, cosine）

---

## 🚀 快速开始

### ⚠️ 前置准备：音乐数据

系统需要本地 MP3 音频文件才能进行向量编码和入库。请将音乐文件放置到以下目录：

```
data/
├── processed_audio/
│   └── audio/          ← 将 MP3 文件放到这里（主音乐目录）
├── online_acquired/    ← 联网获取的音乐会自动存放在此
└── mtg_sample/
    └── audio/          ← MTG 数据集音频（可选）
```

> 📁 音频目录路径可在 `.env` 中通过 `MUSIC_AUDIO_DATA_DIR` 自定义，或在前端设置面板中修改。放入音频后需执行[数据管线](#-数据管线)完成入库。

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
conda activate music_agent; python startup_all.py --no-web    # 终端 A：后端
cd web && npm run dev             # 终端 B：前端（热更新）
```

### 方式三：本地大模型部署（WSL + SGLang）

针对 8GB 显存（如 RTX 4070）设备的优化部署方案，支持本地运行 Qwen3-4B 模型辅助意图识别。

**启动步骤**：

1. **终端A (WSL)**：启动大模型推理引擎

   ```bash
   wsl
   bash /path/to/SoulTuner-Agent/scripts/start_sglang.sh
   ```

2. **终端B (Windows)**：在前端设置面板切换为本地模型
   - 正常运行 `python startup_all.py`
   - 打开系统设置 ⚙️ → **主提供商**选择 `sglang` → 保存

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
│   └── music_graph.py          # StateGraph 工作流（7 意图路由）
│
├── api/                        # FastAPI 接口层
│   ├── server.py               # 主服务 + Settings API
│   └── user_profile.py         # 用户画像 API（GET/POST /api/user-profile）
│
├── config/settings.py          # 全局配置（支持运行时修改）
│
├── retrieval/                  # 检索引擎层
│   ├── hybrid_retrieval.py     # 多路融合 + RRF + Graph Affinity + MMR
│   ├── gssc_context_builder.py # GSSC 上下文管线（Token 预算 + LLM 压缩 + 异步预压缩缓存）
│   ├── audio_embedder.py       # M2D-CLAP 跨模态编码
│   ├── neo4j_client.py         # Neo4j 连接封装
│   ├── music_journey.py        # 音乐旅程编排器
│   └── user_memory.py          # 用户偏好 Neo4j 记忆
│
├── tools/                      # 工具层
│   ├── graphrag_search.py      # 知识图谱检索（Neo4j Cypher，五维标签）
│   ├── semantic_search.py      # 向量检索（M2D-CLAP + OMAR）
│   ├── web_search_aggregator.py # 联网搜索聚合（SearxNG + Tavily）
│   └── acquire_music.py        # 数据飞轮（下载入库）
│
├── llms/                       # LLM 接口 + Prompts
│   ├── prompts.py              # Planner Prompt + 辅助 Prompt
│   └── multi_llm.py            # 多提供商 LLM 工厂（SiliconFlow / Volcengine / Gemini / OpenAI）
│
├── schemas/                    # Pydantic 数据模型
│   └── query_plan.py           # MusicQueryPlan + RetrievalPlan
│
├── services/                   # 外部服务客户端（GraphZep）
│
├── data/pipeline/              # 数据管线
│   ├── ingest_to_neo4j.py      # Neo4j 入库
│   ├── neo4j_schema_v2.py      # 数据集管理工具
│   └── lyrics_analyzer.py      # LLM 歌词标签分析
│
├── web/                        # Next.js 前端
│   ├── components/Settings/    # ⚙️ 运行时设置面板
│   ├── components/Profile/     # 👤 用户画像面板
│   └── components/Navigation/  # 导航、侧边栏
│
├── graphzep_service/           # GraphZep 微服务
├── tests/                      # 测试与评测
│   ├── unit/                   # 单元测试 (51 tests, pytest)
│   │   ├── test_normalize_key.py
│   │   ├── test_gssc_token_budget.py
│   │   ├── test_tag_expansion.py
│   │   ├── test_merge_dedup.py
│   │   └── test_schema_validation.py
│   └── eval/                   # 意图分类评测
│       ├── intent_test_queries.json  # 55 条标注数据
│       └── evaluate_intent.py        # 评测脚本
├── .github/workflows/ci.yml    # GitHub Actions CI
├── docker-compose.yml          # Docker 全栈编排
├── Dockerfile                  # 后端镜像
├── pyproject.toml              # 项目配置 (mypy + ruff + pytest)
├── .env.example                # 环境变量模板
├── startup_all.py              # 本地一键启动器
└── requirements.txt            # Python 依赖
```

---

## 🔧 数据管线

首次部署或新增音乐时执行：

```bash
# 1. 歌词标签提取（LLM 自动化）
python data/pipeline/lyrics_analyzer.py

# 2. 入库 Neo4j
python data/pipeline/ingest_to_neo4j.py              # 完整入库
python data/pipeline/ingest_to_neo4j.py --skip-embeddings   # 仅元数据
python data/pipeline/ingest_to_neo4j.py --update-embeddings # 仅补充向量
```

> 📖 数据集管理 CLI 工具（查看分布 / 验证索引 / 回填标签）详见 [Technical_Report.md](Technical_Report.md)

---

## ⚙️ 配置

### 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `OPENAI_BASE_URL` | LLM API 地址 | `https://api.siliconflow.cn/v1` |
| `OPENAI_API_KEY` | LLM API 密钥 | — |
| `MODEL_NAME` | 主推理模型 | `deepseek-ai/DeepSeek-V3.2` |
| `VOLCENGINE_BASE_URL` | 火山引擎 API 地址 | `https://ark.cn-beijing.volces.com/api/v3` |
| `VOLCENGINE_API_KEY` | 火山引擎 API 密钥 | 可选 |
| `NEO4J_URI` | Neo4j 连接 | `neo4j://127.0.0.1:7687` |
| `NEO4J_PASSWORD` | Neo4j 密码 | — |
| `TAVILY_API_KEY` | 联网搜索 | 可选 |
| `GOOGLE_API_KEY` | Gemini API | 可选 |

> 📖 运行时设置面板（LLM 切换 / 检索参数 / RRF 权重等）及 API 端点文档详见 [Technical_Report.md](Technical_Report.md)

---

## 🙏 致谢

本项目初始架构参考自 [imagist13/Muisc-Research](https://github.com/imagist13/Muisc-Research)，在此基础上进行了大规模重构与功能扩展。

| 项目 | 用途 |
|------|------|
| [aexy-io/graphzep](https://github.com/aexy-io/graphzep) | GraphZep 长期记忆 |
| [nttcslab/m2d](https://github.com/nttcslab/m2d) | M2D-CLAP 跨模态模型 |
| [MTG/omar](https://github.com/MTG/omar) | OMAR-RQ 音频模型 |

---

## 📚 参考文献

1. Niizumi, D. et al. (2025). *M2D-CLAP: Exploring General-purpose Audio-Language Representations Beyond CLAP.*
2. Alonso-Jiménez, P. et al. (2025). *OMAR-RQ: Open Music Audio Representation Model Trained with Multi-Feature Masked Token Prediction.*
3. Rasmussen, P. et al. (2025). *Zep: A Temporal Knowledge Graph Architecture for Agent Memory.*
4. Palumbo, E. et al. (Spotify, 2025). *You Say Search, I Say Recs: A Scalable Agentic Approach to Query Understanding and Exploratory Search.* (RecSys 2025)
5. D'Amico, E. et al. (Spotify, 2025). *Deploying Semantic ID-based Generative Retrieval for Large-Scale Podcast Discovery at Spotify.*
6. Penha, G. et al. (2025). *Semantic IDs for Joint Generative Search and Recommendation.* (RecSys 2025 LBR)
7. Palumbo, E. et al. (2025). *Text2Tracks: Prompt-based Music Recommendation via Generative Retrieval.*
8. Xu, S. et al. (NetEase Cloud Music, 2025). *Climber: Toward Efficient Scaling Laws for Large Recommendation Models.*
9. Wang, S. et al. (2025). *Knowledge Graph Retrieval-Augmented Generation for LLM-based Recommendation.* (ACL 2025)

---

## 📄 许可证

MIT License

⚠️ **免责声明**：本项目仅供学习与架构研究，**严禁商业用途**。不提供、不包含也不分发任何受版权保护的音频或歌词资源。音频数据需用户自行通过合法渠道获取。
