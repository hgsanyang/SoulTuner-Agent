# SoulTuner-Agent 技术架构与原理深度剖析报告

> **版本**: V3.0 | **日期**: 2026-03-31 | **作者**: SoulTuner Team

---

## 目录

1. [引言与业界架构对标](#一-引言与业界架构对标)
2. [全栈技术栈总览](#二-全栈技术栈总览)
3. [前端架构与工程实现](#三-前端架构与工程实现)
4. [后端架构与 API 设计](#四-后端架构与-api-设计)
5. [Agent 编排引擎：LangGraph](#五-agent-编排引擎langgraph)
6. [混合检索管线与排序算法](#六-混合检索管线与排序算法)
7. [底座模型原理剖析](#七-底座模型原理剖析)
8. [长期记忆系统：GraphZep](#八-长期记忆系统graphzep)
9. [用户偏好与行为数据系统](#九-用户偏好与行为数据系统)
10. [数据飞轮与自动化入库](#十-数据飞轮与自动化入库)
11. [上下文管理：GSSC 管线](#十一-上下文管理gssc-管线)
12. [部署与容器化](#十二-部署与容器化)
12.5. [工程质量与评测体系](#十二点五-工程质量与评测体系)
13. [参考文献](#十三-参考文献)

---

## 一、 引言与业界架构对标

### 1.1 行业背景

在千万级用户规模的音乐与音频推荐赛道中，推荐系统正在经历从"传统多阶段管线"向"生成式推荐（Generative Recommendation）"和"Agentic（智能体化）"架构的范式转移。

以业界标杆 **Spotify** 为例，其最新的研究和落地实践全面指向了大模型驱动的探索性搜索与发现：

- **生成式检索与语义 ID (Semantic IDs)**：Spotify 推出了基于 Semantic IDs 的联合生成检索技术 [6]，以及 GLIDE 系统 [5]，将推荐定义为指令跟随任务，打破了传统召回的瓶颈。
- **"You Say Search, I Say Recs" Agentic 架构**：面对用户模糊的探索性查询（Exploratory Search），Spotify 设计了基于 LLM 路由的 Agent 系统，动态分发 Search 与 Recommendation 工具 [4]。
- **大规模推荐模型缩放律 (Scaling Laws)**：网易云音乐的 Climber 研究 [8] 验证了推荐模型遵循类似 LLM 的缩放规律，为工业级系统提供理论基础。

### 1.2 SoulTuner-Agent 的定位

**SoulTuner-Agent** 深刻吸收了上述前沿理念。它不只是一个传统的推荐引擎，更是一个**基于大语言模型的全双工音乐伴游 Agent**。系统采用 LangGraph 进行多节点工作流编排，融合了多模态声学表征（M2D-CLAP, OMAR-RQ）、时序图谱记忆（GraphZep）以及多阶段精排管线，构建了一个具备深层意图理解、长程上下文记忆与声学感知的下一代音乐推荐大脑。

### 1.3 与 Spotify 四阶段管线的对标

| 阶段 | Spotify 实现 | SoulTuner 实现 | 对标状态 |
|------|-------------|---------------|---------|
| **召回** | 协同过滤、内容向量、画像、热门池等 10+ 路同时召回 | GraphRAG + Neo4j Vector + Web Search 三路并发召回 | ✅ 已覆盖 |
| **粗排** | 双塔/DSSM 轻量模型快速打分 | 平等合并去重 + DISLIKES 过滤 + Artist 多样性初筛 | ⚠️ 轻量版 |
| **精排** | DNN/DCN/DeepFM 深度排序 | 双锚精排（M2D-CLAP 语义锚 + OMAR-RQ 声学锚） | ✅ 已覆盖 |
| **重排** | 多样性、去重、策略、Context-aware | Graph Affinity 个性化 + MMR Jaccard 多样性重排 | ✅ 已覆盖 |

---

## 二、 全栈技术栈总览

### 2.1 前端技术栈

| 技术 | 版本 | 用途 |
|------|------|------|
| **Next.js** | 14.2 | React 全栈框架，App Router 路由体系 |
| **React** | 18.3 | UI 组件库，Hooks 驱动 |
| **TypeScript** | 5.5 | 全量静态类型检查 |
| **Framer Motion** | 12.x | 页面转场动画、微交互动效 |
| **Axios** | 1.13 | HTTP 客户端 |
| **CSS-in-JS** | 自定义 theme.ts | Spotify 风格深色主题设计系统 |
| **ESLint** | 8.57 | 代码质量校验 |

### 2.2 后端技术栈

| 技术 | 版本 | 用途 |
|------|------|------|
| **FastAPI** | ≥0.104 | 异步 Web 框架，自动 OpenAPI 文档 |
| **Uvicorn** | ≥0.24 | ASGI 服务器（标准版，含 uvloop） |
| **Pydantic** | ≥2.0 | 数据模型校验、类型安全的结构化输出 |
| **SSE (Server-Sent Events)** | — | 实时流式推送（thinking → song → explanation） |
| **asyncio** | 标准库 | 全异步 I/O（检索引擎并发、GraphZep 写入） |
| **httpx** | — | 异步 HTTP 客户端（SGLang 直连） |

### 2.3 AI/ML 技术栈

| 技术 | 用途 |
|------|------|
| **LangGraph** | Agent 状态机编排框架（StateGraph） |
| **LangChain** | LLM 调用封装、Prompt Template、Structured Output |
| **M2D-CLAP** (2025) | 跨模态声学-语言表征模型，768 维嵌入 |
| **OMAR-RQ** (2025) | 纯听觉声学底座模型，多码本特征提取 |
| **PyTorch** ≥2.2 | 深度学习推理后端（GPU/CPU） |
| **torchaudio** ≥2.2 | 音频波形加载与预处理（16kHz 重采样） |
| **Transformers** ≥4.40 | HuggingFace 模型加载（OMAR-RQ） |
| **sentence-transformers** ≥3.0 | 文本嵌入模型 |
| **NumPy** ≥1.24 | 向量运算（余弦相似度、质心计算） |
| **librosa** ≥0.11 | 音频特征提取（备用） |
| **tiktoken** ≥0.7 | Token 计数与预算管理 |

### 2.4 数据存储与服务

| 技术 | 用途 |
|------|------|
| **Neo4j** 5.x Community | 图数据库：知识图谱 + 原生向量索引 + 用户关系 |
| **GraphZep** (Hono + TypeScript) | 时序知识图谱记忆微服务 |
| **SearxNG** | 联邦搜索引擎（联网搜索回退） |
| **Docker Compose** | 多服务编排（Neo4j + GraphZep + Backend + Frontend） |

### 2.5 大语言模型支持

| 模型 | 模式 | 用途 |
|------|------|------|
| **DeepSeek-V3.2** | API | 主推理、意图分析、推荐解释 |
| **Gemini** | API | 备选推理 |
| **通义千问 (Qwen)** | API | 备选推理 |
| **Qwen3-4B** (微调) | SGLang 本地部署 | 意图分类、实体提取 |
| **BGE-M3** | API/本地 | 文本嵌入（GraphZep 检索） |

---

## 三、 前端架构与工程实现

### 3.1 Next.js 14 App Router 架构

前端基于 **Next.js 14 App Router** 构建，采用文件系统路由：

```
web/app/
├── page.tsx            # 首页（Landing + 场景选择卡片）
├── layout.tsx          # 全局布局（Sidebar + GlobalPlayer）
├── globals.css         # 全局样式（动画关键帧定义）
├── recommendations/    # 智能推荐对话页
├── journey/            # 音乐旅程编排页
├── library/            # 个人音乐库页
├── search/             # 搜索页
├── playlist/           # 歌单详情页
└── api/                # Next.js API Routes（未来扩展）
```

### 3.2 组件化设计

系统采用**分层组件架构**，组件按功能域组织：

| 组件域 | 核心组件 | 职责 |
|--------|---------|------|
| **Content** | `SongCard`, `ResultsDisplay`, `ThinkingIndicator`, `WelcomeScreen` | 歌曲卡片渲染、推荐结果展示、思考动画、场景引导 |
| **Player** | `GlobalPlayer`, `StarryBackground` | 全局吸底播放器、全屏歌词展开、粒子背景 |
| **Navigation** | Sidebar | 侧边导航栏 |
| **Landing** | SceneCards | 首页场景推荐卡片 |
| **Journey** | JourneyPanel | 音乐旅程编排界面 |
| **Profile** | UserProfilePanel | 用户画像可视化面板 |
| **Settings** | SettingsPanel | 运行时配置面板（LLM/检索参数） |
| **Layout** | MainLayout | 响应式布局骨架 |

### 3.3 全局状态管理：React Context

系统使用 **React Context API** 实现跨组件状态共享，避免 prop drilling：

#### PlayerContext（全局播放器状态）

```typescript
interface PlayerContextType {
    currentSong: Song | null;    // 当前播放歌曲
    isPlaying: boolean;          // 播放/暂停状态
    volume: number;              // 音量 0-1
    duration: number;            // 总时长
    currentTime: number;         // 当前播放进度
    playMode: PlayMode;          // 'sequence' | 'random' | 'loop'
    queue: Song[];               // 播放队列
    isExpanded: boolean;         // 全屏歌词展开
    // 操作方法...
    playSong, togglePlay, playNext, playPrev, seek,
    addToQueue, removeFromQueue, addAllToQueue
}
```

核心设计：
- 使用 `useRef<HTMLAudioElement>` 管理全局唯一的 `<audio>` 实例，页面切换不中断播放
- `onended` 事件根据 `playMode` 自动触发下一首（单曲循环 / 顺序 / 随机）
- 播放队列支持动态增减，去重逻辑基于 `title + artist` 唯一键

#### LibraryContext（用户音乐库状态）

```typescript
interface LibraryContextType {
    likedSongs: LikedSong[];     // 点赞歌曲列表
    dislikedSongs: DislikedSong[]; // 不喜欢歌曲列表
    collections: Collection[];    // 自定义歌单集合
    toggleLike, toggleDislike, undoDislike,
    isLiked, isDisliked,
    addCollection, addToCollection, removeFromCollection,
    syncFromBackend,             // 从 Neo4j 后端同步
    toast, showToast             // 全局 Toast 通知
}
```

核心设计：
- **双层存储**：`localStorage`（快速启动）+ 后端 Neo4j（权威数据源）
- **启动时自动同步**：`useEffect → syncFromBackend()` 合并远端与本地数据
- **乐观更新**：用户操作即时反映到 UI，异步上报后端（`sendUserEvent`）
- **矛盾状态自动清理**：标记 dislike 时自动从 likes 列表移除

### 3.4 SSE 流式渲染协议

前端通过 **Fetch API + ReadableStream** 实现 SSE 事件流消费：

```
事件流协议:
  data: {"type": "start"}                        → 开始处理
  data: {"type": "thinking", "message": "..."}   → 思考过程展示
  data: {"type": "song", "song": {...}}          → 逐首歌曲推送
  data: {"type": "response", "text": "..."}      → 推荐解释文本（流式增量）
  data: {"type": "complete"}                     → 处理完成
  data: {"type": "error", "error": "..."}        → 错误信息
```

### 3.5 设计系统 (Design System)

系统采用 **Spotify 深色风格** 的自定义设计系统，通过 `theme.ts` 统一管理设计令牌：

```typescript
export const theme = {
  colors: {
    primary: { accent: '#1db954' },     // Spotify Green
    background: { main: '#000000', card: '#121212' },
    text: { primary: '#ffffff', secondary: '#b3b3b3' },
  },
  gradients: {
    hero: 'linear-gradient(180deg, rgba(83,83,83,0.3), rgba(18,18,18,1))',
  },
  shadows: {
    md: '0 8px 24px rgba(0,0,0,0.5)',   // 卡片阴影
  },
  layout: { sidebarWidth: 260, contentMaxWidth: 1600 },
};
```

- **CSS-in-JS 内联样式**：所有组件使用 `style={{}}` 内联样式，引用 theme 令牌，保证一致性
- **动画系统**：`globals.css` 定义 `@keyframes`（fadeInDown / spin / pulse），Framer Motion 处理页面转场
- **StarryBackground**：CSS Module 实现的粒子星空背景动画

---

## 四、 后端架构与 API 设计

### 4.1 FastAPI 异步服务

后端基于 **FastAPI + Uvicorn** 构建，全链路异步处理：

```python
# 核心 API 端点
POST /api/recommendations/stream  # SSE 流式推荐（主 API）
POST /api/journey/stream          # SSE 音乐旅程流
POST /api/user-event              # 用户行为事件收集
POST /api/search                  # 非流式歌曲搜索
POST /api/acquire-song            # 数据飞轮（按需下载入库）
POST /api/user-profile            # 用户画像保存
POST /api/settings                # 运行时配置更新
GET  /api/liked-songs             # 查询点赞歌曲
GET  /api/disliked-songs          # 查询不喜欢歌曲
DELETE /api/disliked-songs        # 撤销不喜欢
GET  /health                      # 健康检查
```

### 4.2 SSE 流式推送架构

```python
async def stream_recommendations(query, chat_history, ...):
    yield sse_event("start", {"message": "开始处理..."})
    
    # 1. LangGraph Agent 异步执行
    async for step in agent.astream(state):
        if step has recommendations:
            yield sse_event("thinking", {"message": "正在检索..."})
            for song in recommendations:
                yield sse_event("song", {"song": song})
        if step has explanation:
            yield sse_event("response", {"text": chunk})
    
    yield sse_event("complete", {})
```

关键设计：
- **StreamingResponse**：FastAPI 原生支持的 `text/event-stream` 响应
- **背压控制**：使用 `asyncio.Queue` 在 Agent 执行与 SSE 推送之间缓冲
- **超时保护**：每个 SSE 连接设置 180 秒最大存活期

### 4.3 运行时配置热更新

设置面板的修改通过 `POST /api/settings` 即时生效，无需重启：

```python
# 可热更新的参数
settings.llm_default_provider      # LLM 供应商切换
settings.graph_affinity_weight     # Graph Affinity 权重
settings.mmr_lambda                # MMR 多样性参数
settings.dual_anchor_weight_*      # 双锚精排权重
settings.web_search_enabled        # 联网搜索开关
settings.max_songs_per_artist      # 每歌手最多推荐数
```

---

## 五、 Agent 编排引擎：LangGraph

### 5.1 StateGraph 工作流

LangGraph 是系统的主控中枢，通过定义 `MusicAgentState`（TypedDict）精准调度推荐链路：

```
start → recall_graphzep_memory → analyze_intent → route_by_intent
    ├── search_songs → generate_explanation → extract_preferences → persist_to_graphzep → END
    ├── general_chat → persist_to_graphzep → END（闲聊无推荐，跳过偏好提取）
    ├── acquire_online_music → generate_explanation → extract_preferences → persist_to_graphzep → END
    └── generate_recommendations (favorites) → generate_explanation → extract_preferences → persist_to_graphzep → END
```

> **V2 架构升级**：`extract_preferences` 从 `generate_explanation` 中解耦为独立 LangGraph 节点。
> 原本硬编码在 `generate_explanation` 末尾的 ~90 行偏好提取逻辑，现在是独立的 `extract_preferences_node` 方法，
> 通过 `workflow.add_edge("generate_explanation", "extract_preferences")` 串联。
> 闲聊意图（`general_chat`）直接跳到 `persist_to_graphzep`，因为没有推荐结果可供提取偏好。

### 5.2 双模 Planner 架构 (Dual-Planner)

针对不同 LLM 部署场景设计了两套 Planner Prompt：

#### 方案 A：API 大模型融合版 (`UNIFIED_MUSIC_QUERY_PLANNER_PROMPT`)

适用于 DeepSeek / Gemini / GPT 等云端大模型。**单次 LLM 调用**并发完成：
- 意图分类（7 类）
- 实体提取（歌手名、歌曲名，含中外文别名）
- 五维标签过滤（genre / mood / scenario / language / region）
- **内联 HyDE**：直接生成英文声学描述供 M2D-CLAP 编码

#### 方案 B：本地小模型精简版 (`LOCAL_PLANNER_PROMPT`)

适用于 SGLang 部署的 Qwen3-4B。以 `/no_think` 前缀关闭思维链：
- 仅输出意图分类 + 实体提取 + 标签
- HyDE 声学描述由下游 `HYDE_ACOUSTIC_GENERATOR_PROMPT` 独立生成
- SGLang 直连绕过 LangChain：通过 `httpx` 直接构造 HTTP 请求体，确保 `chat_template_kwargs: {enable_thinking: false}` 稳定传递

### 5.3 7 类意图路由体系

| 意图 | 触发条件 | 路由目标 |
|------|---------|---------|
| `graph_search` | 干巴巴的标签组合（"周杰伦情歌"） | search_songs |
| `hybrid_search` | 标签 + 主观体验描述（"重低音砸耳朵"） | search_songs |
| `vector_search` | 纯情绪/氛围（"我心情很差"） | search_songs |
| `web_search` | 时效性内容（"最近流行什么新歌"） | search_songs |
| `general_chat` | 闲聊（"你好"） | general_chat |
| `acquire_music` | 确认下载（"帮我下载"） | acquire_online_music |
| `recommend_by_favorites` | 查看收藏（"我喜欢的歌"） | generate_recommendations |

**判断规则优先级**：
1. 去掉标签词后剩余内容 ≤ 3 字 → `graph_search`
2. 有标签词 + 体验性描述（信号词："感觉""那种""让我""沸腾"） → `hybrid_search`
3. 有实体 + 声学描述 → `hybrid_search`
4. 纯情绪无实体无标签 → `vector_search`

---

## 六、 混合检索管线与排序算法

### 6.1 七阶段精排管线 (V3)

```
用户查询 → Planner (意图 + 检索规划)
              ↓
  ┌──────────┼──────────┐
  ▼          ▼          ▼
GraphRAG   VectorKNN  WebSearch     ← Step 1: 多路并发召回
(Cypher)  (M2D-CLAP) (SearxNG)
  └──────────┼──────────┘
             ▼
  Step 2: 平等合并去重               ← 替代旧版加权 RRF
             ▼
  Step 2.5: DISLIKES 过滤            ← 排除用户明确不喜欢的歌
             ▼
  Step 3: Artist 多样性初筛           ← 每歌手 ≤ N 首（指定歌手豁免）
             ▼
  Step 4: Graph Affinity             ← 图距离 + Jaccard 偏好加分
             ▼
  Step 5: 双锚精排                    ← M2D-CLAP + OMAR-RQ
             ▼
  Step 6: MMR 多维多样性重排          ← genre+mood+theme+scenario
             ▼
  Step 7: FinalCut (≤ 15 首)         ← 安全去重 + 截断
```

### 6.2 核心算法公式

#### 6.2.1 余弦相似度 (Cosine Similarity)

用于双锚精排的语义匹配和声学匹配：

```
cos(A, B) = (A · B) / (‖A‖ × ‖B‖)
```

其中 A、B 为嵌入向量（M2D-CLAP 768 维 / OMAR-RQ 1024 维）。

#### 6.2.2 双锚精排评分 (Dual-Anchor Reranking)

```
Score_final = w_semantic × cos(song_m2d, query_text_emb)
            + w_acoustic × cos(song_omar, omar_centroid)
```

- `w_semantic = 0.7`（默认）：M2D-CLAP 语义锚权重
- `w_acoustic = 0.3`（默认）：OMAR-RQ 声学锚权重
- `query_text_emb`：HyDE 生成的英文声学描述经 M2D-CLAP 文本编码器编码
- `omar_centroid`：所有候选歌曲 OMAR 嵌入的算术均值（质心）
- 当歌曲缺少 OMAR 嵌入时，自动退回纯语义排序（`Score = semantic_score`）

#### 6.2.3 Graph Affinity（图距离亲和力）

```
affinity_distance(u, s) = 1 / (1 + d)
```

其中 `d` 为 Neo4j `shortestPath` 计算的用户 u 到歌曲 s 的图距离（跳数）。

最终融合：
```
Score = (1 - α) × Score_base + α × (affinity_distance + pref_boost)
```

`α = graph_affinity_weight`（默认 0.3）

#### 6.2.4 四维 Jaccard 偏好加分

```
weighted_jaccard = 0.30 × J(user_genres, song_genres)
                 + 0.30 × J(user_moods, song_moods)
                 + 0.25 × J(user_scenarios, song_scenarios)
                 + 0.15 × J(user_themes, song_themes)

pref_boost = 0.3 × weighted_jaccard
```

其中 Jaccard 相似度：
```
J(A, B) = |A ∩ B| / |A ∪ B|
```

用户偏好从 Neo4j 缓存读取（首次加载后常驻内存），包含从 LIKES / LISTENED_TO 关系反推的流派、情绪、场景、主题标签集合。

#### 6.2.5 MMR 多维多样性重排 (Maximal Marginal Relevance)

```
MMR_score(c) = λ × relevance(c) - (1 - λ) × max_{s ∈ Selected} J(tags(c), tags(s))
```

- `λ = 0.7`（默认，偏相关性）
- `tags(c)` = 候选歌曲 c 的多维标签并集 `{genre ∪ mood ∪ theme ∪ scenario}`
- 贪心迭代：每轮从候选池中选出 MMR 分数最高的歌曲加入已选集合
- 打破传统的只依 genre 去重的桎梏，实现多维度多样性保障

#### 6.2.6 时间衰减评分 (Temporal Decay)

用于用户收藏歌曲排序：

```
decayed_score = weight / (1.0 + 0.01 × days_since_action)
```

最近的交互权重更高，越旧的点赞/收藏权重越低。

#### 6.2.7 Key 标准化去重

```python
def _normalize_key(title, artist):
    # 1. NFKC Unicode 标准化（全角→半角）
    # 2. 去除所有标点和空格
    # 3. 全部转小写
    return normalized_title + "_" + normalized_artist
```

保证同一首歌即使来自不同引擎（全角/半角、标点差异）也能正确去重。

### 6.3 三路召回引擎详解

#### GraphRAG（Neo4j Cypher）

- **五维标签过滤**：genre / scenario / mood / language / region
- **200+ 中英文别名映射**：`GENRE_TAG_MAP` 支持"摇滚 → rock / 经典摇滚 / classic-rock"等
- **PERFORMED_BY 关系遍历**：歌手名自动匹配 `Artist` 节点
- **DISLIKES 排除**：查询时自动过滤 `(u)-[:DISLIKES]->(s)` 的歌曲

#### 语义向量检索（Neo4j Native Vector Index）

- **M2D-CLAP 768 维向量**：通过 `db.index.vector.queryNodes` 执行 KNN 检索
- **HyDE 虚拟乐评**：用户查询 → LLM 生成英文声学描述 → M2D-CLAP text encoder 编码 → 向量检索
- **相似度阈值**：结果按 cosine 距离排序，仅返回 Top-K

#### 联网搜索（SearxNG 联邦搜索）

- **触发条件**：用户明确要求联网 / 本地库无结果 / 时效性查询
- **多源聚合**：SearxNG 同时查询 Google / Bing / DuckDuckGo 等搜索引擎
- **LLM 摘要**：将网页碎片交由 LLM 生成结构化音乐资讯摘要

---

## 七、 底座模型原理剖析

### 7.1 跨模态音频表征：M2D-CLAP

> **参考**: Niizumi, D. et al. (2025). *M2D-CLAP: Exploring General-purpose Audio-Language Representations Beyond CLAP* [1]

#### 痛点

传统 CLAP 模型为追求图文/声文对齐，牺牲了音频本身通用声学特征的解析力。

#### 底层原理

M2D-CLAP 创新性地将**掩码自监督学习（Masked Modeling Duo, M2D）**的通用特征提取能力，与**对比学习语言对齐（CLAP）**结合。模型在预训练阶段同时优化两个目标：
1. **M2D 目标**：对音频 Mel spectrogram 进行随机掩码，通过在线编码器和目标编码器的交叉预测恢复缺失的时频块，学习通用音频表征
2. **CLAP 目标**：将音频嵌入与对应的文本描述嵌入拉近（对比损失），实现跨模态对齐

#### 项目中的应用

- **数据飞轮入库**：新歌音频 → M2D-CLAP audio encoder → 768 维向量 → 写入 Neo4j `m2d2_embedding` 属性
- **HyDE 文搜音**：LLM 生成英文声学描述 → M2D-CLAP text encoder → 768 维向量 → cosine KNN 检索
- **双锚精排语义锚**：`cos(song_m2d_emb, query_text_emb)` 计算语义匹配度

#### 优势

- 在 ESC-50、AudioSet、VGGSound 等跨领域 benchmark 上达到 SOTA
- 文本编码器与音频编码器共享嵌入空间，支持零样本跨模态检索
- 适合音乐推荐场景的"主观听感匹配"

### 7.2 纯听觉声学底座：OMAR-RQ

> **参考**: Alonso-Jiménez, P. et al. (2025). *OMAR-RQ: Open Music Audio Representation Model Trained with Multi-Feature Masked Token Prediction* [2]

#### 原理

OMAR-RQ 使用了 **33 万小时**音乐数据与**多特征掩码 Token 预测**训练。它完全摒弃文本依赖，纯粹解析音乐底层的乐理结构：
- 输入：音频波形（16kHz）
- 对波形进行多码本（Multi-Codebook）量化，得到离散音频 token
- 随机掩码部分 token，训练模型预测被掩码的 token
- 预测目标不仅包含波形重建，还包含和弦、节拍、配器、人声基频等多维音乐特征

#### 项目中的应用 —— 声学锚点

仅靠 M2D-CLAP 的跨模态语义锚容易让系统推荐歌名、歌词相似但实际听感不搭的歌曲。OMAR-RQ 作为**声学锚点**：
1. 计算所有候选歌曲 OMAR 嵌入的**质心（centroid）**
2. 每首候选歌曲与质心的 cosine 距离作为声学得分
3. 声学得分与语义得分加权合并，同时保证"文字描述相关"与"实际听感一致"

### 7.3 模型推理工程

两个模型均采用**懒加载 + 全局单例缓存**模式：

```python
_M2D2_MODEL = None   # 首次调用才加载 (~600MB 权重)
_OMAR_MODEL = None   # 首次调用才加载

def get_m2d2_model():
    global _M2D2_MODEL
    if _M2D2_MODEL is None:
        _M2D2_MODEL = PortableM2D(checkpoint_path)  # 加载权重
    return _M2D2_MODEL
```

- 支持 GPU（CUDA）与 CPU 推理自动检测
- 音频统一重采样至 16kHz 单声道
- 嵌入维度：M2D-CLAP 768 维，OMAR-RQ 1024 维（multicodebook 采用 Large backbone）

---

## 八、 长期记忆系统：GraphZep

> **参考**: Rasmussen, P. et al. (2025). *Zep: A Temporal Knowledge Graph Architecture for Agent Memory* [3]

### 8.1 底层机制

GraphZep 是一个基于 **Hono (TypeScript)** 构建的微服务，部署在 Neo4j 之上。它建立了一个**时序感知（Temporal Aware）**的记忆图谱：

- **Episodic Subgraph**：存储对话"事件"的时序序列
- **Semantic Entity Subgraph**：从对话中抽取的实体和关系
- **异步 LLM 实体抽取**：消息送入后，后台异步调用 LLM 抽取 `(用户)-[LIKES]→(歌曲)` 等关系

### 8.2 双轨分离写入模式

音乐推荐极度依赖用户的演化偏好。传统的同步写入 LLM 提取会造成严重的服务卡顿。本系统设计了**双轨分离**：

| 轨道 | 写入目标 | 延迟 | 用途 |
|------|---------|------|------|
| **瞬态即时轨** | Neo4j 图关系 | < 100ms | 用户点赞/收藏/跳过等强信号 → 即时影响下次推荐 |
| **异步沉淀轨** | GraphZep 微服务 | Fire-and-Forget | 对话文本和行为事件 → 后台 LLM 抽取 → 长期记忆 |

### 8.3 GraphZep HTTP 客户端

```python
class GraphZepClient:
    async def add_messages(user_msg, bot_resp)  # 对话写入
    async def add_user_event(event_description)  # 行为事件写入
    async def search_facts(query, max_facts=8)   # 语义检索相关事实
    async def get_memory(recent_messages)         # 基于上下文的记忆检索
    async def healthcheck()                        # 健康检查
```

### 8.4 记忆召回策略

- **双阶段召回**：Stage 1 粗召回 20 条事实 → Stage 2 精排 5 条（相似度 + 时间衰减）
- **混合检索**：支持 `semantic` / `keyword` / `hybrid` / `mmr` 四种检索模式
- **GSSC Token 预算控制**：记忆注入 Prompt 前经过 GSSC 截断，不超过 3000 token

---

## 九、 用户偏好与行为数据系统

### 9.1 UserMemoryManager（Neo4j 直写）

`UserMemoryManager` 类封装了所有用户-歌曲关系的 CRUD 操作：

| 方法 | Neo4j 关系 | 语义 |
|------|-----------|------|
| `record_liked_song` | `(u)-[:LIKES {weight: 1.0}]->(s)` | 点赞（显式正向） |
| `record_saved_song` | `(u)-[:SAVES {weight: 0.8}]->(s)` | 收藏（组织性信号） |
| `record_listened_song` | `(u)-[:LISTENED_TO {play_count, total_duration}]->(s)` | 完整播放 |
| `record_skipped` | `(u)-[:SKIPPED {skip_count}]->(s)` | 跳过（弱负向） |
| `record_dislike` | `(u)-[:DISLIKES]->(s)` | 明确不喜欢（自动清理 LIKES/SAVES） |
| `remove_like` | 删除 `LIKES` 关系 | 取消点赞 |
| `remove_save` | 删除 `SAVES` 关系 | 取消收藏 |

### 9.2 智能歌曲匹配策略

为避免用户行为创建裸副本 Song 节点（没有 embedding 的空壳节点），采用四级匹配策略：

```
策略1: 精确匹配 title + artist 属性
策略2: 匹配 title + PERFORMED_BY Artist 节点（CONTAINS 模糊匹配）
策略3: 仅匹配 title（优先选有 m2d2_embedding 的节点）
策略4: 以上都找不到 → MERGE 新节点 + 建立 PERFORMED_BY 关系
```

### 9.3 语义偏好持久化

LLM 从对话中提取的偏好标签（`MUSIC_PREFERENCE_EXTRACTOR_PROMPT`）写入 Neo4j User 节点属性：

```
User 节点属性:
  - add_genres: ["folk", "indie"]       ← 列表型：追加合并去重
  - avoid_genres: ["metal"]
  - add_artists: ["Taylor Swift"]
  - avoid_artists: []
  - mood_tendency: "治愈"               ← 字符串型：覆盖
  - activity_contexts: ["通勤", "学习"]
  - language_preference: "Chinese"
  - preferences_updated_at: timestamp
```

### 9.4 "Seeds + Discoveries" 收藏推荐策略

当用户查看"我喜欢的歌"时（`recommend_by_favorites`）:

```
Tier 1: Seeds — 从 Neo4j 直接读取可播放的收藏歌曲（最多 N 首）
Tier 2: Discoveries — 基于种子歌曲标签 + 用户画像 + GraphZep 记忆，
         构建偏好文本（零 LLM 调用），通过向量检索发现新歌
→ 合并 Seeds + Discoveries，去重后返回
```

---

## 十、 数据飞轮与自动化入库

### 10.1 Data Flywheel V2 管线

```
新音频文件 → 16kHz 重采样
           → M2D-CLAP 编码 → 768 维向量
           → OMAR-RQ 编码 → 1024 维向量
           → LLM Zero-shot 标签推断 (流派/环境/BPM/情绪)
           → Neo4j MERGE (Song 节点 + embedding + 标签关系)
```

### 10.2 按需数据飞轮

用户在推荐结果中点击"加入本地"按钮触发：

```
前端 acquireSong() → POST /api/acquire-song
→ 网易云 API 搜索 + 下载 (音频/封面/歌词)
→ 写入本地文件系统 (processed_audio/)
→ 自动触发 DataFlywheel 入库管线
→ Neo4j 新 Song 节点立即可被检索
```

---

## 十一、 上下文管理：GSSC 管线

### 11.1 设计动机

随着用户交互积累，GraphZep 与多路召回带来的 Context 极易突破 Token 上限（2000-8000），导致：

- LLM 注意力稀释
- 关键信息被截断
- 推理成本飙升

### 11.2 四阶段处理（V2 升级版）

```
Stage 1: Gather   — 收集所有上下文源（GraphZep / 对话历史 / 检索结果）
Stage 2: Select   — 按优先级排序（UserInput > GraphZep > ChatHistory > Retrieval）
Stage 3: Structure — 分配 Token 预算（min_tokens 保底 + 剩余按优先级分配）
Stage 4: Compress  — 智能压缩（V2 升级）
    ├── chat_history 超出分配预算 1.5 倍 → LLM 摘要压缩（CONTEXT_COMPRESSOR_PROMPT）
    └── 其他情况 / LLM 压缩失败 → 按行截断（V1 兜底逻辑）
```

**Stage 4 V2 升级说明**：

借鉴 Claude Code `compact.ts` 的 Agent 压缩模式，当 `chat_history` 的 token 数远超分配预算（超过 1.5 倍）时，调用轻量 LLM（意图分析专用小模型，如 Qwen3-4B）生成对话摘要，替代硬截断。

摘要提示词 `CONTEXT_COMPRESSOR_PROMPT` 指导 LLM 保留关键信息：
1. 用户音乐偏好（喜欢/不喜欢的流派、歌手）
2. 已推荐过的歌曲列表（避免重复推荐）
3. 当前对话的主题意图走向
4. 用户场景上下文

压缩失败时自动 fall back 到 V1 的按行截断逻辑，确保稳定性。`build_context` 函数已升级为 `async`，调用点均已同步更新为 `await`。

### 11.3 Token 估算

```python
def estimate_tokens(text):
    chinese_chars = count_of('\\u4e00'-'\\u9fff')
    remaining = len(text) - chinese_chars
    return int(chinese_chars * 1.5 + remaining * 0.4)
```

中英混合文本的保守估算，避免依赖 tiktoken 的模型绑定。

---

## 十二、 部署与容器化

### 12.1 Docker Compose 多服务编排

```yaml
services:
  neo4j:        # 图数据库 (http://localhost:7474, bolt://localhost:7687)
  graphzep:     # GraphZep 记忆微服务 (:3100)
  backend:      # FastAPI 后端 (:8501)
  frontend:     # Next.js 前端 (:3003)
```

- 服务间通过 Docker 网络通信（`bolt://neo4j:7687`）
- 健康检查链：Neo4j → GraphZep → Backend → Frontend
- 音乐数据通过 Volume 挂载：`${MUSIC_DATA_PATH:-./data}:/app/data`

### 12.2 本地开发部署

```bash
# 一键启动（推荐）
python startup_all.py         # 启动 Backend + Frontend
python startup_all.py --no-web  # 仅启动 Backend

# 手动启动
cd web && npm run dev          # 前端 :3003
uvicorn api.server:app --port 8501  # 后端 :8501
```

---

## 十二点五、 工程质量与评测体系

### 12.5.1 CI / 测试基础设施

| 维度 | 实现 |
|------|------|
| **项目配置** | `pyproject.toml` 统一管理 mypy / ruff / pytest |
| **CI/CD** | GitHub Actions（`.github/workflows/ci.yml`）：每次 push 自动运行 `ruff check` + `pytest tests/unit/` |
| **单元测试** | 51 tests / 5 模块，覆盖核心检索逻辑（`_normalize_key` 去重、GSSC Token 预算、标签扩展映射、RRF 合并去重、Pydantic Schema 校验） |
| **代码规范** | Ruff 静态检查（E/F/W 规则），CI 自动拦截不规范代码 |

### 12.5.2 意图分类评测

构建了 55 条手工标注的评测数据集（`tests/eval/intent_test_queries.json`），覆盖全部 7 种意图类型，每条标注对应 Prompt 中的优先级规则编号。

**评测脚本**：`python -m tests.eval.evaluate_intent --provider siliconflow`

```
评测日期: 2026-04-09
模型: DeepSeek-V3.2 (SiliconFlow)

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

**唯一错误分析**："安静的日语歌" — LLM 将 "安静" 映射为情绪标签走了 graph_search，但"安静"描述的是声学氛围（低动态、轻柔音色），不属于 `graph_mood_filter` 标准值，应走 hybrid_search 让向量引擎精排。此 case 体现了 Prompt Engineering 中"离散标签 vs 连续声学特征"的根本张力。

### 12.5.3 Token 消耗追踪

GSSC 管线内置结构化 Token 追踪报告（`_track_token_savings()`），每次 Compress 阶段自动输出 Before/After/Savings 对比：

```
[GSSC Token Report]
  Source                 Before   After      Saved
  ──────────────────────────────────────────────────
  graphzep_facts            120     120      0
  chat_history             2400     350   2050 (85%)
  retrieval_context         800     600    200 (25%)
  ──────────────────────────────────────────────────
  TOTAL                    3320    1070   2250 (68%)
  Budget: 3000
```

### 12.5.4 LangGraph MemorySaver Checkpoint

在 `_build_graph()` 编译阶段注入 `MemorySaver` checkpointer，支持同一 `thread_id` 的对话状态持久化：

- 内存级实现（进程重启后丢失），可替换为 `SqliteSaver` / `PostgresSaver`
- `enable_checkpoint=True` 控制开关，不可用时优雅降级为无状态模式
- `ainvoke` 时通过 `config["configurable"]["thread_id"]` 关联会话

---

## 十三、 参考文献

1. Niizumi, D. et al. (2025). *M2D-CLAP: Exploring General-purpose Audio-Language Representations Beyond CLAP.*
2. Alonso-Jiménez, P. et al. (2025). *OMAR-RQ: Open Music Audio Representation Model Trained with Multi-Feature Masked Token Prediction.* GitHub: `mtg/omar-rq`
3. Rasmussen, P. et al. (2025). *Zep: A Temporal Knowledge Graph Architecture for Agent Memory.*
4. Palumbo, E. et al. (Spotify, 2025). *You Say Search, I Say Recs: A Scalable Agentic Approach to Query Understanding and Exploratory Search.* (RecSys 2025)
5. D'Amico, E. et al. (Spotify, 2025). *Deploying Semantic ID-based Generative Retrieval for Large-Scale Podcast Discovery at Spotify.*
6. Penha, G. et al. (2025). *Semantic IDs for Joint Generative Search and Recommendation.* (RecSys 2025 LBR)
7. Palumbo, E. et al. (2025). *Text2Tracks: Prompt-based Music Recommendation via Generative Retrieval.*
8. Xu, S. et al. (NetEase Cloud Music, 2025). *Climber: Toward Efficient Scaling Laws for Large Recommendation Models.*
9. Wang, S. et al. (2025). *Knowledge Graph Retrieval-Augmented Generation for LLM-based Recommendation.* (ACL 2025)

---

> *SoulTuner-Agent Technical Report V3.0 — Generated on 2026-03-31*
