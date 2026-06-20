# SoulTuner-Agent 综合审视报告与优化规划

> 评审日期: 2026-06-18 ｜ 评审范围: 全栈架构 / 推荐算法 / 意图识别 / Harness 工程 / 部署 / 商业化就绪度
> 方法: 通读 ~13K 行 Python 核心 + 前端结构 + 三份设计文档 + 联网核对 2025–2026 业界 SOTA + 结合"满帮2"意图识别竞赛复盘

---

## 0. 执行摘要（TL;DR）

1. **这是一个完成度相当高、设计思路紧跟前沿的项目。** 三路召回 + 三锚精排 + MMR 多样性 + 图亲和 + Thompson 探索的管线，对标 Spotify 四阶段是站得住的；GraphZep 时序记忆、双轨写入、GSSC 上下文预算、双模 Planner（云/本地）都是工业级的好品味。**不要推倒重来。**

2. **真正的瓶颈不在"识别准不准"，而在"表示够不够"。** 当前意图层把用户语言压进 **7 个意图 + 5 个固定词表标签**（genre/mood/scenario/language/region）。这一点同时被三方独立印证：(a) 你自己 eval 里唯一的错例"安静的日语歌"；(b) Deezer 2026-02 论文《Beyond Musical Descriptors》；(c) 我在满帮2 的终局教训。**这是意图识别最大的优化空间，且方向明确。** 你已经有了正确的"逃生口"——HyDE 声学描述（自由文本→连续空间），把它从"附属字段"升级为"一等表示"即可。

3. **架构不臃肿，但有明确的脂肪可减。** 死依赖 7 个（streamlit / llama-index 全家桶 / chromadb / spotipy / mcp / pandas / gdown，全项目零 import）；死代码（`_dual_anchor_rerank` 与活跃的 `_tri_anchor_rerank` 并存）；`analyze_intent` 单函数 ~370 行、4 条按厂商分叉的 httpx 旁路逻辑，而 `multi_llm.py` 本应已抽象掉它们。

4. **部署是"可推广"的最大阻力，且有一个法律级阻断点：** `docker-compose.yml` 用的是 `neo4j:2026-enterprise` + eval 许可证（仅限非商用评估），而技术报告写的是 "Community"。4 容器 + 强制 GPU + ~1.6GB 模型权重，对"让别人一键跑起来"太重。

5. **SFT/RL 并非"拿不到反馈数据"——你其实已经在采集隐式反馈**（likes/saves/skips/dislikes 写进 Neo4j + user-event）。最高 ROI 的不是 RL，而是**蒸馏飞轮**：用云端大模型的 Planner 输出当 SFT 数据，喂给本地 Qwen3-4B，直接抹平"本地/云端"质量差并把意图延迟从 ~11s 砍到亚秒级。

6. **离工业产品最大的两道坎是非技术的：** 音乐版权（`.ncm`/下载链路无法商用）和评测口径（当前 eval 测的是"分类对不对"，不是"推荐好不好"——满帮2 的核心教训：测结果，别测特性）。

7. **隐私体检通过。** `.env` 已 gitignore、历史从未提交、无密钥文件入库。唯一待办：compose 里硬编码的 `neo4j/12345678` 默认口令参数化。

**健康度评分：**

| 维度 | 评分 | 一句话 |
|---|---|---|
| 架构设计 | 8.5 / 10 | 分层清晰、编排合理，前沿对标到位 |
| 推荐/精排算法 | 8 / 10 | 多阶段管线扎实，超出一般个人项目 |
| **意图识别（表示层）** | **6 / 10** | 路由准但表达力受限于固定槽——**最大杠杆** |
| Harness / 工程质量 | 7.5 / 10 | CI + 单测 + 意图 eval 齐全，但 eval 偏"测特性" |
| 评测体系（结果导向） | 5 / 10 | 缺"推荐是否满足用户"的离线/在线闭环 |
| 部署 / 可推广性 | 5 / 10 | 4 容器 + GPU + 企业版许可，门槛偏高 |
| 侧端就绪度 | 4 / 10 | 重模型在服务端；端侧路径清晰但未动工 |
| 商业化就绪度 | 3 / 10 | 版权 + 多用户 + 成本/延迟是硬门槛 |

---

## 1. 架构总览（我实际看到的）

```
前端 Next.js 14 (App Router, SSE 流式)  ── /recommendations /journey /library /search /playlist
        │  POST /api/recommendations/stream (text/event-stream)
        ▼
FastAPI (api/server.py, 1457 行)  ── SSE: start→thinking→song→response→complete
        ▼
LangGraph 编排 (agent/music_graph.py, 2117 行)
  recall_graphzep_memory → analyze_intent → route_by_intent
     ├─ search_songs ─┐
     ├─ web_fallback  ├→ generate_explanation → extract_preferences → persist_to_graphzep → END
     ├─ acquire_music─┤
     └─ generate_recs─┘   (general_chat 直达 persist)
        ▼
混合检索 (retrieval/hybrid_retrieval.py, 1597 行)
  3 路召回(GraphRAG/Neo4j-Vector/SearxNG) → 合并去重 → DISLIKES 过滤
  → Artist 多样性 → Graph Affinity 粗排+TS探索 → 三锚精排(M2D-CLAP/OMAR-RQ/个性化) → MMR → FinalCut
        ▼
存储: Neo4j(图+向量+关系) ｜ GraphZep(TS 微服务, 时序记忆) ｜ SearxNG(联邦搜索)
底座: M2D-CLAP(768d 跨模态) + OMAR-RQ(1024d 纯声学) 懒加载单例
```

**亮点（保持）：**
- 双轨记忆写入（瞬态 Neo4j <100ms / 异步 GraphZep fire-and-forget）——正确处理了"演化偏好"的延迟矛盾。
- GSSC 四阶段上下文预算 + LLM 摘要压缩（借鉴 Claude Code compact），有 Token 追踪报告。
- 双模 Planner（API 融合版单次调用 / 本地 Qwen3-4B 精简版），KV Prefix Cache 拆分 system/human。
- `with_structured_output` + Pydantic，避免手撕正则解析。

---

## 2. ⭐ 核心论断：意图识别的真正瓶颈是"表示"，不是"精度"

这是全报告最重要的一节，也是你问"意图识别是否还有很大优化空间"的直接答案。

### 2.1 现状

`analyze_intent`（`agent/music_graph.py:268`）让 LLM 输出 `MusicQueryPlan`：
- **意图**：7 选 1（graph/hybrid/vector/web/chat/acquire/favorites）
- **硬标签**：5 个固定词表槽 —— genre(12 值) / scenario(13 值) / mood(15 值) / language(6 值) / region(6 值)
- **软表示**：`vector_acoustic_query`（自由英文声学描述，HyDE）→ M2D-CLAP 连续空间

### 2.2 三方印证：固定词表槽会"自信地把话听窄"

| 证据来源 | 说的是同一件事 |
|---|---|
| **你的 eval（唯一错例）** | "安静的日语歌" → LLM 把"安静"塞进 `mood` 走 graph_search；但"安静"是声学质感（低动态/轻柔音色），不在 15 个 mood 标准值里。技术报告自己点破："离散标签 vs 连续声学特征的根本张力"。 |
| **Deezer 2026-02《Beyond Musical Descriptors》** | 现有系统只抽"音乐描述符"（流派/速度/情绪），抽不到 **preference-bearing intent**——用户**为什么**要这首歌（目标/状态/轨迹）。固定 schema 会漏掉自然语言里的偏好信号。 |
| **我的满帮2 终局复盘** | 泛化瓶颈在**表示语言的表达力**，不在识别精度。把生成空间硬塞进固定 N 轴 DSL，根本表示不了"有状态/事件/自引用/行为元"的意图。结论：**LLM→富 IR ≫ LLM→固定槽**。 |

**关键认知：** 98.2% 的意图分类准确率衡量的是"路由器选了作者会选的标签"，**不衡量"用户最后有没有得到对的歌"**。当需求落在 5 个词表之外（"想要能让我从 emo 里走出来的歌"、"像周五下班那种松一口气的感觉"、"别太吵但要有劲"），系统不是识别错，而是**根本没有那个格子可放**，于是被迫四舍五入到最近的错误格子。

### 2.3 你已经做对了一半

`vector_acoustic_query`（HyDE）正是"自由文本→连续空间"的正确直觉——它绕开了固定词表。**问题只是它现在是配角**（hybrid/vector 才填，graph 不填，且被当成"声学描述"而非"意图表示"）。

### 2.4 建议方向（不是推倒，是升级表示）

把 `RetrievalPlan` 从"5 个软槽 + 1 个声学串"升级为**分层意图对象**：

```
hard_constraints   { artist, song, language, region, instrumental }   # 离散就该精确——保留
soft_intent (自由) { goal:"从悲伤中走出/保持专注/派对升温",
                     trajectory:"由静到燃", avoid:"人声太满",
                     vibe:自由文本 }                                  # 不再枚举，交给向量/LLM
acoustic_query     已有的 HyDE                                         # 由 soft_intent 生成
```

- **硬约束**继续走 Cypher（歌手/歌名/语言/纯音乐——这些离散且需要精确，固定槽是对的）。
- **软意图**不再枚举，让 LLM 自由表达"为什么"，再映射到 M2D-CLAP 连续空间 + 用于精排重排序。
- mood/scenario 可以**保留为可选 hint**（命中词表就填，命中不了就进 soft_intent 自由文本），不再是"必须二选一塞进去"。

这样 "安静的日语歌" = `hard{language:Japanese} + soft{vibe:"quiet, low-dynamic, soft timbre"}`，两条腿都站住。

> 注：`schemas/query_plan.py` 文档串写"标签提取由确定性规则完成"，但代码（`analyze_intent` 直接 `plan.retrieval_plan.model_dump()`）是 LLM 填的。**文档与实现已不一致**，顺手修正。

---

## 3. 推荐算法评估

**结论：管线本身没有短板，不建议在这里花大力气。** 三锚精排（语义 0.45 / 声学 0.30 / 个性化 0.25）+ Graph Affinity（图距离 1/(1+d) + 四维 Jaccard）+ Thompson 探索槽 + MMR(λ=0.7) 多维去重，已经超出绝大多数个人项目，逼近报告里对标的 Spotify 四阶段。

**前沿对照（联网核实）：** 业界 2025–2026 的方向是**生成式检索 / Semantic IDs**（Spotify GLIDE 把推荐变成"指令跟随"，非习惯播放 +5.4%、新内容发现 +14.3%；TalkPlay 把推荐变成 next-token 预测）。这是值得**单开一条研究支线**做原型的方向，但它需要训练 + 量化码本，是重投入。**对当前体量，"升级表示层"的 ROI 远高于"换生成式检索引擎"。** 先把 §2 做了，生成式检索作为 v4 预研。

**小修：**
- `_dual_anchor_rerank` 是死代码（活跃的是 `_tri_anchor_rerank`），删；技术报告"双锚"措辞同步更新为"三锚"。
- 缺 OMAR 嵌入时退回纯语义——逻辑正确，但建议加一条监控指标（多少比例的候选缺声学锚），否则声学锚可能名存实亡。

---

## 4. 架构瘦身清单（具体、可执行、低风险）

| 项 | 证据 | 动作 | 收益 |
|---|---|---|---|
| **死依赖 ×7** | `streamlit/llama-index*/chromadb/spotipy/mcp/pandas/gdown` 全项目零 import | 从 `requirements.txt` 删除或移入 `legacy` extras | 镜像更小、安装更快、供应链面更窄 |
| **意图节点厂商分叉** | `analyze_intent` ~370 行，SGLang/DashScope 各一套 httpx 旁路；`multi_llm.py` 已抽象 provider | 抽出 `IntentPlanner` 类，把"渲染→调用→清洗 `<think>`/```json```→Pydantic"收敛成一处，按 provider 注册策略 | 可读性↑、新增 provider 不再改图节点、bug 面↓ |
| **死代码** | `_dual_anchor_rerank` vs `_tri_anchor_rerank` | 删旧 | 减歧义 |
| **静默兜底** | `analyze_intent` 异常 → 退 `general_chat`（`music_graph.py:626`） | 改为"可观测降级"：保留错误事件 + 走一条保守检索而非闲聊 | 求歌时不再被回闲聊（满帮2 教训：兜底会"自信地错") |
| **文档漂移** | 报告写"双锚/Neo4j Community"，代码是"三锚/Enterprise" | 文档与代码对齐 | 可信度 |

> 整体不臃肿（核心 14M、数据 16M），**真正的"重"在运行时**（模型权重 + 4 容器 + GPU），见 §6。

---

## 5. Harness / 工程质量与评测

**有的（好）：** GitHub Actions CI（ruff + pytest）、51 单测（去重/Token 预算/标签扩展/Schema）、55 条意图 eval（98.2%）、LangGraph MemorySaver checkpoint、GSSC Token 追踪。这套 Harness 对个人项目相当完整。

**缺口（关键）：评测在"测特性"，没在"测结果"。**
- 意图 eval 验证"路由器选了我标的标签"，是**半循环**的——它不会因为推荐变好而变好，也不会因为推荐变差而变差。这正是满帮2 反复踩的坑（判别器测特性 = 循环论证）。
- **要补的是结果导向评测：**
  1. **离线回放**：用你已采集的隐式反馈（like/save/skip/dislike）做留一/时间切分，问"系统当时会不会把用户后来点赞的歌排进 Top-K"（Recall@K / NDCG / skip-rate）。
  2. **在线**：埋点 skip 率、完播率、save 率、首次满意轮数，做 A/B（哪怕 N 很小）。
- 加一个**反作弊式自检**：任何"意图改动"必须在离线回放上不掉分，而不是只看 55 条分类准确率。

---

## 6. 部署便利性 + 可推广性

**现状门槛（这是"推广不出去"的主因）：**
- 4 容器（Neo4j + GraphZep + backend + frontend）+ **强制 GPU**（M2D/OMAR）+ ~1.6GB 模型权重 + HF 镜像下载。
- ⚠️ **许可证阻断点**：`docker-compose.yml` 用 `neo4j:2026-enterprise` + `NEO4J_ACCEPT_LICENSE_AGREEMENT: "eval"`——**eval 仅限非商用评估**。要推广/商用必须换 **Neo4j Community**（报告本来就写的 Community），或换嵌入式图库（Kùzu / 甚至退化为 SQLite+关系表）。

**建议：分层部署 Profile（让"想试一下"和"全量上"两类人各取所需）**

| Profile | 组件 | LLM | 向量 | 记忆 | GPU | 面向 |
|---|---|---|---|---|---|---|
| **Lite / Demo** | 单容器 | 云 API | sqlite-vec / LanceDB / FAISS（预算嵌入） | SQLite | 否 | 一键试玩、给人 star |
| **Standard** | backend+前端+Neo4j Community | 云 API | Neo4j Vector | Neo4j | 否（查询期不需要） | 自托管 |
| **Full** | 现状 4 容器 | 云/本地 | Neo4j Vector | GraphZep | 是（仅入库期） | 研究/重度 |

- 关键洞察：**M2D/OMAR 只在"入库"时需要 GPU**，查询期用的是**预算好的嵌入**。把"入库"和"在线服务"解耦后，在线服务可以**完全无 GPU**。这一步能立刻把部署门槛降一个数量级。
- 配一个 `docker compose --profile lite up -d` + 一个公网 Demo（你已有 B 站演示视频，再加可点的在线 Demo 转化率会高很多）。
- compose 里 `neo4j/12345678` 改为 `${NEO4J_PASSWORD}`。

---

## 7. SFT 与强化学习（"拿不到公开反馈数据"的实操路径）

**先纠一个前提：你不是没有反馈数据。** `UserMemoryManager` 已经在把 like(1.0)/save(0.8)/listened/skip/dislike 写进 Neo4j，`/api/user-event` 在收行为。这就是**隐式偏好信号**。按 ROI 排三级阶梯：

**① 蒸馏飞轮（最高 ROI，先做）—— 不需要任何"偏好"标注**
- 把云端大模型 Planner 的 `(用户输入+上下文 → MusicQueryPlan)` 成对日志落盘，攒几千条就是现成 SFT 语料。
- 微调本地 Qwen3-4B（你已有 `LOCAL_PLANNER_PROMPT` + SGLang 部署），目标是**让本地小模型逼近云端 Planner**。
- 收益：抹平本地/云端质量差 + 把意图延迟从 ~11.5s（API 单次）打到**亚秒级本地推理** + 离线/端侧可用 + 成本归零。

**② 隐式反馈 DPO/IPO（用你已有的信号）**
- like/save = 正样本，skip/dislike = 负样本，构造 `(query, song⁺, song⁻)` 偏好对。
- 对**精排打分器**或**Planner 的软意图生成**做离线 DPO/IPO（无需在线 RL，离线即可，省一个 reward model）。
- 这是"用真实人类信号"对齐，比纯 AI 反馈更值钱。

**③ RLAIF / LLM-as-Judge（补冷启动与长尾）**
- 对没有行为覆盖的意图边界，用强模型当裁判生成合成偏好对（rDPO / 自我批判），再 DPO。
- 适合"安静的日语歌"这类 §2 的表达力边界 case 做数据增广。

> 顺序很重要：**先①蒸馏（确定性收益）→ 再②隐式DPO（真实信号）→ 最后③RLAIF（长尾补全）**。不要一上来就上在线 RL，那是这套数据规模撑不起也不需要的复杂度。

---

## 8. 侧端 / 端侧部署需要做什么

**好消息：架构天然适合"端云协同"，因为重模型只在入库期用。**

| 层 | 端侧可行性 | 怎么做 |
|---|---|---|
| Planner（意图） | ✅ 高 | Qwen3-4B → 4-bit AWQ/GPTQ，经 **llama.cpp / MLC-LLM / ExecuTorch** 跑在手机 NPU/CPU；配合 §7① 的蒸馏 |
| 文本嵌入（查询期） | ✅ 高 | BGE-small / 量化版，端侧编码查询 |
| 召回（ANN） | ✅ 中 | 预算好的歌曲嵌入打包，端侧用 FAISS/usearch/sqlite-vec 本地检索 |
| M2D-CLAP / OMAR（入库期） | ❌ 留云 | 音频编码留服务端，端侧只消费"已编码"的嵌入 |
| GraphZep 长期记忆 | ⚠️ 简化 | 端侧用轻量本地记忆（SQLite + 周期性云同步） |

**端侧 MVP 路线：** 量化 Planner + 端侧文本嵌入 + 本地 ANN（离线歌库）→ 一个"断网也能按心情找本地歌"的离线模式。重活（入库、联网发现、长解释生成）按需回云。

---

## 9. 离成熟工业产品的差距 + 功能丰富化

**硬门槛（非技术，但决定能不能"推广/商用"）：**
1. **音乐版权**：`data/` 里的 `.ncm`（网易云加密）/ mp3 / flac + 下载链路是个人/研究用法，**无法商用分发**。商业化要么接正版 API（受限），要么转向"无版权音乐 / 播客 / 用户自有曲库"的赛道。这是第一性约束，越早想清楚越好。
2. **多用户与隔离**：现在是单用户 `local_admin`。要产品化需账号体系、数据隔离、配额/限流、成本护栏（LLM 调用是真金白银）。
3. **延迟**：意图 ~11.5s（缓存命中 2–4s）对产品偏高。§7① 蒸馏 + 召回缓存是主要抓手。

**功能丰富化（按"能感知的价值"排序）：**
- **主动追问（clarifying questions）**：模糊 query 先反问一句（"想要安静放松还是有点律动？"）——TalkPlay/对话式推荐的核心增量，且能直接喂 §7 的偏好数据。
- **解释可控 + 反馈回环**：每首歌"为什么推它"已有；加"不是这个味儿 → 一键微调方向"，把负反馈变成即时重排。
- **音乐旅程**（你已有 journey）做成"情绪曲线编辑器"是差异化卖点，值得打磨。
- **可分享的歌单/旅程**（生成卡片/链接）——天然的增长裂变点。
- **冷启动问卷 → 直接生成首个画像**，缩短"首次满意"路径。

---

## 10. 隐私与安全体检

- ✅ `.env` 已 gitignore、git 历史从未提交、无 `*.key/*.pem/secret` 入库。**密钥未泄露。**
- ⚠️ `docker-compose.yml` 硬编码 `neo4j/12345678`——改 `${NEO4J_PASSWORD}` 并在 README 提示首次改密。
- 🔧 多 provider key 散落环境变量，建议统一从 `.env` 单一入口加载（已基本如此），并在日志里确认无 key 打印（`multi_llm.py` 有 `print` 告警但不含 key，OK）。
- 📋 产品化前补：用户数据导出/删除（GDPR 式）、行为日志脱敏。

---

## 11. 优化规划（分阶段 Roadmap，按 ROI 排序）

### Phase 0 — 清场 + 立尺子（✅ 已完成 2026-06-18，零风险）
- [x] 删 9 个死依赖（streamlit/pandas/gdown/spotipy/mcp/llama-index×3/chromadb）→ `requirements.txt`。
- [x] 删死代码 `_dual_anchor_rerank`（126 行，且引用了已不存在的 settings 字段）→ `retrieval/hybrid_retrieval.py`。
- [x] 修文档漂移：双锚→三锚（4 处）、`query_plan.py` 三处误导性 docstring（"标签由确定性规则填充"实为 LLM 填充）。
- [x] compose 口令参数化 `${NEO4J_PASSWORD:-...}`（3 处）。
- [x] `analyze_intent` 异常兜底：`general_chat` → 可观测降级为保守 `vector_search`（不依赖 LLM，打 `_intent_degraded` 标记）。
- [x] **立尺子**：结果导向离线评测 `tests/eval/evaluate_outcomes.py` + 12 条 `outcome_test_cases.json`（测"返回的歌满不满足意图"而非"路由标签对不对"）。
- [x] 尺子自带尺子：`tests/unit/test_outcome_eval.py`（Phase 1 后 17 测，纯逻辑，CI 可跑，**已全绿**）；全量单测 72 passed。
- 📌 验证状态：完整栈已在 Docker Neo4j `:7687` 上执行；Phase 1 前后 outcome 均为 9/12=75.0%，失败项来自本地库缺歌 / NeteaseAPI 未启动，不是表示层回退。

### Phase 1 — 表示层升级（✅ 已完成 2026-06-18）⭐
- [x] **先决（尺子已就位）**：改动前基线 `tests/eval/results/outcome_eval_siliconflow_20260618_234317.json`，12 例通过 9、失败 3、自动通过率 75.0%。
- [x] **把标签透传进返回 song dict**：GraphRAG / SemanticSearch / Hybrid 标准化结果保留 `language`、`region`、`genres`、`moods`、`themes`、`scenarios`。
- [x] `RetrievalPlan` → 分层意图对象：新增 `hard_constraints` / `soft_intent` / `hints`，并与旧 `graph_*` 字段双向兼容；检索入口优先读取新对象，旧字段兜底。
- [x] 用尺子验证：改动后 `tests/eval/results/outcome_eval_siliconflow_20260618_235848.json` 仍为 9/12=75.0%；`language_match_min_ratio` 4/4 通过，`out_05` 的“日语歌为主”已自动判定且通过，人工核对项从 6 例降到 5 例。

### Phase 2 — 工程收敛 + 部署降门槛（进行中，2026-06-19）
- [x] 抽出 `agent/intent/IntentPlanner`：图节点只负责上下文与状态编排；DashScope、SGLang、本地 structured output 和通用 provider 调用集中到 adapters，JSON 清洗集中到 parsing。
- [x] 默认模型迁移到 DashScope：Planner/主模型/解释统一为已验证可用的 `qwen3.7-plus`；`qwen3.7-flash` 当前 DashScope 返回 `model_not_found`，不作为默认值；增加“明确求歌不能落入 general_chat”的确定性护栏。
- [x] Lite/Standard/Full Compose Profile 与 Windows 统一入口 `soultuner.ps1`；端口统一为前端 3003、API 8501、GraphZep 3100、Neo4j 7474/7687、SearxNG 8888。
- [x] `soultuner.ps1` 增加 `netease-start` / `netease-stop` / `netease-status`，把兼容音乐 API `:3000` 纳入日常运维入口。
- [x] 在线 API 与音频增强解耦：在线请求秒级写元数据并投递文件队列，Full Profile 的独立 ingest worker 再执行歌词标签与 M2D-CLAP/OMAR 向量提取；backend 不再声明强制 GPU。
- [x] `llms/multi_llm.py` 瘦身为兼容门面，核心逻辑拆到 `llms/registry.py`、`llms/chat_models.py`、`llms/native.py`。
- [x] 联网音乐 fallback 质量修复：自然语言 query 先规整为 Netease 友好查询；歌手-only 请求按 `artist_entities` 过滤；“最近新歌”直连 Netease `top/song?type=7` 华语新歌榜。
- [x] 增加 `python start.py --mock` / `.\soultuner.ps1 mock`，无需 LLM、Neo4j、GraphZep 或模型权重即可验证 Agent + SSE 主链。
- [x] 验证：84 个纯逻辑单测通过；mock 健康检查、SSE 歌曲与解释事件通过；DashScope outcome eval 在 NeteaseAPI 启动后提升到 12/12=100.0%，`not_degraded` 12/12。
- [x] P0 清噪音补齐：`doctor.py` 报告实际生效模型；`music_agent` 环境已补 pytest；`ChatOpenAI` 的 Qwen `extra_body` 改为显式参数，消除 LangChain warning；GraphZep 客户端增加 5 分钟不可用熔断缓存，服务离线时快速降级为空记忆。
- [x] 2026-06-19 标准档实测：Docker Neo4j `:7687` 有 1128 个 `Song` 节点；`doctor` 全绿（Neo4j/GraphZep/backend/frontend/SearxNG/NeteaseAPI 均在线）。注意 Compose 内 `soultuner-searxng` 与已有 `music_searxng` 同占 `:8888`，本机当前沿用已运行的 `music_searxng`。
- [ ] 部署公网 Demo，并补齐公开演示环境的监控、限流与成本预算。

### Phase 2.5 — 评测扩充 + Judge 地基（进行中，2026-06-19）
- [x] P1 评测扩充地基：`evaluate_outcomes` 支持 `--split smoke/dev/holdout/all`，报告记录 git sha、实际模型、Planner temperature 与关键 config；新增 50 条 dev + 20 条冻结 holdout，覆盖明确实体、语言、软意图、反向约束、多轮上下文、中英混说、时效性与兜底。
- [x] 2026-06-19 Docker 基线：smoke `11/12 = 91.7%`，dev `44/50 = 88.0%`。失败主要集中在明确歌手/歌名本地缺库或可播放链接不足、以及多轮“保留歌手但换情绪”约束；软意图仍有 35 个 manual_review，说明 P2 judge 是下一块地基。
- [x] P2 judge 地基：新增 `objective_soft_judge`，只读取歌曲客观字段（genre/genres/moods/scenarios/language/region/instrumental），不读取系统 explanation；当前作为可校准初判器，尚未批量替代 manual_review。
- [x] 软意图 judge 校准闭环：新增 `tests/eval/judge_gold/objective_soft_judge_gold.json`（20 条 pass/fail/skip 人工 seed）与 `python -m tests.eval.calibrate_soft_judge`，当前 exact accuracy 100.0%、coverage 95.0%；dev 中 9 个粗粒度软意图转为自动 judge，manual_review 从 35 降到 26，holdout 保持冻结不调。
- [ ] 下一步：扩展人工金标准到 50+ 条，并在校准通过后再接入 M2D-CLAP 文本相似度与 LLM judge；LLM judge 仍只能看 query + 歌曲客观属性，不看 explanation。

### A1 / R1 — 检索层重构（✅ 已完成 2026-06-20）
- [x] `retrieve()` 直接消费 `hard_constraints / soft_intent / hints`；legacy flat 字段仅保留旧调用兼容。
- [x] 图谱、稠密、BM25、个性化、冷启动五路始终并发；`intent_type` 只调整各路非零 RRF 权重，不再控制引擎开关。
- [x] 删除检索入口的关键词清理、纯音乐 graph-only、情绪词升级 hybrid、复合词语言推断等二次改判。
- [x] 平等合并改为加权 RRF；融合后只有 `hard_constraints + DISLIKES` 可以排除候选，genre/mood/scenario 仅作召回与排序软信号。
- [x] 验收：改前 dev `35/50 (70%)`、holdout `15/20 (75%)`；改后 dev `38/50 (76%)`、holdout `15/20 (75%)`；单测 `97 passed`。

### Phase 3 — 蒸馏飞轮 + 反馈对齐（2–4 周）
- [ ] 落盘云端 Planner I/O → SFT 本地 Qwen3-4B（意图延迟亚秒级）。
- [ ] 隐式反馈 DPO 精排打分器。

### Phase 4 — 端侧 MVP / 生成式检索预研（探索）
- [ ] 量化 Planner + 端侧 ANN 的离线模式 PoC。
- [ ] Semantic IDs / 生成式检索 v4 研究支线。

---

## 12. 参考（联网核对，2025–2026）

- Spotify GLIDE / Semantic IDs（生成式检索，instruction-following）：research.atspotify.com（2025-11）、arXiv 2603.17540、arXiv 2508.10478
- 综述《Music Recommendation with LLMs: Challenges, Opportunities, and Evaluation》：arXiv 2511.16478
- TalkPlay-Tools（对话式推荐 + 工具调用, NeurIPS 2025）：arXiv 2510.01698
- ⭐《Beyond Musical Descriptors: Extracting Preference-Bearing Intent in Music Queries》(Deezer, 2026-02)：arXiv 2602.12301 —— 与本报告 §2 直接呼应
- DPO/RLAIF with synthetic data（philschmid 2025；rDPO arXiv 2402.08005；What Matters in Data for DPO arXiv 2508.18312）
- 端侧：ExecuTorch / llama.cpp / MLC-LLM；AWQ/GPTQ 4-bit；On-Device LLMs 2026 综述

---

*本报告为静态审视产物，下一步进入逐项优化。建议从 Phase 0 + Phase 1 的"先立尺子再改表示"开始。*
