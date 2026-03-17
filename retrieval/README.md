# retrieval/

检索与记忆层 — 统一管理向量编码、图谱查询、混合检索、用户记忆。

| 文件 | 职责 |
|------|------|
| `audio_embedder.py` | M2D-CLAP 跨模态 + OMAR-RQ 声学双模型编码 |
| `portable_m2d.py` | M2D-CLAP 模型权重加载器 |
| `neo4j_client.py` | Neo4j 连接与 Cypher 查询封装 |
| `hybrid_retrieval.py` | 三路混合检索引擎（GraphRAG + Vector + Web），加权融合 + MMR 重排 |
| `gssc_context_builder.py` | GSSC Token 预算管理（GraphZep Facts + Chat History 配额分配） |
| `data_flywheel.py` | 数据飞轮（一键下载 → 编码 → 入库） |
| `music_journey.py` | 音乐旅程编排器（LLM 故事→情绪拆解→逐段检索 + SSE 推送） |
| `user_memory.py` | 用户偏好 Neo4j 图谱记忆读写 |
| `history.py` | 对话历史动态截断与格式化 |

**依赖**：`config/`、`llms/`、`schemas/`
