# retrieval/

检索与排序层 — 统一管理向量编码、图谱查询、混合检索、精排管线、用户记忆。

| 文件 | 职责 |
|------|------|
| `hybrid_retrieval.py` | 三路内容召回（GraphRAG + MuQ/Vector + BM25 词法）+ Web 兜底，个性化/冷启动作为召回后加分/减分项，粗排 + 三锚精排 + MMR 重排 |
| `post_recall_adjustments.py` | 召回后统一加分/减分层：个性化、新鲜度、冷门度、过曝时间衰减惩罚，输出限幅 delta |
| `audio_embedder.py` | M2D-CLAP 跨模态 + OMAR-RQ 声学双模型编码（含文本 embedding 缓存） |
| `portable_m2d.py` | M2D-CLAP 模型权重加载器（第三方） |
| `cross_encoder_reranker.py` | Cross-Encoder 可选精排（默认关闭，实验性功能） |
| `neo4j_client.py` | Neo4j 连接与 Cypher 查询封装 |
| `gssc_context_builder.py` | GSSC Token 预算管理（GraphZep Facts + Chat History 配额分配 + LLM 压缩） |
| `data_flywheel.py` | 数据飞轮（联网下载 → 双模型编码 → Neo4j 入库） |
| `music_journey.py` | 音乐旅程编排器（LLM 故事→情绪拆解→逐段检索 + SSE 推送） |
| `user_memory.py` | 用户偏好 Neo4j 图谱记忆读写 |
| `history.py` | 对话历史动态截断与格式化 |

**依赖**：`config/`、`llms/`、`schemas/`
