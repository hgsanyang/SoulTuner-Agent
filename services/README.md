# services/

服务层：记忆、反馈、排序策略、曲库知识、外部服务客户端。

## 记忆

| 文件 | 职责 |
|------|------|
| `memory_gateway.py` | **统一记忆入口**，所有读写都从这里走 |
| `memory_event_store.py` | 本地 append-only L0/L1/L2 账本（SQLite） |
| `memory_retriever.py` / `memory_semantic_scorer.py` | 召回与 BGE 相关性判定（模型不可用时 fail-closed） |
| `memory_consolidator.py` / `memory_links.py` / `memory_models.py` | 归并、关联、数据模型 |
| `profile_synthesizer.py` / `profile_views.py` | 用户画像合成与前端视图 |
| `policy_memory.py` | 排序策略相关的记忆快照 |
| `graphzep_client.py` | 可选 GraphZep 旁路（legacy，非默认） |

## 反馈与排序

| 文件 | 职责 |
|------|------|
| `feedback_store.py` | **正式事件存储**：SQLite + WAL，曝光/逐首/歌单/行为四张表 |
| `feedback_logger.py` | 写入入口 + JSONL 导出快照（训练与重放用） |
| `feedback_diagnostics.py` | 反馈质量诊断 |
| `ranking_learning.py` | 离线重放与排序策略候选学习 |
| `ranking_policy.py` | 运行时策略加载、候选提升、回滚 |
| `llm_feedback_logger.py` | LLM 规划与标签体系审计日志（不自动改标签） |
| `refinement_generator.py` | 出歌后生成微调建议 |

## 曲库与知识

| 文件 | 职责 |
|------|------|
| `music_knowledge_*.py` | 音乐知识的抓取、缓存、图谱化与向量索引 |
| `music_information_answer.py` | 资讯类提问的联网事实回答（查不到就直说查不到） |
| `catalog_enrichment.py` / `catalog_diagnostics.py` | 曲库补全与健康度诊断 |
| `library_quality.py` / `tag_policy.py` | 入库质量闸门与标签策略 |
| `ingest_queue.py` | 待入库队列 |
| `online_audio_flywheel.py` / `online_audio_retention.py` | 正反馈自动入库与音源留存 |
| `recommendation_knowledge_backfill.py` | 推荐结果的知识回填 |
| `runtime_mode.py` | 评测模式下的副作用开关（`side_effects_disabled`） |
| `teacher_log.py` | 蒸馏用 teacher 轨迹日志 |

## 存储分工：SQLite 是真值，JSONL 是快照

`feedback_store.py`（SQLite/WAL）是**正式事件存储**，`feedback_logger.py` 额外写一份 JSONL 作为**导出快照**，让既有的重放与评测脚本继续可用。

为什么必须是 SQLite 而不是纯 JSONL：曝光被**故意写两次**——歌一发出去就先落一条 provisional，整图跑完再用同一个 `exposure_id` 覆盖成终版。"同 id 后写覆盖先写"这件事，upsert 一行就能表达，append-only 日志只能靠全文件扫描，而且崩溃时会在尾部留下半行坏数据。

两条踩过的坑，改这层之前先读：

- **主键名写错会静默吞数据。** 写入侧用 `feedback_id`、存储侧按 `slate_feedback_id` 取，结果每行主键都是 `""`，`INSERT OR REPLACE` 把整张表压成一条。现在空主键会补 uuid 并打 warning，但根治办法是**改契约时两侧一起改**。
- **终写会覆盖预写的全部字段。** 终写忘了带 `context`，就把预写刚采集到的收听上下文（用户当地几点）抹掉了，而这个东西事后无法重建。现在 `log_exposure` 会把这类"只能当场采集"的字段**自动前向携带**，但新增同类字段时记得加进 `_CARRY_FORWARD_FIELDS`。

**依赖**：`config/`、`schemas/`
