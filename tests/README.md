# tests/

测试与评测模块。

## unit/

单元测试（pytest，当前 272 tests）：

| 文件 | 覆盖范围 |
|------|---------|
| `test_normalize_key.py` | 歌曲 key 标准化（中英文/特殊字符/空格处理） |
| `test_gssc_token_budget.py` | GSSC Token 预算分配与压缩 |
| `test_tag_expansion.py` | 标签别名映射与扩展 |
| `test_retrieval_fusion.py` | RRF 融合与硬过滤 |
| `test_post_recall_adjustments.py` | 个性化/新歌/冷门/过曝召回后修正 |
| `test_alignment_adapter_training.py` | 文搜音 linear adapter 训练方向与 split |
| `test_memory_gateway.py` | MemoryGateway 行为事件、歌单反馈、旁路召回 |
| `test_ranking_learning.py` | A3 离线重放与排序策略学习 |
| `test_ranking_policy_readiness.py` | A3 策略成熟度阶段、replay/promote 安全门 |
| `test_p7_smoke.py` | P7 离线 smoke 检查入口 |
| `test_ingest_queue.py` | 入库增强队列生命周期、失败任务重试 |
| `test_schema_validation.py` | Pydantic Schema 校验 |

## eval/

意图分类评测：

| 文件 | 说明 |
|------|------|
| `evaluate_intent.py` | 批量意图分类准确率评测脚本 |
| `intent_test_queries.json` | 55 条手工标注测试数据（覆盖 7 类意图） |

## 本地集成测试（不上传 GitHub）

以下文件保留在本地用于手动集成测试（依赖 Neo4j/可选记忆旁路/LLM），已添加到 `.gitignore`：

| 文件 | 说明 |
|------|------|
| `test_profile_synthesizer.py` | Profile Synthesizer 端到端集成验证 |
| `test_event_consistency.py` | Neo4j 行为事件状态一致性验证 |

## 运行测试

```bash
# 单元测试（CI 自动运行）
pytest tests/unit/ -v

# 发布/日常 smoke（不调用 LLM）
python scripts/p7_smoke.py
python scripts/p7_smoke.py --api-base http://localhost:8501

# 意图分类评测（需要 LLM API Key）
python -m tests.eval.evaluate_intent --provider siliconflow

# 集成测试（本地手动运行，需要 Neo4j；如启用 episodic 记忆则还需要对应旁路）
python tests/test_profile_synthesizer.py
python tests/test_event_consistency.py
```
