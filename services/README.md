# services/

服务层：记忆网关、反馈日志、排序策略、外部微服务客户端。

| 文件 | 职责 |
|------|------|
| `memory_gateway.py` | 统一记忆入口（Neo4j 热路径 + GraphZep/Mem0 可选旁路） |
| `feedback_logger.py` | 曝光、行为、歌单级反馈 JSONL 日志 |
| `ranking_learning.py` | A3 离线重放与排序策略候选学习 |
| `ranking_policy.py` | 运行时排序策略加载、候选提升、回滚 |
| `graphzep_client.py` | 可选 GraphZep 记忆旁路 HTTP 客户端 |

**依赖**：`config/`
