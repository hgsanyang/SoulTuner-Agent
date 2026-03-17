# tools/

LangGraph 工具层 — Agent 可调用的原子操作。

| 文件 | 职责 |
|------|------|
| `graphrag_search.py` | 知识图谱检索（Neo4j Cypher + 五维度标签过滤 + 200+ 别名映射） |
| `semantic_search.py` | M2D-CLAP + OMAR-RQ 向量 KNN 检索 |
| `music_tools.py` | 基础音乐搜索与推荐（Tavily 在线搜索 + 本地 JSON 兜底） |
| `music_fetch_tool.py` | 在线歌曲试听链接抓取 |
| `acquire_music.py` | 联网获取音乐并入库（数据飞轮触发器） |
| `web_search_aggregator.py` | 联网搜索聚合（SearxNG + Tavily + 智谱 WebSearch） |

**依赖**：`retrieval/`、`config/`
