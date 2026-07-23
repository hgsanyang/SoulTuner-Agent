# agent/

Agent 核心模块 — LangGraph 工作流 + 主入口。

| 文件 | 职责 |
|------|------|
| `music_agent.py` | Agent 主类，对外暴露 `get_recommendations()` 接口 |
| `music_graph.py` | LangGraph StateGraph（15 节点 + 4 条件路由），定义完整推荐管线 |
| `catalog_gap.py` | 本地曲库缺口判定：缺歌/缺元数据时才走联网兜底 |
| `web_discovery.py` | 证据驱动的联网补充路线 |

**依赖**：`llms/`、`retrieval/`、`tools/`、`schemas/`、`config/`
