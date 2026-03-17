# agent/

Agent 核心模块 — LangGraph 工作流 + 主入口。

| 文件 | 职责 |
|------|------|
| `music_agent.py` | Agent 主类，对外暴露 `get_recommendations()` 接口 |
| `music_graph.py` | LangGraph StateGraph（11 节点 + 5 条件路由），定义完整推荐管线 |

**依赖**：`llms/`、`retrieval/`、`tools/`、`schemas/`、`config/`
