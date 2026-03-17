# services/

外部微服务客户端与业务 API 层。

| 文件 | 职责 |
|------|------|
| `graphzep_client.py` | GraphZep 记忆微服务 HTTP 客户端（写入对话 / 检索事实） |
| `music_api_service.py` | 纯音乐业务 API（不经过 LangGraph，供前端直接调用） |

**依赖**：`config/`、`tools/`
