# api/

FastAPI 后端接口层 — SSE 流式推荐 + 运行时设置 + 用户画像。

| 文件 | 职责 |
|------|------|
| `server.py` | 主服务（SSE 流式推荐 + Settings API + 静态音频服务 + 组件预热） |
| `user_profile.py` | 用户画像 API（`GET/POST /api/user-profile`，Neo4j + GraphZep 双写） |
| `start_server.py` | 独立启动脚本（开发调试用） |

## 启动方式

```bash
# 推荐：一键启动全栈（Backend + Frontend + GraphZep + SearxNG）
python startup_all.py

# 或仅启动后端
python start.py --mode api
# 后端运行在 http://localhost:8501
```

## 核心端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/recommendations/stream` | POST | SSE 流式推荐（主接口） |
| `/api/settings` | GET/POST | 运行时设置读写 |
| `/api/user-profile` | GET/POST | 用户画像读写 |
| `/api/music-journey/stream` | POST | 音乐旅程 SSE |
| `/api/library/*` | GET/POST/DELETE | 曲库管理（待入库/我的曲库） |
| `/audio/{filename}` | GET | 静态音频文件服务 |

## SSE 事件类型

| 事件 | 说明 |
|------|------|
| `start` | 开始处理 |
| `thinking` | Agent 思考中 |
| `response` | LLM 文本流式输出 |
| `song` | 单首歌曲推荐数据 |
| `complete` | 请求完成 |
| `error` | 错误信息 |

**依赖**：`agent/`、`config/`、`retrieval/`
