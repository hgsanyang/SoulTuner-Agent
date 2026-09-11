# 本地音乐 MCP 原型

本入口复用主 API 的推荐、自然语言理解与记忆管道，不运行第二套 Planner。
它和暂时停用的 GraphZep MCP 没有依赖关系。当前仅提供 `recommend_music`，不是完整插件产品。

## 安装与启动

在仓库根目录创建独立 Python 3.12 环境，避免 MCP SDK 的依赖影响主 API：

```powershell
py -3.12 -m venv .venv-mcp-audit
.venv-mcp-audit/Scripts/python.exe -m pip install -r requirements-mcp.txt
.venv-mcp-audit/Scripts/python.exe -m integrations.music_mcp.server
```

最后一条启动的是 stdio 协议进程，应由支持 stdio 的 MCP 客户端启动并管理，不是网页服务器。
客户端配置使用该 Python 的绝对路径，参数为 `-m integrations.music_mcp.server`，工作目录设为仓库根目录。
不同客户端配置格式不同，本文件不自动修改任何客户端设置。

主 SoulTuner API 需要预先配置并启动；本入口不会启动数据库、下载模型、训练或启动 GraphZep。
默认 API 地址为 `http://127.0.0.1:8000`，可由进程环境变量 `SOULTUNER_MCP_API_URL` 指定其他回环端口。
只支持字面回环 HTTP 地址；不支持远程 API、代理、重定向或公开多用户 MCP 部署。

## 调用与边界

- 首轮传 `query`，例如“外面下暴雨，想听安静但不压抑的音乐”。
- 后续传原样返回的 `conversation_ref` 和自然表达，如“梦幻一点的歌曲有没有”；不要求推荐触发词。
- 同一引用复用服务器保存的历史和对话状态；引用不是用户身份，不能自行编造。进程重启后引用失效。
- 最多 32 个会话，空闲一小时清理，历史最多 24 条且有字符上限；同会话并发请求拒绝，不排队重放。
- 默认固定本地 profile `local_admin`（可通过进程环境 `SOULTUNER_MCP_PROFILE` 设置），采用 developer 模式，联网关闭。
- 此工具**不是只读工具**：主 API 可能保存开发会话、曝光等数据。没有开放任意 owner、记忆修改、下载、路径或 Cypher 参数。
- 失败、取消或结果未确认时不自动重试；该会话后续调用被阻止，避免盲目重复副作用。新建会话也不表示回滚旧请求。
- 返回歌曲及解释的结构化结果。MCP 工具当前等待完成再返回；CLI/Web 的逐首流式体验不等于 MCP 客户端已支持逐首展示。
- 音频/封面字段只是 API 返回的元数据，并不保证每个 MCP 宿主都能播放；未实现播放器、媒体资源托管或自动打开链接。

## 验证与未完成项

隔离环境安装 pytest 后运行 `python -m pytest tests/mcp -q`。
测试包含真实 SDK 内存传输、stdio 子进程握手、多轮上下文、并发拒绝、未确认重放拒绝和失败结果。
这些是协议和适配器测试，使用合成事件，**不是已完成真实模型、真实曲库、跨客户端或线上验收**。

后续需要现成模型/API 端点进行端到端验证；搜索、曲目详情、反馈、授权记忆工具和完整插件打包不在本原型中。
MCP SDK 官方参考：[Python SDK](https://github.com/modelcontextprotocol/python-sdk)、[工具定义](https://py.sdk.modelcontextprotocol.io/servers/tools/)、[stdio 运行](https://py.sdk.modelcontextprotocol.io/run/)。
