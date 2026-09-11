# 共用主 API 的本地 Gradio 展示入口

`integrations/gradio_app.py` 是主项目的薄展示端，不导入展示版的 Planner、目录检索或记忆算法。
主 API 负责模型选择、意图理解、记忆、检索、策略守卫及解释；本入口只处理会话状态与事件呈现。

## 使用

先按主项目配置启动 API 与已有依赖。以下操作不会替你启动模型、数据库或 GraphZep：

```powershell
py -3.12 -m venv .venv-ui-audit
.venv-ui-audit/Scripts/python.exe -m pip install -r requirements-local-ui.txt
.venv-ui-audit/Scripts/python.exe -m integrations.gradio_app
```

在终端输出的本地地址打开页面。UI 只监听 `127.0.0.1`，不启用分享链接。
默认连接 `http://127.0.0.1:8000`；可通过 `SOULTUNER_LOCAL_API_URL` 设置其他字面回环 HTTP 端口。
目前只供本地单用户开发展示，固定 developer 模式与 `local_admin`，联网补充关闭。
developer 模式可能写开发会话数据，不等于只读；不要将此入口公开映射为多用户服务。

## 行为

- 首首歌曲到达即刷新本轮歌单，解释随后追加；推荐更新与普通聊天由 API 决定，不添加触发词。
- 聊天没有推荐事件时保留旧歌单；明确启动推荐但无候选时清空，避免把旧歌曲伪装成本轮结果。
- 历史与 dialog_state 随同一会话传给主 API；历史有界，新建会话换 ID。刷新/进程退出后的恢复不在本入口保证范围内。
- 上一首/下一首可切换播放器。只播放 API 提供的 HTTP(S) 音频地址（audio_url 或 preview_url），需用户点击播放；不复制或公开本机任意音频路径。
- HTML 转义曲目文本，拒绝脚本 URL、file URL 和带凭据媒体地址；不在服务端替用户下载音频或封面。
- 错误或取消不能冒充成功，未确认请求不会自动重试；会话被标记为不可继续，用户可新建会话。底层连接关闭不保证已经发出的远端 GPU 工作已停止。

## 验收边界

纯适配器回归验证首歌先到、多轮上下文、空候选清空、普通聊天保留、取消关闭、并发拒绝和媒体转义。
这不等于真实模型质量/延迟、浏览器播放兼容性或线上验收。

旧 `deploy/modelscope_space/app.py` 保留原行为，不自动切换、不部署。迁移线上前仍需相同 API 配置、相同 catalog 的真实对照，并处理公开身份、持久存储与音频可达性。
此入口没有移植原展示版的反馈/记忆编辑 UI，不代表完整创空间产品迁移结束。
