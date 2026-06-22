# llms/

LLM 接口封装与 Prompt 模板。

| 文件 | 职责 |
|------|------|
| `registry.py` | Provider 注册表，默认 DashScope，兼容高级自定义 provider |
| `chat_models.py` | LangChain ChatModel 工厂（意图 / 解释 / 压缩） |
| `native.py` | LiteLLM 原生字符串调用器 |
| `multi_llm.py` | 向后兼容旧 import 的门面 |
| `prompts.py` | 所有 LLM Prompt 模板（Planner / Explainer / Chat / Memory / Journey） |

**依赖**：`config/`
