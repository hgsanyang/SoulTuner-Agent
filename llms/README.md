# llms/

LLM 接口封装与 Prompt 模板。

| 文件 | 职责 |
|------|------|
| `base.py` | LLM 基类接口定义 |
| `multi_llm.py` | 多 LLM 提供商统一接入（SiliconFlow / Ollama / vLLM / DashScope） |
| `siliconflow_llm.py` | SiliconFlow API 专用封装 |
| `prompts.py` | 所有 LLM Prompt 模板（Planner / Explainer / Chat / Memory / Journey） |

**依赖**：`config/`
