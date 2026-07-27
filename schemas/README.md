# schemas/

Pydantic 契约。这一层是"约好的话怎么说"——跨模块传的数据长什么样，都在这里定死。

| 文件 | 职责 |
|------|------|
| `music_state.py` | `MusicAgentState` — LangGraph 全局状态 |
| `query_plan.py` | `MusicQueryPlan` / `RetrievalPlan` — Planner 的 LLM 结构化输出 |
| `planner_decision.py` | `PlannerDecisionV2` — 蒸馏用的紧凑决策格式 |
| `planner_decision_v3.py` | V3 契约（**RC，尚未冻结**）：把"用户要什么"和"该调哪条检索路"拆开 |
| `tool_plan.py` | ToolPlan — 决策编译成的可执行调用计划 |
| `dialog_state.py` | 多轮对话状态与偏好继承 |
| `refinement.py` | 出歌之后由 LLM 生成的微调建议 chips |
| `feedback_events.py` | 反馈与曝光事件契约（口味 / 语境双通道、曝光记账、收听上下文） |
| `agent_context.py` | 组装给 Agent 的上下文与习惯卡（**已设计、尚未接入运行链路**） |

## 两条约定

**契约变更必须同时改写入方和存储方。** `feedback_events.py` 里声明的字段，只有真的被写进去才算数——曾经出现过 schema 声明了 `overall`/`propensity`，写入侧从没填过，而测试用手写 fixture 把它们喂进去，于是测试全绿、线上全是 None。改契约时问一句：**生产链路上真的有谁在写这个字段吗？**

**标记清楚哪些是"设计好但没接线"的。** 上表里 `agent_context.py` 和 `planner_decision_v3.py` 目前没有运行时消费者，只有迁移脚本引用。放在这里是有意的，但必须标出来，免得读代码的人以为它已经在跑。
