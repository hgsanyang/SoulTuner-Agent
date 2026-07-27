# data/sft/

Planner 蒸馏训练数据。格式为 JSONL，每行一个 ChatML `messages` 对象。

## 当前在用（Phase B）

| 文件 | 条数 | 说明 |
|------|------|------|
| `train_v2_chatml.jsonl` | 1272 | 训练集 |
| `eval_v2_chatml.jsonl` | 219 | 验证集 |
| `pilot_episodes_1200.jsonl` | 1128 | 原始 episode 种子（按比例生成） |

训练/验证是**按 episode 切分**而不是按单轮切分——同一段多轮对话的不同轮次必须留在同一侧，否则模型在验证集上看到的是自己训练过的上下文，分数会虚高。

## 已废弃

`planner_sft_data.jsonl`（600 条）、`train_chatml.jsonl`、`eval_chatml.jsonl` 是早期单轮种子，缺少真实对话上下文，**不要再用来训练**，保留仅为复现历史结果。
