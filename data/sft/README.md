# data/sft/

Planner 蒸馏训练数据。格式为 JSONL，每行一个 ChatML `messages` 对象。

## 当前在用（Planner V3）

| 文件 | 条数 | 说明 |
|------|------|------|
| `train_v3_chatml.jsonl` | 1289 | 冻结训练集 |
| `eval_v3_chatml.jsonl` | 226 | 冻结验证集 |
| `pilot_episodes_1200.jsonl` | 1128 | 原始 episode 种子（按比例生成） |

训练/验证是**按 episode 切分**而不是按单轮切分——同一段多轮对话的不同轮次必须留在同一侧，否则模型在验证集上看到的是自己训练过的上下文，分数会虚高。

## 已废弃

`planner_sft_data.jsonl`（600 条）、`train_chatml.jsonl`、`eval_chatml.jsonl` 是早期单轮种子，缺少真实对话上下文，**不要再用来训练**，保留仅为复现历史结果。

## Planner V3 闸门数据

- `reviews/v3_ambiguous_review.jsonl`：71 条 V2→V3 争议样本的逐条裁决，保留原始 query、原决策、修订决策和理由。
- `reviews/v3_resolved_trainable.jsonl`：64 条人工复核后解除隔离的争议样本。
- `curated_v3_gap_seeds.jsonl`：conversation、acquisition、library 各 8 条人工策划契约种子。
- 7 条过度澄清样本由强 teacher 重新采集后全部通过严格 V3 schema；私有结果不进入 Git。

最终冻结集共 1515 条，按 seed family 切分为 1289/226；episode、seed family、query 三种交叉泄漏均为 0。

这些文件都带有 `source_type`、`data_purpose` 和 `training_eligible`。审计输入由 SHA-256 冻结，不允许在源文件变化后继续机械套用旧裁决。

重新生成：

```powershell
python -m data.sft.review_v3_ambiguous --source data/teacher/private/ambiguous_samples.jsonl
python -m data.sft.generate_v3_gap_seeds
```

`read_library` 与 `stage_ingest` 从 ToolPlan `1.1` 起可用。后者只生成 shadow 预演，不写队列；主图的 `TOOL_PLAN_EXECUTION_ENABLED` 仍保持关闭。
