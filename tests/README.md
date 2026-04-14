# tests/

测试与评测模块。

## unit/

单元测试（pytest，51 tests）：

| 文件 | 覆盖范围 |
|------|---------|
| `test_normalize_key.py` | 歌曲 key 标准化（中英文/特殊字符/空格处理） |
| `test_gssc_token_budget.py` | GSSC Token 预算分配与压缩 |
| `test_tag_expansion.py` | 标签别名映射与扩展 |
| `test_merge_dedup.py` | 多路检索结果合并去重 |
| `test_schema_validation.py` | Pydantic Schema 校验 |

## eval/

意图分类评测：

| 文件 | 说明 |
|------|------|
| `evaluate_intent.py` | 批量意图分类准确率评测脚本 |
| `intent_test_queries.json` | 55 条手工标注测试数据（覆盖 7 类意图） |

## 其他

| 文件 | 说明 |
|------|------|
| `test_event_consistency.py` | SSE 事件格式一致性验证 |

## 运行测试

```bash
# 单元测试
pytest tests/unit/ -v

# 意图分类评测
python -m tests.eval.evaluate_intent --provider siliconflow
```
