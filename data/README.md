# data/

数据层 — 数据预处理与 Neo4j 入库管道。

## pipeline/

核心入库流水线：

| 文件 | 职责 |
|------|------|
| `ncm_pipeline.py` | 从网易云下载音频/封面/歌词 |
| `lyrics_analyzer.py` | LLM 歌词分析 → 自动生成 genre/mood/scenario 标签 |
| `ingest_to_neo4j.py` | 批量入库 Neo4j（元数据 + 歌词标签 + 双模型向量编码） |
| `ingest_tags_to_neo4j.py` | 仅写入标签到已有 Song 节点（增量标签补充） |
| `neo4j_schema_v2.py` | 初始化 Neo4j 向量索引与约束 + 数据集管理 CLI |

辅助工具：

| 文件 | 职责 |
|------|------|
| `migrate_cover_lrc_urls.py` | 封面/歌词 URL 迁移脚本 |
| `prepare_gemini_lrc_prompt.py` | 准备 Gemini 批量歌词分析 Prompt |
| `mtg_adapter.py` | MTG-Jamendo 数据集适配器 |
| `diag_embeddings.py` | 向量诊断（检查 Neo4j 中 embedding 维度与缺失情况） |
| `extract_missing_embeddings.py` | 补充缺失的音频向量编码 |

**使用方式**：
```bash
python data/pipeline/lyrics_analyzer.py       # 1. 标签提取
python data/pipeline/ingest_to_neo4j.py       # 2. 入库（含向量编码）
```
