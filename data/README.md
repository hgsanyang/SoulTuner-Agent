# data/

数据层 — 数据预处理与 Neo4j 入库管道。

## pipeline/

| 文件 | 职责 |
|------|------|
| `ncm_pipeline.py` | 从网易云下载音频/封面/歌词 |
| `lyrics_analyzer.py` | LLM 歌词分析 → 自动生成 genre/mood/scenario 标签 |
| `ingest_to_neo4j.py` | 批量入库 Neo4j（元数据 + 向量编码） |
| `neo4j_schema_v2.py` | 初始化 Neo4j 向量索引与约束 |
| `migrate_cover_lrc_urls.py` | 封面/歌词 URL 迁移脚本 |
| `prepare_gemini_lrc_prompt.py` | 准备 Gemini 批量歌词分析 Prompt |
| `gemini_prompts/` | Gemini 批量任务的 Prompt 与结果数据 |

**使用方式**：
```bash
python data/pipeline/ncm_pipeline.py          # 1. 下载
python data/pipeline/lyrics_analyzer.py       # 2. 标签提取
python data/pipeline/ingest_to_neo4j.py       # 3. 入库
```
