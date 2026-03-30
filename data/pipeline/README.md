# Data Pipeline — 工具链脚本

> ⚠️ 本目录包含**一次性数据准备脚本**，不属于运行时代码。  
> 正常启动服务（`python start.py`）时不会加载此目录下的任何模块。

## 脚本说明

| 脚本 | 用途 | 运行时机 |
|------|------|----------|
| `ingest_to_neo4j.py` | 批量将本地音频目录入库 Neo4j | 初始化知识图谱时运行一次 |
| `ingest_tags_to_neo4j.py` | 批量导入标签数据到 Neo4j | 补充标签时运行 |
| `ncm_pipeline.py` | 网易云歌单爬取 + 下载 | 扩充本地曲库时运行 |
| `mtg_adapter.py` | MTG-Jamendo 数据集适配 | 导入 MTG 数据集时运行一次 |
| `lyrics_analyzer.py` | 歌词情感/主题分析 | 歌词标注时运行 |
| `neo4j_schema_v2.py` | V2 图谱 Schema 迁移 | Schema 升级时运行一次 |
| `migrate_cover_lrc_urls.py` | 封面/歌词 URL 字段迁移 | 字段重构时运行一次 |
| `extract_missing_embeddings.py` | 补充缺失的向量嵌入 | 检测到缺失嵌入时运行 |
| `diag_embeddings.py` | 诊断嵌入完整性 | 调试用 |
| `prepare_gemini_lrc_prompt.py` | 生成 Gemini 批量打标 Prompt | 批量标注时运行 |

## 子目录

- `gemini_prompts/` — Gemini 批量打标的 Prompt 文件
- `mtg_metadata/` — MTG 数据集元数据（.tsv，已 gitignore）

## 运行方式

```bash
# 所有脚本需从项目根目录运行，确保 PYTHONPATH 正确
cd /path/to/Muisc-Research
python data/pipeline/ingest_to_neo4j.py
```
