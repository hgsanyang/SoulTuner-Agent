# 🎵 数据处理管线 (Data Pipeline)

本目录包含音乐推荐系统的**完整数据入库工具链**：从网易云原始加密文件 → AI 标签生成 → Neo4j 图谱入库 → 音频向量提取。

---

## 📁 目录结构

```
data/pipeline/
├── ncm_pipeline.py                  # Step 1: NCM 格式解密转换
├── prepare_gemini_lrc_prompt.py     # Step 2: 生成 Gemini 标签提示词
├── ingest_to_neo4j.py               # Step 3: Neo4j 图谱入库（标签 + 向量）
├── extract_missing_embeddings.py    # Step 4: 补提取缺失向量（精准定向）
├── neo4j_schema_v2.py               # 辅助: 图谱 Schema 管理与数据集清理
├── mtg_adapter.py                   # 辅助: MTG 数据集标签适配器
├── ingest_progress.json             # 自动生成: 断点续传进度文件
├── gemini_prompts/                  # 自动生成: Gemini 提示词与结果
│   ├── gemini_prompt_batch_*.txt    # AI 标签提示词批次文件
│   └── gemini_result.json           # Gemini 返回的标签 JSON（主文件）
└── README.md                        # 本文档
```

---

## 🔄 完整处理流程

```
网易云 .ncm 文件
       │
       ▼  Step 1: ncm_pipeline.py
  解密 → mp3/flac + 封面 + 歌词 + 元数据
       │
       ▼  Step 2: prepare_gemini_lrc_prompt.py
  歌词打包 → gemini_prompt_batch_*.txt
       │
       ▼  (手动) 发给 Gemini 网页版
  AI 标签分析 → gemini_result.json
       │
       ▼  Step 3: ingest_to_neo4j.py --skip-embeddings
  标签入库 → Neo4j (Song + Mood/Theme/Scenario/Genre/...)
       │
       ▼  Step 4: extract_missing_embeddings.py 或 ingest_to_neo4j.py --update-embeddings
  音频向量 → Neo4j (m2d2_embedding + omar_embedding)
```

---

## Step 1️⃣ — NCM 格式解密转换

将网易云下载的 `.ncm` 加密文件转换为标准 `mp3/flac`，同时提取封面、歌词、元数据。

```bash
python data/pipeline/ncm_pipeline.py
```

**输入目录**: `data/raw_ncm/`（将 `.ncm` 和同名 `.lrc` 文件放入此目录）

**输出目录**: `data/processed_audio/`
- `audio/` — 解密后的 mp3/flac 音频
- `covers/` — 专辑封面大图
- `lyrics/` — 歌词 `.lrc` 文件
- `metadata/` — 每首歌的 `_meta.json`（含网易云原始 musicId、歌手、专辑等）

**防重机制**: 已存在的音频文件会自动跳过（终端显示 `⏭️ 已存在`）。

---

## Step 2️⃣ — 生成 Gemini 标签提示词

将歌词和元数据打包成批次提示词，供 Gemini 网页版进行多维度标签分析。

```bash
# 正常模式：只为未处理过的新歌生成提示词
python data/pipeline/prepare_gemini_lrc_prompt.py

# 全量重标注模式：忽略历史记录，为所有歌曲生成提示词
python data/pipeline/prepare_gemini_lrc_prompt.py --force-all
```

**输出**: `data/pipeline/gemini_prompts/gemini_prompt_batch_*.txt`（每批 100 首）

**防重机制**: 读取 `gemini_result.json` 中已有的歌曲名单，自动跳过已标注的歌。

### 手动提交给 Gemini

1. 打开 [Gemini 网页版](https://gemini.google.com/)
2. 将每个 `batch_*.txt` 的**全部内容**粘贴发送
3. Gemini 返回纯 JSON 数组，将所有批次的结果**合并**为一个大数组
4. 保存到 `data/pipeline/gemini_prompts/gemini_result.json`

### Gemini 提取的标签维度

| 维度 | 字段名 | 示例 | Neo4j 关系 |
|------|--------|------|-----------|
| 情绪 | `moods` | `["Melancholy", "Healing"]` | `HAS_MOOD` |
| 主题 | `themes` | `["Love", "Life"]` | `HAS_THEME` |
| 场景 | `scenarios` | `["Late Night", "Driving"]` | `FITS_SCENARIO` |
| 氛围 | `vibe` | `"Indie"` | Song 节点属性 |
| 语言 | `language` | `"Chinese"` | `HAS_LANGUAGE` |
| 地区 | `region` | `"Mainland China"` | `IN_REGION` |
| 流派 | `genre` | `["Folk", "Indie"]` | `BELONGS_TO_GENRE` |

---

## Step 3️⃣ — Neo4j 图谱入库

将元数据和 Gemini 标签写入 Neo4j，建立图谱节点和关系边。

```bash
# 只写标签（秒级完成，8 线程并发，不需要 GPU）
python data/pipeline/ingest_to_neo4j.py --skip-embeddings

# 标签 + 向量一步到位（需要 GPU，耗时较长）
python data/pipeline/ingest_to_neo4j.py

# 强制全量重跑（忽略进度文件，覆盖更新旧标签）
python data/pipeline/ingest_to_neo4j.py --skip-embeddings --force

# 只补充向量（已入库歌曲补提取 M2D2/OMAR 向量）
python data/pipeline/ingest_to_neo4j.py --update-embeddings

# 清空进度文件
python data/pipeline/ingest_to_neo4j.py --reset-progress
```

**标签覆盖更新**: 入库时会先 DELETE 旧的标签关系边（Mood/Theme/Scenario/Genre/Language/Region），再 MERGE 新边，确保标签总是最新的。

**向量安全保护**: `--skip-embeddings` 模式下绝不会触碰已有的 `m2d2_embedding` / `omar_embedding` 属性。

**断点续传**: `ingest_progress.json` 记录每首歌的处理状态（`processing` / `done_meta` / `done_full`），中断后重跑自动跳过已完成的歌。

---

## Step 4️⃣ — 补提取缺失向量（精准定向）

直接查询 Neo4j 中缺少 `m2d2_embedding` 或 `omar_embedding` 的歌曲，只对这些歌提取向量。

```bash
# 预览：只列出缺向量的歌曲（不做任何修改）
python data/pipeline/extract_missing_embeddings.py --dry-run

# 正式提取：为缺向量的歌曲补充 M2D2+OMAR 向量
python data/pipeline/extract_missing_embeddings.py
```

**需要 GPU**: M2D-CLAP (768 维) + OMAR-RQ (768 维) 双模型推理。

**安全中断**: Ctrl+C 中断后重跑，已写入 Neo4j 的歌不会重复处理。

---

## 🛠️ 辅助工具

### 数据集管理 (neo4j_schema_v2.py)

```bash
# 查看所有数据集及歌曲数量
python data/pipeline/neo4j_schema_v2.py --list-datasets

# 按数据集名称删除（需输入 YES 确认）
python data/pipeline/neo4j_schema_v2.py --delete-dataset mtg

# 创建/验证向量索引
python data/pipeline/neo4j_schema_v2.py --verify
```

---

## ⚡ 快速参考：新歌入库一条龙

```bash
# 1. 把 .ncm 文件放入 data/raw_ncm/，然后解密
python data/pipeline/ncm_pipeline.py

# 2. 生成 Gemini 提示词（只处理新歌）
python data/pipeline/prepare_gemini_lrc_prompt.py

# 3. 手动发给 Gemini，保存结果到 gemini_result.json

# 4. 标签入库（秒级）
python data/pipeline/ingest_to_neo4j.py --skip-embeddings

# 5. 向量补提取（需要 GPU）
python data/pipeline/extract_missing_embeddings.py
```
