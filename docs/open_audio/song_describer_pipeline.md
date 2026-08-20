# Song Describer 开放音乐入库流水线

这条流水线把 Song Describer Dataset v1.0.0 的 **validated subset** 准备成 SoulTuner
可消费的曲目级 JSONL。默认只下载元数据；音频、生成清单和审计报告均存放在源码仓库外，
不会进入 Git，也不会进入 35B Planner 的训练或 sealed 评测流程。

## 为什么选它作为第一批公开可播音乐

- Zenodo 固定记录：`10072001`（DOI `10.5281/zenodo.10072001`）。
- validated subset 是 746 条人工复核描述、547 首音乐，适合验证“自然语言 → 音乐检索”。
- 每首曲目都包含原始 Creative Commons 许可证和归属文本。
- 音频包约 3.09 GiB，规模足够做完整体验，但不会像 MTG-Jamendo 全集那样占用数百 GiB。

上游把本数据集定位为评测数据，并不建议拿它训练模型。因此本项目只将它用于检索、评测和
带归属信息的公开演示播放；脚本会在每一行写入 `planner_training_allowed=false`。

## 一次性准备

PowerShell 示例：

```powershell
$env:SOULTUNER_OPEN_AUDIO_CACHE = "D:\SoulTunerData\song_describer"
python -m tools.data.song_describer_pipeline prepare
```

该命令会：

1. 从 Zenodo API 读取固定记录；
2. 下载四个小型元数据文件；
3. 按 Zenodo 公布的 MD5 与字节数校验；
4. 关联 captions、曲名/艺人/专辑、MTG-Jamendo tags 和逐曲许可证；
5. 生成 `artifacts/song_describer_validated.jsonl`、`source_inventory.json` 和
   `license_audit_validated.json`。

下载并只解压 validated subset 的音频：

```powershell
python -m tools.data.song_describer_pipeline prepare --download-audio --extract-audio
```

下载采用 `.part` 临时文件、支持 HTTP Range 续传，完成后校验 Zenodo 的 MD5；只有全部校验
通过才会改名为 `audio.zip`。脚本不会把整包内容盲目展开，只提取清单中的 547 首曲目，并为
每个音频记录 SHA-256。

快速演练前 5 首（仍需下载完整上游 ZIP）：

```powershell
python -m tools.data.song_describer_pipeline prepare --download-audio --extract-audio --max-tracks 5
```

更快的 5 首冒烟路径会使用 Zenodo HTTP Range，只读取 ZIP 中被选中的成员，不先下载整个 3.09 GiB：

```powershell
$env:MUSIC_DATA_ROOT = "D:\SoulTunerData"
$cache = "$env:MUSIC_DATA_ROOT\mtg_sample"
python -m tools.data.song_describer_pipeline prepare --cache-dir $cache --stream-audio --max-tracks 5
python -m tools.data.song_describer_pipeline verify --cache-dir $cache
python -m tools.data.queue_song_describer --cache-dir $cache --limit 5
```

最后一条默认只做入库预览。Neo4j、MuQ/M2D GPU Worker 和音频静态路由都就绪后，再显式提交：

```powershell
python -m tools.data.queue_song_describer --cache-dir $cache --limit 5 --commit
python scripts/ingest_worker.py
```

`queue_song_describer` 不另造一套入库逻辑：它先复核每个 MP3 的 SHA-256，随后复用
`_quick_ingest_to_neo4j` 与 `services.ingest_queue`，由现有 `ingest_worker` 生成 MuQ/M2D 向量。
同时补写逐曲许可证、captions、genre/mood 图谱关系。播放器地址沿用现有
`/static/mtg_audio/{filename}` 路由。

## 复核

```powershell
python -m tools.data.song_describer_pipeline verify
```

复核包含：上游元数据 MD5、manifest SHA-256、许可证 URL 可识别性，以及所有声明
`audio_available=true` 的本地音频 SHA-256。

## 输出约定

每首曲目包含：

- 稳定 ID：`sdd-<track_id>`；
- 曲名、艺人、专辑、发布日期、Jamendo 来源链接；
- 所有 validated captions；
- genre / instrument / mood-theme 标签；
- 相对音频路径、大小和 SHA-256；
- 数据集级 CC-BY-SA-4.0 以及逐曲原始 CC 许可证；
- `noncommercial_only`、`no_derivatives`、`share_alike` 等机器可读限制；
- 上游原始 attribution 与 license statement，供前端许可证卡片展示。

为播放器直接消费，manifest 同时在顶层提供 `attribution`、`license_id`、`license_url`、
`source_url`；审计报告的 `provenance_coverage` 必须显示这三类来源字段逐曲覆盖。

许可证注意事项：

- 逐曲许可证优先，不能只展示数据集级许可证。
- `NC` 曲目只能用于非商业演示。
- `ND` 曲目只能原样传播；在做裁切、变换或可能构成衍生作品的处理前需单独复核。
- 本流水线对 `ND` 采取强制保守策略：播放器直接返回 ZIP 中的原始 MP3 字节，禁止转码、
  裁剪、响度/增益修改、重混和重封装。抽取只是字节复制，manifest 的 SHA-256 绑定该文件。
- `SA` 曲目的衍生作品需按相同许可证共享。
- 音频用于播放时，页面必须可见地展示 `attribution_text` 和许可证链接。
- 本审计提供可追溯证据，不构成法律意见。

## 与后续 MuQ / Neo4j 入库的边界

本脚本只负责合法来源、逐曲许可、音频身份和检索输入清单。后续 GPU Worker 应读取 manifest，
在非商业部署约束下生成 512 维 MuQ-MuLan 向量，再写入 Neo4j/Qdrant。不要把 MuQ 权重、
`.npz` 向量或音频提交到 Git；它们应放在持久化数据盘或独立 ModelScope 数据集/模型仓库。
MuQ/M2D 可以在内存中解码音频并生成**非音频特征向量**，但不得把解码、重采样结果写回 MP3；
因此 ND 源文件在磁盘和播放链路中保持原字节不变。
