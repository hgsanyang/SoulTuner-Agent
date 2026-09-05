# 1,806 首开放音乐曲库与 ModelScope 打包

## 已盘点的数据

- Song Describer/Jamendo：706 首，约 3.111 GiB。
- FMA Small 平衡扩展：1,100 首（Rock 300、Hip-Hop 220、Pop 220、Folk 160、
  International 120、Electronic 80）。选择过程排除了 Experimental 和 Instrumental
  顶层类别，并在类别内优先保留具有人声线索的曲目。
- 合并后的公开演示曲库：1,806 首；`bundle_audit.json` 必须同时显示 1,806 个音频文件、
  1,806 个封面文件和逐曲来源/许可证覆盖。
- 上游归档：Zenodo `audio.zip`，MD5 `2126b8facfe9468cf806c6154e09bbe5`，已匹配。
- 完整清单：1106 条 captions，逐曲音频许可证、归属和 SHA-256 均有记录。

## 封面来源与许可边界

Jamendo 官方 tracks/albums API 文档给出的封面地址格式是：

```text
https://usercontent.jamendo.com/?type=album&id=<album_id>&width=600&trackid=<track_id>
```

脚本使用这个固定 600px 端点取得 JPEG，只验证媒体类型、JPEG 尺寸和 SHA-256，不在本地
转码、裁剪或重编码。请求不存在的 album ID 可能返回 Jamendo 默认 JPEG，也可能由边缘节点
返回 GIF 错误资源；脚本会尝试记录默认 JPEG 的 SHA-256，并始终以 MIME + JPEG 尺寸双重
校验兜底。相同默认 SHA、非 JPEG 或无有效尺寸都会判定为“源端无封面”。

重要：Song Describer 的逐曲 CC 许可证证明的是发布包内的音频，不应自动扩展到 Jamendo
专辑美术作品。Jamendo API 条款还要求创作者/Jamendo 归属、逐内容回链，并限制专门的离线
缓存服务。因此：

- `runtime_cache/*.jpg` 只用于可用性审计和合理运行时缓存；
- ModelScope 公共曲库不打包或再分发这些 JPEG；
- `cover_url` 保留官方远程展示 URL；
- `cover_attribution` 和 `cover_source_page_url` 必须同时展示；
- 每首曲目都有 SoulTuner 稳定生成的本地 SVG 回退封面；
- 源端无图、返回默认图或请求失败时，直接使用本地 SVG。

Jamendo API 条款：https://devportal.jamendo.com/api_terms_of_use

## 准备 Song Describer 706 首封面审计

```powershell
$cache = "C:\Users\sanyang\sanyangworkspace\music_recommendation\data\mtg_sample"
$manifest = "$cache\artifacts\song_describer_full.jsonl"
python -m tools.data.song_describer_public_bundle covers `
  --audio-manifest $manifest `
  --cover-root "$cache\covers" `
  --workers 8
```

该命令可重复运行：已有运行时 JPEG 会复核后复用；`--refresh` 才强制重新请求。

产物：

- `covers/cover_manifest_full.jsonl`
- `covers/cover_audit_full.json`
- `covers/runtime_cache/*.jpg`（禁止进入公共包）
- `covers/placeholders/*.svg`（可进入公共包，CC-BY-4.0）

## 构建 ModelScope 公共曲库目录

```powershell
python -m tools.data.song_describer_public_bundle bundle `
  --audio-manifest "$cache\artifacts\song_describer_full.jsonl" `
  --cover-manifest "$cache\covers\cover_manifest_full.jsonl" `
  --cache-dir $cache `
  --output-dir "C:\Users\sanyang\sanyangworkspace\music_recommendation\data\modelscope_public_library" `
  --mode hardlink
```

`hardlink` 不复制第二份 3.1 GiB 音频数据；跨盘失败时自动退回普通复制。正式移动到其他磁盘
或上传工具不识别硬链接时可用 `--mode copy`。

公共目录包含：

- `catalog.jsonl`：音频、captions、逐曲许可证、封面远程 URL 与回退路径；
- `audio/`：原始 MP3 字节，不转码、不裁剪；
- `covers/placeholders/`：稳定生成的 SVG；
- `bundle_audit.json`：体积、覆盖率、物化方式与 catalog SHA-256；
- `README.md`：ModelScope 数据集卡片草稿。

## 构建并合并 FMA 平衡扩展

FMA Small 上游音频包约 7.2 GiB。脚本读取 ZIP 中央目录并只下载确定性选中的 1,100 个
30 秒 MP3，不需要先展开完整归档：

```powershell
$fma = "C:\Users\sanyang\sanyangworkspace\music_recommendation\data\fma_open_expansion"
python -m tools.data.fma_balanced_pipeline --root $fma --workers 8

$base = "C:\Users\sanyang\sanyangworkspace\music_recommendation\data\modelscope_public_library"
$merged = "C:\Users\sanyang\sanyangworkspace\music_recommendation\data\modelscope_public_library_v2"
python -m tools.data.merge_public_audio_bundles `
  --base $base `
  --expansion $fma `
  --output $merged
```

最终上传与创空间物化使用合并目录 `modelscope_public_library_v2`。首次成功物化后，文件位于
创空间持久目录，休眠唤醒只复核清单并复用现有音频，不重新下载整个曲库。GPU 档为 1,806
首歌曲生成 MuQ 512 维与 OMAR 1024 维向量；M2D 768 维只属于显式纯 CPU 兼容档。

本流程只构建本地产物，不会自行上传或触发创空间部署。
